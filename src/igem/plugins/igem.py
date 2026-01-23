import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward

from igem.plugins.base import BaseGEMPlugin


class IGEMPlugin(BaseGEMPlugin):
    """
    I-GEM: fixed-budget dual PGD projector in the LoRA adapter subspace.

    Implements Algorithm 1. The dual of Eq. (2) is

        min_lambda  0.5 lambda^T (G G^T) lambda + (G g)^T lambda,  lambda >= 0

    which is solved with K projected-gradient steps instead of a QP solve,
    then mapped back with g_tilde = g + G^T lambda.
    """

    def __init__(
        self,
        patterns_per_exp: int,
        pgd_iterations: int,
        lr: float,
        use_adaptive_lr: bool,
        use_warm_start: bool,
        memory_strength: float,
        proj_interval: int,
        n_experiences: int,
        memory_size: int,
        param_space: str = "adapter",
        step_size_c: float = 0.9,
        power_iterations: int = 3,
        normalize_rows: bool = True,
        proj_metric=None,
    ):
        super().__init__(
            patterns_per_exp=patterns_per_exp,
            memory_strength=memory_strength,
            proj_interval=proj_interval,
            n_experiences=n_experiences,
            memory_size=memory_size,
            param_space=param_space,
            proj_metric=proj_metric,
        )
        self.pgd_iterations = pgd_iterations
        self.lr = lr
        self.use_adaptive_lr = use_adaptive_lr
        self.use_warm_start = use_warm_start
        self.step_size_c = step_size_c
        self.power_iterations = power_iterations
        self.normalize_rows = normalize_rows

        self.memory_x = {}
        self.memory_y = {}
        self.memory_tid = {}

        self.G = torch.empty(0)
        self.GGT = torch.empty(0)

        # dual iterate, carried across minibatches within a task
        self.v = None
        # persistent power-iteration vector for the spectral norm estimate
        self._power_vec = None

    def _has_memory(self) -> bool:
        return bool(self.memory_x)

    def _compute_reference_gradients(self, strategy) -> Tensor:
        rows = []

        for t in range(strategy.clock.train_exp_counter):
            strategy.optimizer.zero_grad()

            xref = self.memory_x[t].to(strategy.device)
            yref = self.memory_y[t].to(strategy.device)
            tid = self.memory_tid[t].to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            strategy._criterion(out, yref).backward()

            rows.append(self._flat_grad(strategy))

        G = torch.stack(rows, dim=0)

        if self.normalize_rows:
            # Positive row scaling leaves the feasible set of Eq. (2) unchanged
            # but stops a single large constraint dominating the dual geometry.
            G = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)

        self.G = G
        self.GGT = G @ G.T
        return self.G

    def _step_size(self) -> float:
        """eta = c / sigma_max(G G^T), with sigma_max from power iteration."""
        if not self.use_adaptive_lr:
            return self.lr

        M = self.GGT
        v = self._power_vec
        if v is None or v.shape[0] != M.shape[0]:
            v = torch.ones(M.shape[0], device=M.device, dtype=M.dtype)

        for _ in range(self.power_iterations):
            v = torch.mv(M, v)
            v = v / v.norm().clamp_min(1e-12)

        self._power_vec = v
        sigma = torch.dot(v, torch.mv(M, v)).clamp_min(1e-12)
        return float(self.step_size_c / sigma)

    def _solve_projection(self, g: Tensor, reference: Tensor, memory_strength: float):
        G = self.G
        g32 = g.float()

        t = G.shape[0]
        v = self.v
        if v is None or v.shape[0] != t or not self.use_warm_start:
            v = torch.zeros(t, device=G.device, dtype=G.dtype)

        eta = self._step_size()
        Gg = torch.mv(G, g32)

        for _ in range(self.pgd_iterations):
            grad_dual = torch.mv(self.GGT, v) + Gg
            v = torch.clamp(v - eta * grad_dual, min=0.0)

        if self.use_warm_start:
            self.v = v.detach()

        g_proj = torch.mv(G.T, v) + g32
        return g_proj.to(dtype=g.dtype)

    def _update_memory(self, strategy):
        """Store a random sample of the finished experience, then reset the warm start."""
        dataset = strategy.experience.dataset
        t = strategy.clock.train_exp_counter

        indices = torch.randperm(len(dataset))[: self.patterns_per_exp].tolist()
        subset = dataset.subset(indices)

        loader = DataLoader(
            subset,
            batch_size=len(subset),
            collate_fn=getattr(dataset, "collate_fn", None),
        )

        for mb in loader:
            self.memory_x[t] = mb[0]
            self.memory_y[t] = mb[1]
            self.memory_tid[t] = mb[-1]
            break

        # lambda is warm-started within a task only
        self.v = None
        self._power_vec = None
