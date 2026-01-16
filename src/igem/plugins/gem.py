import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from avalanche.models import avalanche_forward

try:
    import qpsolvers
except ImportError:
    raise ImportError(
        "GEM requires qpsolvers. Install it with `pip install qpsolvers[quadprog]`"
    )

from igem.plugins.base import BaseGEMPlugin


class GEMPlugin(BaseGEMPlugin):
    """
    Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017).

    Solves the dual QP exactly with an external solver, which is the cost
    I-GEM replaces with a fixed-budget PGD.
    """

    def __init__(
        self,
        patterns_per_exp: int,
        memory_strength: float,
        proj_interval: int,
        n_experiences: int,
        memory_size: int,
        param_space: str = "adapter",
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

        self.memory_x = {}
        self.memory_y = {}
        self.memory_tid = {}

    def _has_memory(self) -> bool:
        return bool(self.memory_x)

    def _compute_reference_gradients(self, strategy) -> Tensor:
        """One gradient row per stored experience, stacked into G."""
        grads = []

        for t in range(strategy.clock.train_exp_counter):
            strategy.optimizer.zero_grad()

            xref = self.memory_x[t].to(strategy.device)
            yref = self.memory_y[t].to(strategy.device)
            tid = self.memory_tid[t].to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            strategy._criterion(out, yref).backward()

            grads.append(self._flat_grad(strategy))

        return torch.stack(grads, dim=0)

    def _solve_projection(self, g: Tensor, reference: Tensor, memory_strength: float):
        """
        Dual of Eq. (2), following the reference GEM formulation:

            min_v  0.5 v^T (G G^T) v + (G g)^T v    s.t.  v >= memory_strength

        qpsolvers minimises 0.5 x^T P x + q^T x subject to Gx <= h, so the
        lower bound enters as -I v <= -memory_strength.
        """
        G_np = reference.detach().cpu().double().numpy()
        g_np = g.detach().cpu().contiguous().view(-1).double().numpy()

        t = G_np.shape[0]

        P = G_np @ G_np.T
        P = 0.5 * (P + P.T) + np.eye(t) * 1e-3
        q = G_np @ g_np

        A_ineq = -np.eye(t)
        b_ineq = np.full(t, -memory_strength)

        try:
            v = qpsolvers.solve_qp(P=P, q=q, G=A_ineq, h=b_ineq, solver="quadprog")
        except Exception:
            v = None

        if v is None:
            return g

        g_proj = v @ G_np + g_np
        return torch.from_numpy(g_proj).to(device=g.device, dtype=g.dtype)

    def _update_memory(self, strategy):
        """Fill this experience's buffer with the first patterns_per_exp samples."""
        dataset = strategy.experience.dataset
        t = strategy.clock.train_exp_counter

        loader = DataLoader(
            dataset,
            batch_size=strategy.train_mb_size,
            collate_fn=getattr(dataset, "collate_fn", None),
            shuffle=True,
        )

        xs, ys, tids = [], [], []
        count = 0

        for mb in loader:
            remaining = self.patterns_per_exp - count
            if remaining <= 0:
                break

            x, y, tid = mb[0], mb[1], mb[-1]
            take = min(x.size(0), remaining)
            xs.append(x[:take])
            ys.append(y[:take])
            tids.append(tid[:take])
            count += take

        if not xs:
            return

        self.memory_x[t] = torch.cat(xs, dim=0)
        self.memory_y[t] = torch.cat(ys, dim=0)
        self.memory_tid[t] = torch.cat(tids, dim=0)
