from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import Tensor
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin

from igem.plugins import core


class BaseGEMPlugin(SupervisedPlugin, ABC):
    """
    Shared machinery for the GEM family.

    The projection is applied to a flattened gradient vector built from a
    chosen slice of the model's parameters. With ``param_space="adapter"``
    only trainable (LoRA) parameters are collected, so the constraint matrix
    G is (t x d_phi) rather than (t x d_theta). This is the setting the
    method is defined in -- see Eq. (2) of the paper. ``param_space="full"``
    keeps every parameter, including the frozen base weights, which is what
    the exact-GEM cost comparison in Sec. III-D refers to.
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
        super().__init__()
        if param_space not in ("adapter", "full"):
            raise ValueError(f"param_space must be 'adapter' or 'full', got {param_space!r}")

        self.patterns_per_exp = patterns_per_exp
        self.memory_strength = memory_strength
        self.proj_interval = proj_interval
        self.param_space = param_space
        self.proj_metric = proj_metric

        self.per_exp_memory_size = memory_size // n_experiences

        self.projection_iteration = 0
        self.reference: Optional[Tensor] = None

        # run-level projection accounting, used for mean projection overhead
        self.n_projections = 0
        self.total_projection_time = 0.0

    def _projected_params(self, strategy) -> List[torch.nn.Parameter]:
        params = strategy.model.parameters()
        if self.param_space == "adapter":
            return [p for p in params if p.requires_grad]
        return list(params)

    def _flat_grad(self, strategy) -> Tensor:
        """Flatten the parameter slice's gradients into one float32 vector.

        The projection always runs in float32 even when the adapters are held
        in bfloat16, since the dual involves an eigenvalue estimate.
        """
        chunks = [
            p.grad.detach().flatten().float()
            if p.grad is not None
            else torch.zeros(p.numel(), device=strategy.device)
            for p in self._projected_params(strategy)
        ]
        return torch.cat(chunks, dim=0)

    def _write_grad(self, strategy, g_proj: Tensor):
        offset = 0
        for p in self._projected_params(strategy):
            n = p.numel()
            if p.grad is not None:
                p.grad.copy_(g_proj[offset : offset + n].view_as(p))
            offset += n
        assert offset == g_proj.numel(), (
            f"projected gradient has {g_proj.numel()} entries but the "
            f"parameter slice needs {offset}"
        )

    def before_training_iteration(self, strategy, *args, **kwargs):
        """Build G from the replay buffers before the current step."""
        if not self._has_memory():
            return

        self.reference = self._compute_reference_gradients(strategy)
        strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, *args, **kwargs):
        """Project the gradient if it violates the non-interference constraints."""
        if not self._has_memory():
            return

        g = self._flat_grad(strategy)

        should_project = (
            self.projection_iteration % self.proj_interval == 0
            and self._should_project(g)
        )
        self.projection_iteration += 1

        if not should_project:
            return

        g_proj, elapsed = core.time_projection(
            self._solve_projection,
            g=g,
            reference=self.reference,
            memory_strength=self.memory_strength,
        )

        self.n_projections += 1
        self.total_projection_time += elapsed
        if self.proj_metric:
            self.proj_metric.record(elapsed)

        self._write_grad(strategy, g_proj)

    def after_training_exp(self, strategy, *args, **kwargs):
        self._update_memory(strategy)

    def _should_project(self, g: Tensor) -> bool:
        """Constraint of Eq. (2) is G g >= 0; project as soon as a row is violated."""
        return bool((torch.mv(self.reference, g) < 0).any())

    @abstractmethod
    def _has_memory(self) -> bool:
        pass

    @abstractmethod
    def _compute_reference_gradients(self, strategy) -> Tensor:
        pass

    @abstractmethod
    def _solve_projection(self, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def _update_memory(self, strategy):
        pass
