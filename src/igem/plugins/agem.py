import random

import torch
from torch import Tensor
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.models import avalanche_forward

from igem.plugins.base import BaseGEMPlugin


class AGEMPlugin(BaseGEMPlugin):
    """
    Averaged GEM (Chaudhry et al., 2019).

    Collapses the t constraints of GEM into a single averaged reference
    gradient, which turns the projection into a closed-form expression.

    ``memory_strength`` is not used here: the reference formulation projects
    onto the constraint boundary exactly, so there is no margin to set.
    """

    def __init__(
        self,
        patterns_per_exp: int,
        sample_size: int,
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
        self.sample_size = sample_size
        self.buffers = []
        self.buffer_dataloader = None
        self.buffer_dliter = None

    def _has_memory(self) -> bool:
        return len(self.buffers) > 0

    def _compute_reference_gradients(self, strategy) -> Tensor:
        """A single averaged gradient over one batch drawn from the buffers."""
        try:
            batch = next(self.buffer_dliter)
        except StopIteration:
            self.buffer_dliter = iter(self.buffer_dataloader)
            batch = next(self.buffer_dliter)

        x, y, tid = batch[0], batch[1], batch[-1]
        x = x.to(strategy.device)
        y = y.to(strategy.device)
        tid = tid.to(strategy.device)

        strategy.optimizer.zero_grad()
        out = avalanche_forward(strategy.model, x, tid)
        strategy._criterion(out, y).backward()

        return self._flat_grad(strategy)

    def _should_project(self, g: Tensor) -> bool:
        return bool(torch.dot(self.reference, g) < 0)

    def _solve_projection(self, g: Tensor, reference: Tensor, memory_strength: float):
        """
        g_proj = g - (g . ref / ref . ref) ref

        which is the point of the constraint boundary closest to g, i.e.
        g_proj . ref = 0.
        """
        ref_sq = torch.dot(reference, reference)
        if ref_sq < 1e-12:
            return g

        alpha = torch.dot(g, reference) / ref_sq
        return g - alpha * reference

    def _update_memory(self, strategy):
        """Keep a per-experience buffer and sample from all of them evenly."""
        dataset = strategy.experience.dataset

        indices = list(range(len(dataset)))
        random.shuffle(indices)
        self.buffers.append(dataset.subset(indices[: self.patterns_per_exp]))

        self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
            self.buffers,
            batch_size=self.sample_size,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        self.buffer_dliter = iter(self.buffer_dataloader)
