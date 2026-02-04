"""End-to-end runs on a small synthetic stream.

These exercise the Avalanche plugin hooks, the parameter-space selection and
the R-matrix bookkeeping, which unit tests on the projection maths do not.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import accuracy_metrics
from torch.utils.data import Dataset

from igem.plugins import AGEMPlugin, GEMPlugin, IGEMPlugin
from igem.utils.common import train_and_evaluate
from igem.utils.metrics import ProjectionOverheadMetric

N_EXP, N_FEAT, N_CLASS = 3, 12, 4


class Synthetic(Dataset):
    def __init__(self, n, shift, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, N_FEAT, generator=g) + shift
        self.y = torch.randint(0, N_CLASS, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], torch.tensor(0)


class Model(nn.Module):
    """A frozen 'base' followed by a small trainable head, mirroring LoRA."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(N_FEAT, 16)
        self.head = nn.Sequential(nn.Tanh(), nn.Linear(16, N_CLASS))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.head(self.base(x))


def make_benchmark():
    train, test = [], []
    for e in range(N_EXP):
        train.append(AvalancheDataset(Synthetic(48, e * 2.0, seed=e)))
        test.append(AvalancheDataset(Synthetic(24, e * 2.0, seed=100 + e)))
    return benchmark_from_datasets(train=train, test=test)


def build(plugin_name, param_space, proj_metric):
    common = dict(
        patterns_per_exp=8, memory_strength=0.3, proj_interval=1,
        n_experiences=N_EXP, memory_size=24, param_space=param_space,
        proj_metric=proj_metric,
    )
    if plugin_name == "gem":
        return GEMPlugin(**common)
    if plugin_name == "agem":
        return AGEMPlugin(**common, sample_size=8)
    return IGEMPlugin(**common, pgd_iterations=3, lr=1e-3,
                      use_adaptive_lr=True, use_warm_start=True)


def run(plugin_name, param_space="adapter"):
    torch.manual_seed(0)
    model = Model()
    proj_metric = ProjectionOverheadMetric()
    plugin = build(plugin_name, param_space, proj_metric)

    strategy = Naive(
        model=model,
        optimizer=torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        ),
        criterion=nn.CrossEntropyLoss(),
        train_mb_size=16, train_epochs=1, eval_mb_size=16,
        device=torch.device("cpu"), plugins=[plugin],
        evaluator=EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            proj_metric, loggers=[], strict_checks=False,
        ),
        eval_every=-1,
    )

    bench = make_benchmark()
    R, baseline, _ = train_and_evaluate(strategy, bench.train_stream, bench.test_stream)
    return plugin, R, baseline


@pytest.mark.parametrize("name", ["gem", "agem", "igem"])
def test_runs_end_to_end_and_fills_the_r_matrix(name):
    _, R, baseline = run(name)

    assert R.shape == (N_EXP, N_EXP)
    assert baseline.shape == (N_EXP,)
    assert np.isfinite(R).all()
    assert ((R >= 0) & (R <= 1)).all()


@pytest.mark.parametrize("name", ["gem", "agem", "igem"])
def test_projections_are_counted(name):
    plugin, _, _ = run(name)

    assert plugin.n_projections > 0, "no projection ever fired"
    assert plugin.total_projection_time > 0.0


def test_adapter_space_excludes_the_frozen_base():
    """The whole point of Eq. (2): G has d_phi columns, not d_theta."""
    plugin, _, _ = run("igem", param_space="adapter")
    trainable = sum(p.numel() for p in Model().parameters() if p.requires_grad)
    total = sum(p.numel() for p in Model().parameters())

    assert plugin.G.shape[1] == trainable
    assert plugin.G.shape[1] < total


def test_full_space_includes_the_frozen_base():
    plugin, _, _ = run("igem", param_space="full")
    total = sum(p.numel() for p in Model().parameters())

    assert plugin.G.shape[1] == total


def test_adapter_space_is_cheaper_than_full_space():
    adapter, _, _ = run("igem", param_space="adapter")
    full, _, _ = run("igem", param_space="full")

    assert adapter.G.numel() < full.G.numel()


def test_igem_resets_the_warm_start_at_task_boundaries():
    plugin, _, _ = run("igem")
    assert plugin.v is None, "lambda must be reset once the experience ends"
