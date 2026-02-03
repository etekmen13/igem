"""The projection routines are the core of the method, so they get checked
against an exact QP solve rather than only for self-consistency."""

import numpy as np
import pytest
import qpsolvers
import torch

from igem.plugins import AGEMPlugin, GEMPlugin, IGEMPlugin


def exact_dual(G, g, lower=0.0):
    """Reference solution of the dual in Eq. (3) via an external QP solver."""
    G_np = G.double().numpy()
    g_np = g.double().numpy()
    t = G_np.shape[0]

    P = G_np @ G_np.T
    P = 0.5 * (P + P.T) + np.eye(t) * 1e-9
    q = G_np @ g_np

    v = qpsolvers.solve_qp(
        P=P, q=q, G=-np.eye(t), h=np.full(t, -lower), solver="quadprog"
    )
    return torch.from_numpy(v @ G_np + g_np).float()


def make_igem(**kw):
    kw.setdefault("pgd_iterations", 500)
    kw.setdefault("lr", 0.0)
    kw.setdefault("use_adaptive_lr", True)
    kw.setdefault("use_warm_start", False)
    return IGEMPlugin(
        patterns_per_exp=1,
        memory_strength=0.0,
        proj_interval=1,
        n_experiences=2,
        memory_size=2,
        normalize_rows=False,
        **kw,
    )


def load(plugin, G):
    plugin.G = G
    plugin.GGT = G @ G.T
    return plugin


@pytest.mark.parametrize("seed", range(8))
def test_igem_pgd_matches_exact_qp(seed):
    """Given enough iterations the fixed-budget PGD reaches the QP optimum."""
    torch.manual_seed(seed)
    G = torch.randn(3, 40)
    g = torch.randn(40)

    plugin = load(make_igem(), G)
    got = plugin._solve_projection(g, G, 0.0)

    torch.testing.assert_close(got, exact_dual(G, g), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("seed", range(8))
def test_igem_projection_is_feasible(seed):
    """The projected gradient must satisfy the constraints of Eq. (2)."""
    torch.manual_seed(seed)
    G = torch.randn(3, 40)
    g = -G.sum(dim=0)  # violates every constraint

    assert (torch.mv(G, g) < 0).all()

    plugin = load(make_igem(), G)
    g_proj = plugin._solve_projection(g, G, 0.0)

    assert (torch.mv(G, g_proj) >= -1e-4).all()


def test_igem_leaves_feasible_gradients_alone():
    """A gradient that already satisfies the constraints is a fixed point."""
    torch.manual_seed(0)
    G = torch.randn(2, 30)
    g = G.sum(dim=0)
    assert (torch.mv(G, g) > 0).all()

    plugin = load(make_igem(), G)
    torch.testing.assert_close(plugin._solve_projection(g, G, 0.0), g, rtol=1e-4, atol=1e-5)


def test_warm_start_reaches_the_same_solution():
    """Warm starting changes the path, not the destination."""
    torch.manual_seed(3)
    G = torch.randn(3, 40)
    g = torch.randn(40)

    cold = load(make_igem(use_warm_start=False), G)._solve_projection(g, G, 0.0)

    warm = load(make_igem(pgd_iterations=100, use_warm_start=True), G)
    for _ in range(5):
        out = warm._solve_projection(g, G, 0.0)

    torch.testing.assert_close(out, cold, rtol=1e-3, atol=1e-4)


def test_row_normalisation_preserves_the_optimum():
    """Positive row scaling leaves the feasible set, hence the optimum, unchanged."""
    torch.manual_seed(5)
    G = torch.randn(3, 40) * torch.tensor([[1.0], [50.0], [0.02]])
    g = torch.randn(40)
    Gn = G / G.norm(dim=1, keepdim=True)

    converged = load(make_igem(), Gn)._solve_projection(g, Gn, 0.0)

    torch.testing.assert_close(converged, exact_dual(G, g), rtol=1e-3, atol=1e-4)


def test_row_normalisation_helps_a_small_budget():
    """Sec. III-D: normalising rows stops one constraint dominating the dual,
    which is what makes a K of 2-6 enough on badly scaled G."""
    torch.manual_seed(5)
    G = torch.randn(3, 40) * torch.tensor([[1.0], [50.0], [0.02]])
    g = torch.randn(40)
    Gn = G / G.norm(dim=1, keepdim=True)
    target = exact_dual(G, g)

    raw = load(make_igem(pgd_iterations=3), G)._solve_projection(g, G, 0.0)
    normed = load(make_igem(pgd_iterations=3), Gn)._solve_projection(g, Gn, 0.0)

    assert (normed - target).norm() < (raw - target).norm()


@pytest.mark.parametrize("margin", [0.0, 0.3])
def test_gem_qp_satisfies_its_constraints(margin):
    """The exact solver must not move the gradient further into violation."""
    torch.manual_seed(1)
    G = torch.randn(3, 25)
    g = -G.sum(dim=0)

    plugin = GEMPlugin(
        patterns_per_exp=1,
        memory_strength=margin,
        proj_interval=1,
        n_experiences=2,
        memory_size=2,
    )
    g_proj = plugin._solve_projection(g, G, margin)

    before = torch.mv(G, g)
    after = torch.mv(G, g_proj)

    # the solver adds a 1e-3 ridge to P for conditioning, so the constraints
    # are met up to that perturbation rather than exactly
    assert (after >= -5e-3).all()
    assert (after > before).all()


def test_gem_and_igem_agree_without_a_margin():
    """Both solve the same dual, so they must land in the same place."""
    torch.manual_seed(7)
    G = torch.randn(3, 50)
    g = torch.randn(50)

    gem = GEMPlugin(
        patterns_per_exp=1, memory_strength=0.0, proj_interval=1,
        n_experiences=2, memory_size=2,
    )
    igem = load(make_igem(), G)

    torch.testing.assert_close(
        igem._solve_projection(g, G, 0.0),
        gem._solve_projection(g, G, 0.0),
        rtol=1e-3, atol=1e-4,
    )


def test_agem_projects_onto_the_boundary():
    """A-GEM's closed form puts the gradient exactly on g.ref = 0."""
    ref = torch.tensor([1.0, 2.0, 3.0])
    g = torch.tensor([-1.0, -0.5, 0.2])
    assert torch.dot(g, ref) < 0

    plugin = AGEMPlugin(
        patterns_per_exp=1, sample_size=1, memory_strength=0.3,
        proj_interval=1, n_experiences=2, memory_size=2,
    )
    g_proj = plugin._solve_projection(g, ref, 0.3)

    assert torch.dot(g_proj, ref).abs() < 1e-5


def test_agem_degenerate_reference_is_a_noop():
    g = torch.randn(10)
    plugin = AGEMPlugin(
        patterns_per_exp=1, sample_size=1, memory_strength=0.0,
        proj_interval=1, n_experiences=2, memory_size=2,
    )
    torch.testing.assert_close(plugin._solve_projection(g, torch.zeros(10), 0.0), g)
