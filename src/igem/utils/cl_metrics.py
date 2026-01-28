"""Continual-learning metrics, as defined in Appendix F.

R is indexed R[j, i] = accuracy on task i after training through task j, and
`baseline` holds R[0, i], the accuracy on task i before any continual training
(pretrained base, freshly initialised adapters).
"""

import numpy as np


def avg_acc(R):
    """Mean accuracy over all tasks at the end of training."""
    return float(R[-1, :].mean())


def bwt(R):
    """Average change in a task's accuracy between learning it and finishing."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[T - 1, i] - R[i, i] for i in range(T - 1)]))


def fwt(R, baseline=None, random_baseline=0.25):
    """Zero-shot accuracy on a task just before learning it, minus its baseline.

    `baseline` should be the measured R[0, :] row. The chance-level fallback is
    only there for runs saved before that row was recorded.
    """
    T = R.shape[0]
    if T < 2:
        return 0.0

    if baseline is None:
        b = np.full(T, random_baseline)
    else:
        b = np.asarray(baseline, dtype=float)

    return float(np.mean([R[i - 1, i] - b[i] for i in range(1, T)]))


def forgetting(R):
    """Average drop from a task's best pre-final accuracy to its final one."""
    T = R.shape[0]
    if T < 2:
        return 0.0
    return float(np.mean([R[: T - 1, i].max() - R[T - 1, i] for i in range(T - 1)]))


def mean_projection_overhead(total_seconds, n_projections):
    """MPO: mean wall-clock time of a single projection."""
    if not n_projections:
        return 0.0
    return float(total_seconds) / int(n_projections)
