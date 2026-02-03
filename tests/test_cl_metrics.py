"""The metric definitions in Appendix F are checked against values worked out
by hand, since an off-by-one here silently changes every number in the paper."""

import numpy as np
import pytest

from igem.utils.cl_metrics import (
    avg_acc, bwt, forgetting, fwt, mean_projection_overhead,
)

# R[j, i] = accuracy on task i after training through task j
R = np.array([
    [0.90, 0.40, 0.30],
    [0.70, 0.85, 0.35],
    [0.60, 0.75, 0.80],
])


def test_avg_acc_is_the_final_row():
    assert avg_acc(R) == pytest.approx((0.60 + 0.75 + 0.80) / 3)


def test_bwt_compares_final_against_just_learned():
    # (0.60 - 0.90) and (0.75 - 0.85)
    assert bwt(R) == pytest.approx(((0.60 - 0.90) + (0.75 - 0.85)) / 2)


def test_bwt_is_negative_when_forgetting():
    assert bwt(R) < 0


def test_fwt_uses_the_measured_baseline():
    baseline = np.array([0.25, 0.20, 0.30])
    # (R[0,1] - b[1]) and (R[1,2] - b[2])
    assert fwt(R, baseline) == pytest.approx(((0.40 - 0.20) + (0.35 - 0.30)) / 2)


def test_fwt_falls_back_to_chance():
    assert fwt(R) == pytest.approx(((0.40 - 0.25) + (0.35 - 0.25)) / 2)


def test_forgetting_uses_best_pre_final_accuracy():
    # task 0: max(0.90, 0.70) - 0.60 ; task 1: max(0.40, 0.85) - 0.75
    assert forgetting(R) == pytest.approx(((0.90 - 0.60) + (0.85 - 0.75)) / 2)


def test_forgetting_and_bwt_agree_when_accuracy_only_declines():
    """With monotone decline the best pre-final value is the diagonal, so
    forgetting is exactly -BWT."""
    assert forgetting(R) == pytest.approx(-bwt(R))


def test_single_task_metrics_are_zero():
    one = np.array([[0.5]])
    assert bwt(one) == 0.0 and fwt(one) == 0.0 and forgetting(one) == 0.0


def test_mean_projection_overhead():
    assert mean_projection_overhead(4.0, 8) == pytest.approx(0.5)
    assert mean_projection_overhead(0.0, 0) == 0.0
