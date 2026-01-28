import json
import os
import random

import numpy as np
import pandas as pd
import torch
from avalanche.training.templates import SupervisedTemplate

from igem.utils import cl_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _accuracy_row(metrics, n_tasks):
    """Pull the per-experience accuracies out of an Avalanche metrics dict."""
    row = np.zeros(n_tasks)

    for i in range(n_tasks):
        suffix = f"Exp{i:03d}"
        for key, value in metrics.items():
            if "Top1_Acc_Exp" in key and key.endswith(suffix):
                row[i] = value
                break
        else:
            raise KeyError(
                f"no Top1_Acc_Exp entry for experience {i}; "
                f"available keys: {sorted(metrics)[:5]}..."
            )

    return row


def train_and_evaluate(strategy: SupervisedTemplate, train_stream, test_stream):
    """Train over the stream, evaluating every task after every experience.

    Returns the R matrix, the pre-training baseline row R[0, :] used by FWT,
    and the raw per-experience metric dicts.
    """
    n_tasks = len(train_stream)

    # R[0, :]: what the frozen base with untrained adapters already scores
    baseline = _accuracy_row(strategy.eval(test_stream), n_tasks)

    R = np.zeros((n_tasks, n_tasks))
    results = []

    for j, train_exp in enumerate(train_stream):
        print(f"\n--- training experience {j} ---")
        strategy.train(train_exp)

        metrics = strategy.eval(test_stream)
        results.append(metrics)
        R[j] = _accuracy_row(metrics, n_tasks)

    print("\nR matrix:\n", R)
    return R, baseline, results


def save_results(R, baseline, results, output_dir, filename, summary=None):
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.join(output_dir, filename)

    np.save(stem + ".npy", R)
    np.save(stem + "_baseline.npy", baseline)
    pd.DataFrame(results).to_csv(stem + ".csv", index=False)

    record = {
        "avg_acc": cl_metrics.avg_acc(R),
        "bwt": cl_metrics.bwt(R),
        "fwt": cl_metrics.fwt(R, baseline),
        "forgetting": cl_metrics.forgetting(R),
        **(summary or {}),
    }
    with open(stem + ".json", "w") as f:
        json.dump(record, f, indent=2)

    print(f"results written to {stem}.{{npy,csv,json}}")
    return record
