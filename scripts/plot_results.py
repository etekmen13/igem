import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed per-method colours so a method keeps its colour no matter which subset
# of runs is present in a directory.
COLORS = {
    "GEM": "#2a78d6",
    "A-GEM": "#eb6834",
    "I-GEM": "#1baf7a",
    "NAIVE": "#eda100",
}
ORDER = ["GEM", "A-GEM", "I-GEM", "NAIVE"]
LABELS = {"igem": "I-GEM", "agem": "A-GEM", "gem": "GEM", "naive": "NAIVE"}


def load_runs(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            record = json.load(f)
        record["Method"] = LABELS.get(record.get("plugin", ""), record.get("plugin", "?"))
        rows.append(record)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    present = [m for m in ORDER if m in set(df["Method"])]
    df["Method"] = pd.Categorical(df["Method"], categories=present, ordered=True)
    return df.sort_values("Method")


def summarise(df):
    cols = ["avg_acc", "bwt", "fwt", "forgetting", "mpo"]
    return df.groupby("Method", observed=True)[cols].agg(["mean", "std"])


def _style(ax):
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("0.7")
    ax.tick_params(length=0, colors="0.35")


def plot_cl_metrics(stats, output_dir):
    metrics = [("avg_acc", "AvgAcc"), ("bwt", "BWT"), ("forgetting", "Forgetting")]
    methods = list(stats.index)

    x = np.arange(len(metrics))
    width = min(0.8 / len(methods), 0.26)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, method in enumerate(methods):
        means = [stats.loc[method, (m, "mean")] for m, _ in metrics]
        errs = [np.nan_to_num(stats.loc[method, (m, "std")]) for m, _ in metrics]
        offset = (k - (len(methods) - 1) / 2) * width

        bars = ax.bar(x + offset, means, width * 0.92, yerr=errs, capsize=3,
                      label=method, color=COLORS.get(method, "#777"), linewidth=0)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color="0.35")

    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Continual learning metrics on AG News", loc="left", fontsize=11)
    ax.legend(frameon=False, ncols=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    _style(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_cl_metrics.png"), dpi=200)
    plt.close(fig)


def plot_overhead(stats, output_dir):
    """Overhead spans orders of magnitude, so this is a log axis. Bars imply a
    zero baseline that a log scale does not have, hence dots on stems."""
    methods = list(stats.index)
    means = [stats.loc[m, ("mpo", "mean")] for m in methods]
    errs = [np.nan_to_num(stats.loc[m, ("mpo", "std")]) for m in methods]

    floor = min(v for v in means if v > 0) / 6
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for xi, mean, err, method in zip(x, means, errs, methods):
        color = COLORS.get(method, "#777")
        ax.vlines(xi, floor, mean, color=color, linewidth=2, alpha=0.45)
        ax.errorbar(xi, mean, yerr=err, fmt="o", markersize=9, color=color,
                    capsize=3, elinewidth=1.2)
        ax.annotate(f"{mean:.2e}", (xi, mean), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="0.35")

    ax.set_yscale("log")
    ax.set_ylim(bottom=floor, top=max(means) * 4)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_xticks(x, methods)
    ax.set_ylabel("Mean projection overhead (s, log scale)")
    ax.set_title("Cost of a single projection", loc="left", fontsize=11)
    _style(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_overhead.png"), dpi=200)
    plt.close(fig)


def latex_tables(stats):
    def cell(method, key, fmt="{:.3f}"):
        mean = stats.loc[method, (key, "mean")]
        std = stats.loc[method, (key, "std")]
        if np.isnan(std):
            return fmt.format(mean)
        return f"{fmt.format(mean)} $\\pm$ {fmt.format(std)}"

    out = ["", "% Table I", "\\begin{tabular}{lcc}", "\\toprule",
           "Method & AvgAcc (\\%) & Mean Projection Overhead (s) \\\\", "\\midrule"]
    for m in stats.index:
        acc = stats.loc[m, ("avg_acc", "mean")] * 100
        acc_sd = stats.loc[m, ("avg_acc", "std")] * 100
        sd = "" if np.isnan(acc_sd) else f" $\\pm$ {acc_sd:.2f}"
        out.append(f"{m} & {acc:.2f}{sd} & {cell(m, 'mpo', '{:.2e}')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "", "% Table II",
            "\\begin{tabular}{lccc}", "\\toprule",
            "Method & BWT & FWT & Forgetting \\\\", "\\midrule"]
    for m in stats.index:
        out.append(f"{m} & {cell(m, 'bwt')} & {cell(m, 'fwt')} & {cell(m, 'forgetting')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results/agnews/")
    args = parser.parse_args()

    df = load_runs(args.results_dir)
    if df.empty:
        raise SystemExit(f"no run summaries found in {args.results_dir}; run train.py first")

    stats = summarise(df)
    print(f"loaded {len(df)} runs\n")
    print(stats.to_string())
    print(latex_tables(stats))

    plot_cl_metrics(stats, args.results_dir)
    plot_overhead(stats, args.results_dir)
    print(f"figures written to {args.results_dir}")


if __name__ == "__main__":
    main()
