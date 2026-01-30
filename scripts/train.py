import argparse
import itertools
import os

import torch
import yaml
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import TextLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

from igem.benchmarks.agnews import make_agnews_benchmark
from igem.models.gpt2 import get_gpt2_lora
from igem.plugins import AGEMPlugin, GEMPlugin, IGEMPlugin
from igem.utils.cl_metrics import mean_projection_overhead
from igem.utils.common import save_results, set_seed, train_and_evaluate
from igem.utils.metrics import ProjectionOverheadMetric


def get_device(cuda_id):
    if torch.cuda.is_available() and cuda_id >= 0:
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def build_plugin(name, cfg, proj_metric):
    common = dict(
        patterns_per_exp=cfg["patterns_per_exp"],
        memory_strength=cfg["memory_strength"],
        proj_interval=cfg["proj_interval"],
        n_experiences=cfg["n_experiences"],
        memory_size=cfg["memory_size"],
        param_space=cfg.get("param_space", "adapter"),
        proj_metric=proj_metric,
    )

    if name == "igem":
        return IGEMPlugin(
            **common,
            pgd_iterations=cfg.get("pgd_iterations", 3),
            lr=cfg["lr"],
            use_adaptive_lr=cfg.get("adaptive_lr", True),
            use_warm_start=cfg.get("warm_start", True),
        )
    if name == "gem":
        return GEMPlugin(**common)
    if name == "agem":
        return AGEMPlugin(**common, sample_size=cfg.get("sample_size", 256))
    if name == "naive":
        return None
    raise ValueError(f"unknown plugin: {name}")


def run_experiment(cfg):
    print(f"\n=== {cfg['plugin']} | seed {cfg['seed']} | {cfg['param_space']} space ===")

    set_seed(cfg["seed"])
    device = get_device(cfg.get("cuda", 0))

    benchmark = make_agnews_benchmark(
        n_experiences=cfg["n_experiences"],
        seed=cfg["seed"],
        exp_size=cfg.get("exp_size", 2000),
        max_len=cfg.get("max_len", 512),
    )

    model = get_gpt2_lora(model_name=cfg.get("model_name", "gpt2-medium")).to(device)

    proj_metric = ProjectionOverheadMetric()
    plugin = build_plugin(cfg["plugin"], cfg, proj_metric)
    plugins = [plugin] if plugin is not None else []

    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True),
        proj_metric,
        loggers=[TextLogger()],
        strict_checks=False,
    )

    strategy = Naive(
        model=model,
        optimizer=torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["lr"],
            weight_decay=1e-2,
        ),
        criterion=torch.nn.CrossEntropyLoss(),
        train_mb_size=cfg["train_mb_size"],
        train_epochs=cfg["train_epochs"],
        eval_mb_size=cfg["eval_mb_size"],
        device=device,
        plugins=plugins,
        evaluator=evaluator,
        eval_every=-1,
    )

    R, baseline, results = train_and_evaluate(
        strategy, benchmark.train_stream, benchmark.test_stream
    )

    summary = {
        "plugin": cfg["plugin"],
        "seed": cfg["seed"],
        "param_space": cfg["param_space"],
        "n_projections": getattr(plugin, "n_projections", 0),
        "total_projection_time": getattr(plugin, "total_projection_time", 0.0),
        "mpo": mean_projection_overhead(
            getattr(plugin, "total_projection_time", 0.0),
            getattr(plugin, "n_projections", 0),
        ),
    }

    return save_results(
        R,
        baseline,
        results,
        cfg["output_dir"],
        f"{cfg['plugin']}_s{cfg['seed']}",
        summary,
    )


def as_list(value):
    return value if isinstance(value, list) else [value]


def expand(section):
    keys = list(section)
    grids = [as_list(section[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--plugin", help="run only this plugin")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue the sweep when a run fails")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    plugins = cfg["plugins"]
    if args.plugin:
        plugins = {args.plugin: plugins[args.plugin]}

    failures = []
    for plugin_name, plugin_cfg in plugins.items():
        for global_cfg in expand(cfg["global"]):
            for local_cfg in expand(plugin_cfg or {}):
                run_cfg = {**global_cfg, **local_cfg, "plugin": plugin_name}
                try:
                    run_experiment(run_cfg)
                except Exception:
                    if not args.keep_going:
                        raise
                    import traceback
                    traceback.print_exc()
                    failures.append((plugin_name, run_cfg["seed"]))

    if failures:
        raise SystemExit(f"{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
    main()
