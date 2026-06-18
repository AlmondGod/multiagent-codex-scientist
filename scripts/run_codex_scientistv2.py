#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from commsci.codex_scientistv2 import run_codex_scientistv2


DEFAULTS = {
    "output_dir": "runs/codex_scientistv2_smoke",
    "tinyworlds_dir": "/Users/almondgod/Repositories/tinyworlds-autoresearch",
    "generations": 1,
    "num_agents": 3,
    "time_budget_seconds": 120,
    "timeout_seconds": 240,
    "parallel_workers": 3,
    "seed": 0,
    "doctrine_doc": "docs/codex-aiscientistv2.md",
    "live_actions_src": None,
    "init_default_actions": False,
    "skip_experiments": False,
    "run_initial_literature_nodes": True,
    "rerun_initial_literature_nodes": False,
    "literature_node_count": None,
    "literature_max_results_per_node": 6,
    "run_controlled_ablations": True,
    "rerun_controlled_ablations": False,
    "max_controlled_ablations": 3,
    "ablation_time_budget_seconds": 60,
    "ablation_timeout_seconds": 240,
    "ablation_parallel_workers": 1,
    "run_codex_tasks": True,
    "codex_task_names": "idea_generation,figure_making,paper_writeup,paper_reflection,llm_review",
    "codex_task_model_url": None,
    "codex_task_model": None,
    "codex_task_temperature": None,
    "codex_task_max_completion_tokens": 12000,
    "codex_task_dry_run": False,
    "codex_task_fail_fast": False,
    "apply_codex_task_outputs": True,
    "execute_plot_task_outputs": True,
    "codex_task_execution_timeout_seconds": 120,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex-Scientist-v2 staged TinyWorlds autoresearch.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tinyworlds_dir", default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--time_budget_seconds", type=int, default=None)
    parser.add_argument("--timeout_seconds", type=int, default=None)
    parser.add_argument("--parallel_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--doctrine_doc", default=None)
    parser.add_argument("--live_actions_src", default=None)
    parser.add_argument("--init_default_actions", action="store_true", default=None)
    parser.add_argument("--skip_experiments", action="store_true", default=None)
    parser.add_argument("--run_initial_literature_nodes", action="store_true", default=None)
    parser.add_argument("--rerun_initial_literature_nodes", action="store_true", default=None)
    parser.add_argument("--literature_node_count", type=int, default=None)
    parser.add_argument("--literature_max_results_per_node", type=int, default=None)
    parser.add_argument("--run_controlled_ablations", action="store_true", default=None)
    parser.add_argument("--rerun_controlled_ablations", action="store_true", default=None)
    parser.add_argument("--max_controlled_ablations", type=int, default=None)
    parser.add_argument("--ablation_time_budget_seconds", type=int, default=None)
    parser.add_argument("--ablation_timeout_seconds", type=int, default=None)
    parser.add_argument("--ablation_parallel_workers", type=int, default=None)
    parser.add_argument("--run_codex_tasks", dest="run_codex_tasks", action="store_true", default=None)
    parser.add_argument("--no_run_codex_tasks", dest="run_codex_tasks", action="store_false")
    parser.add_argument("--codex_task_names", default=None, help="Comma-separated codex task names, e.g. idea_generation,figure_making,paper_writeup,llm_review.")
    parser.add_argument("--codex_task_model_url", default=None)
    parser.add_argument("--codex_task_model", default=None)
    parser.add_argument("--codex_task_temperature", type=float, default=None)
    parser.add_argument("--codex_task_max_completion_tokens", type=int, default=None)
    parser.add_argument("--codex_task_dry_run", dest="codex_task_dry_run", action="store_true", default=None)
    parser.add_argument("--no_codex_task_dry_run", dest="codex_task_dry_run", action="store_false")
    parser.add_argument("--codex_task_fail_fast", action="store_true", default=None)
    parser.add_argument("--apply_codex_task_outputs", dest="apply_codex_task_outputs", action="store_true", default=None)
    parser.add_argument("--no_apply_codex_task_outputs", dest="apply_codex_task_outputs", action="store_false")
    parser.add_argument("--execute_plot_task_outputs", dest="execute_plot_task_outputs", action="store_true", default=None)
    parser.add_argument("--no_execute_plot_task_outputs", dest="execute_plot_task_outputs", action="store_false")
    parser.add_argument("--codex_task_execution_timeout_seconds", type=int, default=None)
    args = parser.parse_args()
    return apply_config_defaults(args)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = dict(DEFAULTS)
    if args.config:
        config = load_yaml(args.config)
        runner = (config.get("experiment") or {}).get("runner")
        if runner and runner != "codex_scientistv2":
            raise ValueError(f"{args.config} sets experiment.runner={runner!r}, expected 'codex_scientistv2'")
        defaults.update(defaults_from_config(config))
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def load_yaml(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def defaults_from_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment") or {}
    compute = config.get("compute") or {}
    paths = config.get("paths") or {}
    model = config.get("model") or {}
    v2 = config.get("codex_scientistv2") or {}
    out: dict[str, Any] = {}
    direct_keys = {
        "output_dir",
        "tinyworlds_dir",
        "generations",
        "time_budget_seconds",
        "timeout_seconds",
        "parallel_workers",
        "doctrine_doc",
        "live_actions_src",
        "init_default_actions",
        "skip_experiments",
        "run_initial_literature_nodes",
        "rerun_initial_literature_nodes",
        "literature_node_count",
        "literature_max_results_per_node",
        "run_controlled_ablations",
        "rerun_controlled_ablations",
        "max_controlled_ablations",
        "ablation_time_budget_seconds",
        "ablation_timeout_seconds",
        "ablation_parallel_workers",
        "run_codex_tasks",
        "codex_task_names",
        "codex_task_model_url",
        "codex_task_model",
        "codex_task_temperature",
        "codex_task_max_completion_tokens",
        "codex_task_dry_run",
        "codex_task_fail_fast",
        "apply_codex_task_outputs",
        "execute_plot_task_outputs",
        "codex_task_execution_timeout_seconds",
    }
    for key in direct_keys:
        if key in v2:
            out[key] = v2[key]
    if "output_dir" in paths:
        out.setdefault("output_dir", paths["output_dir"])
    if "tinyworlds_dir" in paths:
        out.setdefault("tinyworlds_dir", paths["tinyworlds_dir"])
    if "num_agents" in compute:
        out.setdefault("num_agents", compute["num_agents"])
    if "seed" in model:
        out.setdefault("seed", model["seed"])
    if model.get("mock_model"):
        out.setdefault("codex_task_dry_run", True)
    mapping = {
        "codex_scientist_time_budget_seconds": "time_budget_seconds",
        "codex_scientist_timeout_seconds": "timeout_seconds",
        "codex_scientist_parallel_workers": "parallel_workers",
        "codex_scientist_doctrine_doc": "doctrine_doc",
        "ai_scientist_data_dir": "tinyworlds_dir",
    }
    for source, target in mapping.items():
        if source in experiment:
            out.setdefault(target, experiment[source])
    return out


def main() -> int:
    output = run_codex_scientistv2(parse_args())
    print(f"Wrote Codex-Scientist-v2 artifacts to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
