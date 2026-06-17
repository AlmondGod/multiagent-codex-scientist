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
