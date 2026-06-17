from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


CONDITIONS = {"self_critique", "peer_critique", "peer_critique_with_roles"}


DEFAULT_CONFIG: dict[str, Any] = {
    "base_system": {
        "repo_url": "https://github.com/SakanaAI/AI-Scientist-v2",
        "use_existing_tree_search": True,
        "use_existing_reviewer": True,
        "write_full_paper": False,
        "reviewer_enabled": True,
        "reviewer_backend": "ai_scientist",
        "reviewer_model": "gpt-4o-mini",
        "reviewer_model_url": None,
        "reviewer_num_reflections": 1,
        "reviewer_num_fs_examples": 0,
        "reviewer_temperature": 0.2,
    },
    "model": {
        "backend": "openai_compatible",
        "model_url": "http://localhost:1234/v1",
        "default_model": "qwen3-coder-30b-a3b-instruct",
        "fallback_models": ["qwen2.5-coder-14b", "qwen2.5-coder-7b"],
        "temperature": 0.2,
        "seed": 0,
        "max_prompt_tokens": 6000,
        "max_completion_tokens": 1000,
        "mock_model": False,
    },
    "compute": {
        "reduced_mode": True,
        "num_agents": 3,
        "max_depth_per_agent": 2,
        "max_experiments_per_agent": 2,
        "max_total_experiments": 6,
        "max_training_steps": 1000,
        "max_runtime_minutes_per_experiment": 20,
        "max_tokens_per_critique": 1000,
    },
    "experiment": {
        "substrate": "TinyWorlds",
        "runner": "tinyworlds_command",
        "conditions": ["self_critique", "peer_critique", "peer_critique_with_roles"],
        "task_spec": "Improve TinyWorlds world-model quality under a fixed small compute budget.",
        "train_command": None,
        "eval_command": None,
        "dataset_config_path": None,
        "allowed_files": [],
        "forbidden_files": [],
        "metric_json_paths": {},
        "metric_regexes": {},
        "primary_metric": "primary_score",
        "ai_scientist_data_dir": None,
        "ai_scientist_config_template": None,
        "ai_scientist_code_model": "ollama/qwen3:32b",
        "ai_scientist_feedback_model": "ollama/qwen3:32b",
        "ai_scientist_timeout_seconds": 600,
        "ai_scientist_time_budget_seconds": 100,
        "ai_scientist_copy_data": False,
        "codex_scientist_timeout_seconds": 600,
        "codex_scientist_time_budget_seconds": 100,
        "codex_scientist_shared_memory": False,
        "codex_scientist_backend": "supervised_thread_tools",
        "codex_scientist_parallel_workers": 3,
        "codex_scientist_action_overrides_dir": None,
        "codex_scientist_critique_overrides_dir": None,
        "codex_scientist_decision_overrides_dir": None,
        "codex_scientist_wait_for_live_overrides": False,
        "codex_scientist_population_context": True,
        "codex_scientist_live_override_timeout_seconds": 1800,
        "codex_scientist_live_override_poll_seconds": 5,
    },
    "paths": {
        "tinyworlds_dir": None,
        "ai_scientist_v2_dir": None,
        "output_dir": "runs/test_self",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        elif value is not None:
            out[key] = value
    return out


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = deep_merge(DEFAULT_CONFIG, load_yaml_config(args.config))
    cli_override = {
        "paths": {
            "tinyworlds_dir": args.tinyworlds_dir,
            "ai_scientist_v2_dir": args.ai_scientist_v2_dir,
            "output_dir": args.output_dir,
        },
        "model": {
            "model_url": args.model_url,
            "default_model": args.model_name,
            "temperature": args.temperature,
            "seed": args.seed,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_completion_tokens": args.max_completion_tokens,
            "mock_model": args.mock_model,
        },
        "compute": {
            "num_agents": args.num_agents,
            "max_depth_per_agent": args.max_depth_per_agent,
            "max_experiments_per_agent": args.max_experiments_per_agent,
            "max_total_experiments": args.max_total_experiments,
            "max_training_steps": args.max_training_steps,
            "max_runtime_minutes_per_experiment": args.max_runtime_minutes_per_experiment,
            "max_tokens_per_critique": args.max_tokens_per_critique,
        },
        "base_system": {
            "write_full_paper": args.write_full_paper,
            "reviewer_enabled": args.reviewer_enabled,
            "reviewer_backend": args.reviewer_backend,
            "reviewer_model": args.reviewer_model,
            "reviewer_model_url": args.reviewer_model_url,
        },
        "experiment": {
            "runner": args.runner,
            "train_command": args.train_command,
            "eval_command": args.eval_command,
            "dataset_config_path": args.dataset_config_path,
            "allowed_files": args.allowed_files,
            "forbidden_files": args.forbidden_files,
            "primary_metric": args.primary_metric,
            "ai_scientist_data_dir": args.ai_scientist_data_dir,
            "ai_scientist_config_template": args.ai_scientist_config_template,
            "ai_scientist_code_model": args.ai_scientist_code_model,
            "ai_scientist_feedback_model": args.ai_scientist_feedback_model,
        },
    }
    config = deep_merge(config, cli_override)
    if args.max_tokens_per_critique is not None:
        config["model"]["max_completion_tokens"] = args.max_tokens_per_critique
    if args.max_runtime_minutes_per_experiment is not None:
        runtime_seconds = int(float(args.max_runtime_minutes_per_experiment) * 60)
        config["experiment"]["codex_scientist_time_budget_seconds"] = runtime_seconds
        config["experiment"]["ai_scientist_time_budget_seconds"] = runtime_seconds
    return config


def validate_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition {condition!r}; expected one of {sorted(CONDITIONS)}")


def write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
