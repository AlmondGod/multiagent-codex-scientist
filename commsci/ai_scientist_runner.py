from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from .artifacts import ensure_dir, write_json, write_text

_CANONICAL_RUNFILE_TEMPLATE = """\
#!/usr/bin/env python3
# Canonical TinyWorlds harness. Called as subprocess from experiment_changes.py.
# Runs from workspace/input/ -- DO NOT MODIFY.
import json, os, subprocess, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TIME_BUDGET = int(os.environ.get("TW_TIME_BUDGET", "{time_budget}"))
_WORKING = os.path.join(_HERE, "working")
os.makedirs(_WORKING, exist_ok=True)
_METRICS_PATH = os.path.join(_WORKING, "metrics.json")

cmd = [
    sys.executable,
    os.path.join(_HERE, "train.py"),
    "--time_budget_sec", str(_TIME_BUDGET),
    "--depth", "1",
    "--dataset", "minigrid",
    "--out", _METRICS_PATH,
]
print(f"[runfile] Running: {{' '.join(cmd)}}", flush=True)
proc = subprocess.run(cmd, cwd=_HERE)
if proc.returncode != 0:
    print(f"[runfile] ERROR: train.py exited {{proc.returncode}}", file=sys.stderr, flush=True)
    sys.exit(proc.returncode)
if not os.path.exists(_METRICS_PATH):
    print(f"[runfile] ERROR: metrics file missing: {{_METRICS_PATH}}", file=sys.stderr, flush=True)
    sys.exit(1)
with open(_METRICS_PATH) as _f:
    _metrics = json.load(_f)
print(json.dumps(_metrics, indent=2))
_score = _metrics.get("score") or _metrics.get("val_mse") or _metrics.get("loss")
if _score is None:
    print("[runfile] ERROR: score/val_mse/loss missing from metrics.json", file=sys.stderr, flush=True)
    sys.exit(1)
print(f"score: {{_score}}")
"""

# NOTE: BFTS reads task_desc["Code"] (not "Code To Use") and shows it to the LLM
# as the starting-point template. The stage-1 goal explicitly says "If you are
# given Code To Use, use it as a starting point" -- so we provide a COMPLETE,
# ALREADY-WORKING script and ask the LLM to add one change before the run line.
#
# Path contract (BFTS workspace layout):
#   workspace/input/  <- copy of prepared_data_dir (train.py, models.py, runfile.py)
#   workspace/working/ <- exec cwd for experiment_changes.py (os.getcwd() here)
_EXPERIMENT_CHANGES_EXAMPLE = """\
# experiment_changes.py -- use THIS as your starting code. Do NOT rewrite it.
# BFTS workspace layout:
#   workspace/input/   <- TinyWorlds source files (train.py, models.py, runfile.py)
#   workspace/working/ <- this file runs here (os.getcwd())
#
# YOUR ONLY TASK: add ONE small targeted change in the marked section.
# Do NOT touch anything below "DO NOT MODIFY BELOW".
import os, pathlib, re, subprocess, sys

_INPUT = os.path.normpath(os.path.join(os.getcwd(), "..", "input"))

# === ADD EXACTLY ONE CHANGE HERE ===
# Patch a constant in models.py (example -- replace with your actual change):
#   _p = pathlib.Path(_INPUT) / "models.py"
#   _p.write_text(re.sub(r"(hidden_dim\\s*=\\s*)\\d+", r"\\g<1>128", _p.read_text(), count=1))
# Or write a config JSON that train.py reads:
#   (pathlib.Path(_INPUT) / "exp_config.json").write_text('{"learning_rate": 0.001}')
# ===================================

# DO NOT MODIFY BELOW: run canonical TinyWorlds harness and relay output
_r = subprocess.run(
    [sys.executable, os.path.join(_INPUT, "runfile.py")],
    cwd=_INPUT,
    capture_output=True,
    text=True,
)
print(_r.stdout)
if _r.stderr:
    print(_r.stderr, file=sys.stderr)
sys.exit(_r.returncode)
"""


def build_canonical_runfile(time_budget: int) -> str:
    return _CANONICAL_RUNFILE_TEMPLATE.format(time_budget=time_budget)


def prepare_ai_scientist_data_dir(config: dict[str, Any], run_root: Path) -> Path:
    """Create a prepared copy of the TinyWorlds dir with canonical runfile.py seeded.

    BFTS gets copy_data=True so it copies this dir to its own workspace.  Our
    runfile.py ends up in that workspace, and the LLM-generated experiment_changes.py
    cannot overwrite it because agent_file_name is different.
    """
    source = resolve_ai_scientist_data_dir(config)
    prepared = ensure_dir(run_root / "prepared_data_dir")
    if not (prepared / "train.py").exists():
        shutil.copytree(
            source,
            prepared,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".venv",
                "runs", "results", "working", "checkpoints", "wandb",
                "*.pt", "*.pth", "*.ckpt",
            ),
            dirs_exist_ok=True,
        )
        data_src = source / "data"
        if data_src.exists() and not (prepared / "data").exists():
            (prepared / "data").symlink_to(data_src, target_is_directory=True)
    time_budget = int(config["experiment"].get("ai_scientist_time_budget_seconds", 100))
    write_text(prepared / "runfile.py", build_canonical_runfile(time_budget))
    return prepared


def run_direct_tinyworlds_expansion(
    artifact_dir: Path,
    agent_id: str,
    branch_id: str,
    step: int,
    config: dict[str, Any],
    seed: int,
    critique_context: str | None,
    revised_plan: str | None = None,
) -> dict[str, Any]:
    """Run TinyWorlds directly, bypassing BFTS code generation.

    Used when BFTS LLM code generation is unreliable (e.g. model ignores template).
    Calls the canonical runfile.py from the prepared_data_dir.  The full protocol
    (critique, decision_change, reviewer) runs normally on top of the real scores.
    """
    run_root = ensure_dir(artifact_dir / "ai_scientist_v2" / f"step_{step}")
    data_dir = prepare_ai_scientist_data_dir(config, run_root)
    stdout_path = run_root / "stdout_stderr.log"

    env = os.environ.copy()
    env.setdefault("TW_TIME_BUDGET", str(config["experiment"].get("ai_scientist_time_budget_seconds", 100)))
    env.setdefault("WANDB_MODE", "disabled")
    ensure_local_ollama_credentials(config, env)

    runfile = data_dir / "runfile.py"
    cmd = [sys.executable, str(runfile)]
    result = subprocess.run(
        cmd,
        cwd=str(data_dir),
        env=env,
        capture_output=True,
        text=True,
        timeout=int(config["experiment"].get("ai_scientist_timeout_seconds", 600)) + 120,
    )
    combined_logs = f"$ {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
    write_text(stdout_path, combined_logs)

    metrics_path = data_dir / "working" / "metrics.json"
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = raw.get("score") or raw.get("val_mse") or raw.get("loss")
        metrics: dict[str, Any] = {
            "experiment_success": result.returncode == 0 and score is not None,
            "ai_scientist_returncode": result.returncode,
        }
        if score is not None:
            metrics["primary_score"] = score
            metrics["score"] = score
        metrics.update({k: v for k, v in raw.items() if k not in metrics})
    else:
        score = parse_results_score(combined_logs)
        metrics = {
            "experiment_success": result.returncode == 0 and score is not None,
            "ai_scientist_returncode": result.returncode,
        }
        if score is not None:
            metrics["primary_score"] = score
            metrics["score"] = score

    hypothesis = f"{agent_id} direct TinyWorlds run (step {step})"
    plan = "Direct TinyWorlds harness invocation via canonical runfile.py."
    if critique_context:
        hypothesis += f"\nCritique context: {critique_context[:300]}"
    if revised_plan:
        plan = revised_plan[:500]

    expansion = {
        "metrics": metrics,
        "logs": combined_logs,
        "hypothesis": hypothesis,
        "experiment_plan": plan,
        "analysis": f"score={metrics.get('primary_score', 'N/A')}  val_mse={metrics.get('val_mse', 'N/A')}",
        "proposed_next_experiment": "Vary TinyWorlds depth or model_type under same time budget.",
        "code_diff": "",
        "code_diff_summary": "Direct TinyWorlds execution — no code modification in this step.",
        "workspace_path": str(data_dir),
        "artifact_paths": [str(run_root)],
        "ai_scientist_node": {},
    }
    write_json(run_root / "branch_expansion.json", expansion)
    return expansion


def run_ai_scientist_branch_expansion(
    artifact_dir: Path,
    agent_id: str,
    branch_id: str,
    step: int,
    config: dict[str, Any],
    seed: int,
    critique_context: str | None,
    revised_plan: str | None = None,
) -> dict[str, Any]:
    """Run one branch expansion step.

    When experiment.ai_scientist_use_bfts is False (the default for smoke runs),
    calls run_direct_tinyworlds_expansion to bypass BFTS code generation and run
    the canonical TinyWorlds harness directly.  Set ai_scientist_use_bfts: true in
    the config only when the LLM reliably follows the seeded template.
    """
    use_bfts = bool(config["experiment"].get("ai_scientist_use_bfts", False))
    if not use_bfts:
        return run_direct_tinyworlds_expansion(
            artifact_dir, agent_id, branch_id, step, config, seed, critique_context, revised_plan
        )

    ai_dir = require_path(config["paths"].get("ai_scientist_v2_dir"), "AI-Scientist-v2")
    run_root = ensure_dir(artifact_dir / "ai_scientist_v2" / f"step_{step}")
    data_dir = prepare_ai_scientist_data_dir(config, run_root)
    task_desc_path = run_root / "task_desc.json"
    config_path = run_root / "bfts_config.yaml"
    stdout_path = run_root / "stdout_stderr.log"

    task_desc = build_task_description(agent_id, branch_id, step, config, critique_context, revised_plan)
    write_text(task_desc_path, json.dumps(task_desc, indent=2))
    bfts_config = build_bfts_config(config, data_dir, run_root, task_desc_path, agent_id, step)
    write_text(config_path, yaml.safe_dump(bfts_config, sort_keys=False))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ai_dir}:{env.get('PYTHONPATH', '')}"
    env.setdefault("TW_TIME_BUDGET", str(config["experiment"].get("ai_scientist_time_budget_seconds", 100)))
    env.setdefault("WANDB_MODE", "disabled")
    ensure_local_ollama_credentials(config, env)
    cmd = [
        sys.executable,
        "-c",
        (
            "from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager "
            "import perform_experiments_bfts; "
            f"perform_experiments_bfts({str(config_path)!r})"
        ),
    ]
    result = subprocess.run(
        cmd,
        cwd=ai_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=int(config["experiment"].get("ai_scientist_timeout_seconds", 600)) + 120,
    )
    combined_logs = f"$ {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
    write_text(stdout_path, combined_logs)
    expansion = extract_expansion(run_root, combined_logs, result.returncode, ai_dir)
    write_json(run_root / "branch_expansion.json", expansion)
    return expansion


def ensure_local_ollama_credentials(config: dict[str, Any], env: dict[str, str]) -> None:
    experiment = config["experiment"]
    models = [
        experiment.get("ai_scientist_code_model", ""),
        experiment.get("ai_scientist_feedback_model", ""),
    ]
    if any(str(model).startswith("ollama/") for model in models):
        env.setdefault("OPENAI_API_KEY", "ollama-local")
        env.setdefault("OLLAMA_API_KEY", "ollama-local")


def build_task_description(
    agent_id: str,
    branch_id: str,
    step: int,
    config: dict[str, Any],
    critique_context: str | None,
    revised_plan: str | None,
) -> dict[str, Any]:
    task_spec = config["experiment"]["task_spec"]
    time_budget = int(config["experiment"].get("ai_scientist_time_budget_seconds", 100))
    experiments = [
        (
            "Use the provided 'Code To Use' template EXACTLY as your starting point. "
            "The template is a complete, working experiment_changes.py -- do not rewrite it from scratch."
        ),
        (
            f"TinyWorlds source files (train.py, models.py, runfile.py) live in workspace/input/. "
            f"Your script runs in workspace/working/. Access input files via: "
            f"os.path.normpath(os.path.join(os.getcwd(), '..', 'input')). "
            f"runfile.py calls: python train.py --time_budget_sec {time_budget} "
            f"--depth 1 --dataset minigrid  and writes working/metrics.json."
        ),
        "Add EXACTLY ONE targeted change (e.g. patch one hyperparameter in models.py).",
        "Do NOT rewrite train.py, do NOT implement a training loop, do NOT modify runfile.py.",
        "Write raw executable Python only. Do not wrap code in Markdown fences.",
    ]
    if critique_context:
        experiments.append("Additional critique context before this expansion:\n" + critique_context)
    if revised_plan:
        experiments.append("Current revised plan from communication checkpoint:\n" + revised_plan)
    return {
        "Title": f"{agent_id} {branch_id} TinyWorlds branch expansion step {step}",
        "Abstract": task_spec,
        "Short Hypothesis": (
            "A real AI-Scientist-v2 branch expansion can identify a bounded TinyWorlds "
            "experiment that improves the configured primary metric."
        ),
        "Experiments": experiments,
        "Code": _EXPERIMENT_CHANGES_EXAMPLE,
        "Risk Factors and Limitations": [
            "runfile.py is the fixed harness contract. It must not be modified.",
            "experiment_changes.py must end by calling runfile.py via subprocess.",
            "This is a reduced-compute smoke and should not be interpreted as final scientific evidence.",
            "The branch expansion must preserve the communication ablation budget and avoid full paper writing.",
        ],
    }


def build_bfts_config(
    config: dict[str, Any],
    data_dir: Path,
    run_root: Path,
    task_desc_path: Path,
    agent_id: str,
    step: int,
) -> dict[str, Any]:
    template_path = config["experiment"].get("ai_scientist_config_template")
    if template_path:
        base = yaml.safe_load(Path(template_path).read_text(encoding="utf-8")) or {}
    else:
        base = {}
    code_model = config["experiment"].get("ai_scientist_code_model") or "ollama/qwen3:32b"
    feedback_model = config["experiment"].get("ai_scientist_feedback_model") or code_model
    timeout = int(config["experiment"].get("ai_scientist_timeout_seconds", 600))
    base.update(
        {
            "data_dir": str(data_dir),
            "preprocess_data": False,
            "desc_file": str(task_desc_path),
            "goal": None,
            "eval": None,
            "log_dir": str(run_root / "logs"),
            "workspace_dir": str(run_root / "workspaces"),
            "copy_data": True,
            "exp_name": f"{agent_id}_step_{step}",
            "exec": {
                "timeout": timeout,
                "agent_file_name": "experiment_changes.py",
                "format_tb_ipython": False,
            },
            "generate_report": False,
            "report": {"model": feedback_model, "temp": 0.2},
            "experiment": {"num_syn_datasets": 1},
            "debug": {"stage4": False},
            "agent": {
                "type": "parallel",
                "num_workers": 1,
                "steps": 1,
                "stages": {
                    "stage1_max_iters": 1,
                    "stage2_max_iters": 1,
                    "stage3_max_iters": 1,
                    "stage4_max_iters": 1,
                },
                "k_fold_validation": 1,
                "multi_seed_eval": {"num_seeds": 1},
                "expose_prediction": False,
                "data_preview": False,
                "code": {"model": code_model, "temp": 0.7, "max_tokens": 4096},
                "feedback": {"model": feedback_model, "temp": 0.2, "max_tokens": 2048},
                "vlm_feedback": {"model": feedback_model, "temp": 0.2, "max_tokens": None},
                "summary": {"model": feedback_model, "temp": 0.2},
                "search": {"max_debug_depth": 1, "debug_prob": 0.0, "num_drafts": 1},
            },
        }
    )
    return base


def extract_expansion(run_root: Path, logs: str, returncode: int, ai_dir: Path | None = None) -> dict[str, Any]:
    journal = load_latest_journal(run_root, ai_dir)
    if journal:
        write_json(run_root / "journal_snapshot.json", journal)
    node = latest_node(journal) if journal else {}
    metric_value = extract_metric_value(node, logs)
    metrics = {
        "experiment_success": returncode == 0 and not node.get("is_buggy", False),
        "ai_scientist_returncode": returncode,
    }
    if metric_value is not None:
        metrics["primary_score"] = metric_value
        metrics["ai_scientist_metric"] = metric_value
    parsed_score = parse_results_score(logs)
    if parsed_score is not None:
        metrics.setdefault("primary_score", parsed_score)
        metrics["score"] = parsed_score
    if node.get("exec_time") is not None:
        metrics["runtime_seconds"] = node.get("exec_time")
    return {
        "metrics": metrics,
        "logs": logs,
        "hypothesis": node.get("overall_plan") or node.get("plan") or "",
        "experiment_plan": node.get("plan") or "",
        "analysis": node.get("analysis") or "",
        "proposed_next_experiment": node.get("analysis") or node.get("plan") or "",
        "code_diff": node.get("code") or "",
        "code_diff_summary": summarize_code(node.get("code") or ""),
        "workspace_path": str(first_existing(run_root.glob("workspaces/*")) or (run_root / "workspaces")),
        "artifact_paths": [str(run_root)],
        "ai_scientist_node": node,
    }


def resolve_ai_scientist_data_dir(config: dict[str, Any]) -> Path:
    explicit = config["experiment"].get("ai_scientist_data_dir")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    tiny = config["paths"].get("tinyworlds_dir")
    if tiny:
        tiny_path = Path(tiny).expanduser()
        candidates.extend([tiny_path, tiny_path.parent / "tinyworlds-autoresearch"])
    for candidate in candidates:
        if (candidate / "train.py").exists() and (candidate / "models.py").exists() and (candidate / "setup.py").exists():
            return candidate.resolve()
    raise RuntimeError(
        "AI-Scientist-v2 TinyWorlds runner requires a tinyworlds-autoresearch harness "
        "containing train.py, models.py, and setup.py. Set experiment.ai_scientist_data_dir "
        "or pass --ai_scientist_data_dir."
    )


def require_path(path_value: str | None, label: str) -> Path:
    if not path_value:
        raise RuntimeError(f"{label} path is required.")
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"{label} path does not exist: {path}")
    return path


def load_latest_journal(run_root: Path, ai_dir: Path | None = None) -> dict[str, Any] | None:
    journals = sorted(run_root.glob("logs/*/stage_*/journal.json"), key=lambda p: p.stat().st_mtime)
    if not journals:
        return load_latest_manager_journal(run_root, ai_dir)
    return json.loads(journals[-1].read_text(encoding="utf-8"))


def load_latest_manager_journal(run_root: Path, ai_dir: Path | None) -> dict[str, Any] | None:
    managers = sorted(run_root.glob("logs/*/manager.pkl"), key=lambda p: p.stat().st_mtime)
    if not managers:
        return None
    if ai_dir and str(ai_dir) not in sys.path:
        sys.path.insert(0, str(ai_dir))
    try:
        manager = pickle.loads(managers[-1].read_bytes())
    except Exception:
        return None
    journals = list(getattr(manager, "journals", {}).items())
    if not journals:
        return None
    stage_name, journal = journals[-1]
    return {
        "source": str(managers[-1]),
        "stage": stage_name,
        "nodes": [node_to_dict(node) for node in getattr(journal, "nodes", [])],
    }


def node_to_dict(node: Any) -> dict[str, Any]:
    metric = getattr(node, "metric", None)
    metric_value = getattr(metric, "value", None)
    return {
        "id": getattr(node, "id", None),
        "plan": getattr(node, "plan", "") or "",
        "overall_plan": getattr(node, "overall_plan", "") or "",
        "analysis": getattr(node, "analysis", "") or "",
        "code": getattr(node, "code", "") or "",
        "metric": {
            "value": metric_value if isinstance(metric_value, (int, float)) else None,
            "name": getattr(metric, "name", None),
            "description": getattr(metric, "description", None),
            "maximize": getattr(metric, "maximize", None),
        },
        "is_buggy": getattr(node, "is_buggy", None),
        "exec_time": getattr(node, "exec_time", None),
        "exc_type": getattr(node, "exc_type", None),
        "exc_info": getattr(node, "exc_info", None),
        "exp_results_dir": getattr(node, "exp_results_dir", None),
    }


def latest_node(journal: dict[str, Any]) -> dict[str, Any]:
    nodes = journal.get("nodes") or []
    return nodes[-1] if nodes else {}


def extract_metric_value(node: dict[str, Any], logs: str) -> float | None:
    metric = node.get("metric")
    if isinstance(metric, dict):
        for key in ("value", "final_value", "best_value"):
            if isinstance(metric.get(key), (int, float)):
                return float(metric[key])
    if isinstance(metric, (int, float)):
        return float(metric)
    match = re.search(r"score['\"]?\s*[:=]\s*([0-9.eE+-]+)", logs)
    return float(match.group(1)) if match else None


def parse_results_score(logs: str) -> float | None:
    match = re.search(r'"score"\s*:\s*([0-9.eE+-]+)', logs)
    return float(match.group(1)) if match else None


def summarize_code(code: str) -> str:
    lines = code.splitlines()
    if not lines:
        return "No AI-Scientist-v2 code captured."
    return "\n".join(lines[:80])


def first_existing(paths: Any) -> Path | None:
    for path in paths:
        return path
    return None
