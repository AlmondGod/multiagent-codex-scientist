from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from commsci.artifacts import ensure_dir, utc_now, write_json, write_text

from .actions import action_diff, action_summary, apply_patch_recipe, initial_action, normalize_action, revise_action
from .prompts import worker_task_prompt
from .schemas import CodexNode
from .workspace import prepare_node_workspace


def run_codex_scientist_branch_expansion(
    artifact_dir: Path,
    agent_id: str,
    branch_id: str,
    step: int,
    config: dict[str, Any],
    seed: int,
    critique_context: str | None,
    revised_plan: str | None = None,
    previous_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition = artifact_dir.parent.parent.name
    node_id = f"{branch_id}_node_{step}"
    parent_id = f"{branch_id}_node_{step - 1}" if step > 1 else None
    node_root = ensure_dir(artifact_dir / "codex_scientist" / "nodes" / node_id)
    workspace = prepare_node_workspace(config, node_root)
    memory = load_memory(artifact_dir, agent_id)
    agent_index = int(agent_id.rsplit("_", 1)[-1])
    action_override = load_action_override(config, agent_id, step)
    if action_override:
        action = normalize_action(action_override, config, f"supervised_{agent_id}_step_{step}")
    else:
        action = (
            revise_action(
                agent_index=agent_index,
                config=config,
                current_action=previous_action or {},
                critique=critique_context or "",
                revised_plan=revised_plan or "",
            )
            if step > 1
            else initial_action(agent_index, config)
        )
    patch_result = apply_patch_recipe(workspace, action)
    action["patch_result"] = patch_result
    if patch_result.get("code_diff"):
        action["code_diff"] = patch_result["code_diff"]
    write_json(node_root / "action.json", action)
    write_json(node_root / "patch_result.json", patch_result)
    if patch_result.get("code_diff"):
        write_text(node_root / "code_diff.patch", patch_result["code_diff"])
    prompt = worker_task_prompt(
        node_id=node_id,
        parent_id=parent_id,
        agent_id=agent_id,
        branch_id=branch_id,
        step=step,
        task_spec=config["experiment"]["task_spec"],
        action=action,
        critique_context=critique_context,
        memory=memory,
    )
    write_text(node_root / "worker_task.md", prompt)
    write_text(node_root / "memory.md", "\n".join(memory) + ("\n" if memory else ""))

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    time_budget = config["experiment"].get("codex_scientist_time_budget_seconds",
                                           config["experiment"].get("ai_scientist_time_budget_seconds", 100))
    env.setdefault("TW_TIME_BUDGET", str(time_budget))
    baseline = config["experiment"].get("tinyworlds_baseline_knobs") or {}
    env.setdefault("TW_DATASET", str(baseline.get("TW_DATASET", "minigrid")))
    env.setdefault("TW_DEPTH", str(baseline.get("TW_DEPTH", "1")))
    for key, value in baseline.items():
        env.setdefault(key, str(value))
    for key, value in (action.get("env") or {}).items():
        env[key] = str(value)

    cmd = [sys.executable, str(workspace / "runfile.py")]
    timeout = int(config["experiment"].get("codex_scientist_timeout_seconds",
                                           config["experiment"].get("ai_scientist_timeout_seconds", 600))) + 120
    result = subprocess.run(
        cmd,
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    logs = f"$ {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n"
    write_text(node_root / "logs.txt", logs)
    metrics = parse_metrics(workspace / "working" / "metrics.json", logs, result.returncode)
    write_json(node_root / "metrics.json", metrics)

    node = CodexNode(
        node_id=node_id,
        parent_id=parent_id,
        agent_id=agent_id,
        branch_id=branch_id,
        condition=condition,
        depth=step,
        hypothesis=build_hypothesis(agent_id, step, action, config),
        action=action,
        metrics=metrics,
        command=cmd,
        workspace_path=str(workspace),
        artifact_paths=[str(node_root)],
        critique_received=critique_context,
        memory=memory,
        logs_summary=logs[:1200],
        failure=None if metrics.get("experiment_success") else metrics.get("error", "experiment failed"),
    )
    write_json(node_root / "node.json", node.to_dict())
    update_memory(artifact_dir, node)
    expansion = {
        "metrics": metrics,
        "logs": logs,
        "hypothesis": node.hypothesis,
        "experiment_plan": build_plan(step, action),
        "analysis": f"primary_score={metrics.get('primary_score', 'N/A')} val_mse={metrics.get('val_mse', 'N/A')}",
        "proposed_next_experiment": next_experiment(action, metrics),
        "code_diff": action_diff(action),
        "code_diff_summary": action_summary(action),
        "workspace_path": str(workspace),
        "artifact_paths": [str(node_root)],
        "codex_node": node.to_dict(),
        "ai_scientist_node": {},
        "action": action,
    }
    write_json(node_root / "branch_expansion.json", expansion)
    return expansion


def parse_metrics(metrics_path: Path, logs: str, returncode: int) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    loss = raw.get("val_mse", raw.get("score", raw.get("loss"))) if raw else None
    metrics: dict[str, Any] = {
        "experiment_success": returncode == 0 and loss is not None,
        "codex_scientist_returncode": returncode,
    }
    if loss is not None:
        metrics["val_mse"] = float(loss)
        metrics["primary_score"] = round(1.0 / (1.0 + float(loss)), 6)
    metrics.update({key: value for key, value in raw.items() if key not in metrics})
    if returncode != 0:
        metrics["error"] = tail(logs, 2000)
    return metrics


def load_action_override(config: dict[str, Any], agent_id: str, step: int) -> dict[str, Any] | None:
    override_dir = config["experiment"].get("codex_scientist_action_overrides_dir")
    if not override_dir:
        return None
    path = Path(override_dir).expanduser() / f"{agent_id}_step_{step}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_memory(artifact_dir: Path, agent_id: str) -> list[str]:
    path = artifact_dir / "codex_scientist" / "memory.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get(agent_id, []))


def update_memory(artifact_dir: Path, node: CodexNode) -> None:
    path = artifact_dir / "codex_scientist" / "memory.json"
    data: dict[str, list[str]] = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.setdefault(node.agent_id, [])
    entries.append(
        f"{utc_now()} {node.node_id}: success={node.metrics.get('experiment_success')} "
        f"primary_score={node.metrics.get('primary_score')} action={action_summary(node.action)}"
    )
    write_json(path, data)


def build_hypothesis(agent_id: str, step: int, action: dict[str, Any], config: dict[str, Any]) -> str:
    return (
        f"{agent_id} Codex-Scientist node {step}: {action_summary(action)} may improve "
        f"{config['experiment'].get('primary_metric', 'primary_score')} under the fixed TinyWorlds budget."
    )


def build_plan(step: int, action: dict[str, Any]) -> str:
    return f"Run Codex-Scientist node step {step} with {action_summary(action)}."


def next_experiment(action: dict[str, Any], metrics: dict[str, Any]) -> str:
    if not metrics.get("experiment_success"):
        return "Keep the canonical harness and choose a simpler validated knob recipe."
    return "Compare this action against another single-knob TinyWorlds recipe under the same budget."


def tail(text: str, chars: int) -> str:
    return text[-chars:] if len(text) > chars else text
