#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from commsci.ai_scientist_v2_wrapper import (
    ensure_ai_scientist_v2,
    propose_initial_branch,
    propose_next_experiment,
    run_reviewer,
)
from commsci.ai_scientist_runner import run_ai_scientist_branch_expansion
from commsci.artifacts import (
    agent_artifact_dir,
    agent_workspace_dir,
    ensure_dir,
    environment_info,
    git_commit_hash,
    utc_now,
    write_json,
    write_text,
)
from commsci.config import build_config, bool_arg, validate_condition, write_config
from commsci.critique import assign_roles, run_critique_round
from commsci.evaluation import aggregate_run
from commsci.model_client import OpenAICompatibleClient
from commsci.tinyworlds_runner import run_experiment
from commsci.workspace import collect_git_diff, prepare_agent_workspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal peer/self critique TinyWorlds ablation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--condition", required=True, choices=["self_critique", "peer_critique", "peer_critique_with_roles"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_agents", type=int, default=None)
    parser.add_argument("--tinyworlds_dir", default=None)
    parser.add_argument("--ai_scientist_v2_dir", default=None)
    parser.add_argument("--model_url", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_depth_per_agent", type=int, default=None)
    parser.add_argument("--max_experiments_per_agent", type=int, default=None)
    parser.add_argument("--max_total_experiments", type=int, default=None)
    parser.add_argument("--max_training_steps", type=int, default=None)
    parser.add_argument("--max_runtime_minutes_per_experiment", type=int, default=None)
    parser.add_argument("--max_tokens_per_critique", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--max_completion_tokens", type=int, default=None)
    parser.add_argument("--mock_model", action="store_true", default=None)
    parser.add_argument("--allowed_files", nargs="*", default=None)
    parser.add_argument("--forbidden_files", nargs="*", default=None)
    parser.add_argument("--write_full_paper", type=bool_arg, default=None)
    parser.add_argument("--reviewer_enabled", type=bool_arg, default=None)
    parser.add_argument("--reviewer_backend", default=None)
    parser.add_argument("--reviewer_model", default=None)
    parser.add_argument("--reviewer_model_url", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--train_command", default=None)
    parser.add_argument("--eval_command", default=None)
    parser.add_argument("--dataset_config_path", default=None)
    parser.add_argument("--primary_metric", default=None)
    parser.add_argument("--runner", choices=["tinyworlds_command", "ai_scientist_v2"], default=None)
    parser.add_argument("--ai_scientist_data_dir", default=None)
    parser.add_argument("--ai_scientist_config_template", default=None)
    parser.add_argument("--ai_scientist_code_model", default=None)
    parser.add_argument("--ai_scientist_feedback_model", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_condition(args.condition)
    config = build_config(args)
    num_agents = int(config["compute"]["num_agents"])
    max_total = int(config["compute"]["max_total_experiments"])
    if max_total < num_agents * 2:
        raise RuntimeError("max_total_experiments must be at least num_agents * 2 for v0.")
    output_dir = Path(config["paths"]["output_dir"]).expanduser().resolve()
    condition_dir = output_dir / args.condition
    ensure_dir(condition_dir)
    seed = int(config["model"]["seed"])
    run_id = output_dir.name
    roles = assign_roles(num_agents, seed)
    ai_status = ensure_ai_scientist_v2(config, args.dry_run)
    client = OpenAICompatibleClient(
        model_url=config["model"]["model_url"],
        model_name=config["model"]["default_model"],
        temperature=float(config["model"]["temperature"]),
        seed=seed,
        max_completion_tokens=int(config["model"]["max_completion_tokens"]),
        dry_run=args.dry_run or bool(config["model"].get("mock_model")),
    )
    global_dir = ensure_dir(output_dir / "global")
    write_config(global_dir / "config.yaml", config)
    write_json(
        global_dir / f"metadata_{args.condition}.json",
        {
            "run_id": run_id,
            "condition": args.condition,
            "seed": seed,
            "dry_run": args.dry_run,
            "timestamp": utc_now(),
            "argv": sys.argv,
            "tinyworlds_git_commit": git_commit_hash(config["paths"].get("tinyworlds_dir")),
            "ai_scientist_v2_git_commit": git_commit_hash(config["paths"].get("ai_scientist_v2_dir")),
            "ai_scientist_v2_status": ai_status,
            "model_endpoint": config["model"]["model_url"],
            "model_name": config["model"]["default_model"],
            "roles": roles if args.condition == "peer_critique_with_roles" else {},
            "environment": environment_info(),
        },
    )
    for agent_index in range(num_agents):
        run_agent_first_step(output_dir, args.condition, agent_index, run_id, roles, config, args.dry_run, seed)
    run_critique_round(
        output_dir=output_dir,
        condition=args.condition,
        num_agents=num_agents,
        seed=seed,
        max_prompt_tokens=int(config["model"]["max_prompt_tokens"]),
        client=client,
        roles=roles,
    )
    for agent_index in range(num_agents):
        run_agent_second_step(output_dir, args.condition, agent_index, config, args.dry_run, seed, client)
    aggregate_run(output_dir)
    print(f"Completed {args.condition} run in {output_dir}")
    return 0


def run_agent_first_step(
    output_dir: Path,
    condition: str,
    agent_index: int,
    run_id: str,
    roles: dict[str, str],
    config: dict[str, Any],
    dry_run: bool,
    seed: int,
) -> None:
    agent_id = f"agent_{agent_index}"
    artifact_dir = ensure_dir(agent_artifact_dir(output_dir, condition, agent_id))
    workspace_dir = agent_workspace_dir(output_dir, condition, agent_id)
    workspace_mode = "ai_scientist_v2" if use_ai_scientist_runner(config, dry_run) else prepare_agent_workspace(
        config["paths"].get("tinyworlds_dir"),
        workspace_dir,
        dry_run,
        f"commsci-{condition}-{agent_id}",
    )
    write_config(artifact_dir / "config.yaml", config)
    ensure_dir(artifact_dir / "prompts")
    ensure_dir(artifact_dir / "completions")
    hypothesis, plan1 = propose_initial_branch(agent_index, config)
    write_text(artifact_dir / "hypothesis.md", hypothesis + "\n")
    write_text(artifact_dir / "experiment_plan_1.md", plan1 + "\n")
    baseline_metrics = {"primary_score": 0.5, "experiment_success": True} if dry_run else {}
    try:
        if use_ai_scientist_runner(config, dry_run):
            expansion = run_ai_scientist_branch_expansion(
                artifact_dir=artifact_dir,
                agent_id=agent_id,
                branch_id=f"{condition}_{agent_id}",
                step=1,
                config=config,
                seed=seed,
                critique_context=None,
            )
            metrics1, logs1 = expansion["metrics"], expansion["logs"]
            hypothesis = expansion.get("hypothesis") or hypothesis
            plan1 = expansion.get("experiment_plan") or plan1
            workspace_dir = Path(expansion.get("workspace_path") or workspace_dir)
        else:
            expansion = {}
            metrics1, logs1 = run_experiment(workspace_dir, artifact_dir, agent_index, 1, config, dry_run, seed)
        failure_notes = ""
    except Exception as exc:
        expansion = {}
        metrics1, logs1 = {"experiment_success": False, "error": str(exc)}, str(exc)
        failure_notes = str(exc)
    write_json(artifact_dir / "metrics_experiment_1.json", metrics1)
    write_text(artifact_dir / "logs_experiment_1.txt", logs1)
    diff = expansion.get("code_diff", "") if expansion else collect_git_diff(workspace_dir)
    if dry_run and not diff:
        diff = f"diff --git a/world_model.py b/world_model.py\n+ dry-run branch edit for {agent_id}\n"
    write_text(artifact_dir / "git_diff.patch", diff)
    summary = {
        "run_id": run_id,
        "condition": condition,
        "agent_id": agent_id,
        "branch_id": f"{condition}_{agent_id}",
        "role": roles.get(agent_id) if condition == "peer_critique_with_roles" else None,
        "hypothesis": hypothesis,
        "experiment_plan_1": plan1,
        "code_diff_summary": expansion.get("code_diff_summary") if expansion else summarize_diff(diff),
        "baseline_metrics": baseline_metrics,
        "metrics_experiment_1": metrics1,
        "logs_summary": logs1[:1200],
        "failure_notes": failure_notes,
        "current_interpretation": expansion.get("analysis") if expansion else interpretation(metrics1),
        "proposed_next_experiment": expansion.get("proposed_next_experiment") if expansion else propose_next_experiment(agent_index),
        "workspace_path": str(workspace_dir),
        "artifact_paths": [str(artifact_dir), *expansion.get("artifact_paths", [])] if expansion else [str(artifact_dir)],
    }
    write_json(artifact_dir / "branch_summary.json", summary)
    write_json(
        artifact_dir / "metadata.json",
        {
            "agent_id": agent_id,
            "workspace_mode": workspace_mode,
            "workspace_path": str(workspace_dir),
            "created_at": utc_now(),
        },
    )


def run_agent_second_step(
    output_dir: Path,
    condition: str,
    agent_index: int,
    config: dict[str, Any],
    dry_run: bool,
    seed: int,
    client: OpenAICompatibleClient,
) -> None:
    agent_id = f"agent_{agent_index}"
    artifact_dir = agent_artifact_dir(output_dir, condition, agent_id)
    workspace_dir = agent_workspace_dir(output_dir, condition, agent_id)
    critique = (artifact_dir / "critique.md").read_text(encoding="utf-8")
    decision_prompt = f"""Given this critique, decide whether the second TinyWorlds experiment should change.
Return JSON with decision_changed, change_type, reason, and revised_experiment_plan.

Critique:
{critique}
"""
    decision_response = client.complete(decision_prompt, condition, artifact_dir, "decision_change")
    decision = parse_decision(decision_response.text)
    revised_plan = decision["revised_experiment_plan"]
    write_text(artifact_dir / "experiment_plan_2.md", revised_plan + "\n")
    try:
        if use_ai_scientist_runner(config, dry_run):
            expansion = run_ai_scientist_branch_expansion(
                artifact_dir=artifact_dir,
                agent_id=agent_id,
                branch_id=f"{condition}_{agent_id}",
                step=2,
                config=config,
                seed=seed,
                critique_context=critique,
                revised_plan=revised_plan,
            )
            metrics2, logs2 = expansion["metrics"], expansion["logs"]
            if expansion.get("code_diff"):
                write_text(artifact_dir / "git_diff.patch", expansion["code_diff"])
        else:
            metrics2, logs2 = run_experiment(workspace_dir, artifact_dir, agent_index, 2, config, dry_run, seed)
    except Exception as exc:
        metrics2, logs2 = {"experiment_success": False, "error": str(exc)}, str(exc)
    metrics1 = json.loads((artifact_dir / "metrics_experiment_1.json").read_text(encoding="utf-8"))
    decision["later_helped"], decision["evidence"] = helped_evidence(metrics1, metrics2, decision)
    write_json(artifact_dir / "decision_change.json", decision)
    write_json(artifact_dir / "metrics_experiment_2.json", metrics2)
    write_text(artifact_dir / "logs_experiment_2.txt", logs2)
    note = research_note(artifact_dir, decision, metrics1, metrics2, critique)
    note_path = artifact_dir / "research_note.md"
    write_text(note_path, note)
    run_reviewer(note_path, artifact_dir / "review.json", dry_run, config)


def summarize_diff(diff: str) -> str:
    lines = [line for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    return "\n".join(lines[:40]) if lines else "No code diff captured."


def use_ai_scientist_runner(config: dict[str, Any], dry_run: bool) -> bool:
    return (not dry_run) and config["experiment"].get("runner") == "ai_scientist_v2"


def interpretation(metrics: dict[str, Any]) -> str:
    if not metrics.get("experiment_success"):
        return "Experiment failed; debugging or narrower controls are needed."
    return f"Primary score after experiment 1 is {metrics.get('primary_score', 'unknown')} under the bounded budget."


def parse_decision(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    change_type = data.get("change_type", "added_ablation")
    allowed = {
        "added_control",
        "changed_metric",
        "abandoned_idea",
        "debugged_code",
        "added_ablation",
        "changed_hyperparameter",
        "changed_interpretation",
        "no_change",
    }
    if change_type not in allowed:
        change_type = "added_ablation"
    return {
        "input_received": text,
        "decision_changed": bool(data.get("decision_changed", True)),
        "change_type": change_type,
        "reason": data.get("reason", "Critique suggested a more controlled second experiment."),
        "revised_experiment_plan": data.get(
            "revised_experiment_plan",
            "Run a controlled second experiment under the same TinyWorlds budget.",
        ),
        "later_helped": None,
        "evidence": None,
    }


def helped_evidence(metrics1: dict[str, Any], metrics2: dict[str, Any], decision: dict[str, Any]) -> tuple[bool | None, str]:
    if not decision.get("decision_changed"):
        return False, "No critique-induced decision change was recorded."
    if not metrics1.get("experiment_success") and metrics2.get("experiment_success"):
        return True, "The revised experiment avoided a previous failure."
    score1 = metrics1.get("primary_score")
    score2 = metrics2.get("primary_score")
    if isinstance(score1, (int, float)) and isinstance(score2, (int, float)):
        return score2 > score1, f"Primary score changed from {score1} to {score2}."
    return None, "No comparable primary metric was available."


def research_note(
    artifact_dir: Path,
    decision: dict[str, Any],
    metrics1: dict[str, Any],
    metrics2: dict[str, Any],
    critique: str,
) -> str:
    branch = json.loads((artifact_dir / "branch_summary.json").read_text(encoding="utf-8"))
    return f"""# {branch['agent_id']} TinyWorlds Research Note

## Hypothesis
{branch['hypothesis']}

## Method
{branch['experiment_plan_1']}

## Experiment 1 result
{json.dumps(metrics1, indent=2, sort_keys=True)}

## Critique received
{critique}

## Decision change
{json.dumps(decision, indent=2, sort_keys=True)}

## Experiment 2 result
{json.dumps(metrics2, indent=2, sort_keys=True)}

## Claim-to-evidence table
| Claim | Evidence |
| --- | --- |
| The second experiment followed critique input. | decision_change.json records change_type={decision['change_type']}. |
| TinyWorlds metric changed after revision. | metrics_experiment_1.json and metrics_experiment_2.json report primary_score when available. |

## Limitations
This v0 note is intentionally short and bounded to two experiment meta-steps. Dry-run outputs are orchestration tests, not scientific evidence.

## Next step
Run the same condition on real TinyWorlds with matched training, runtime, token, and experiment budgets.
"""


if __name__ == "__main__":
    raise SystemExit(main())
