#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from commsci.ai_scientist_v2_wrapper import (
    ensure_ai_scientist_v2,
    propose_initial_branch,
    propose_next_experiment,
    run_reviewer,
)
from commsci.ai_scientist_runner import run_ai_scientist_branch_expansion
from commsci.codex_scientist import (
    codex_decision,
    codex_reviewer,
    run_codex_critique_round,
    run_codex_scientist_branch_expansion,
)
from commsci.codex_scientist.communication import check_artifact_completeness
from commsci.tinyworlds_knobs import (
    allowlist_from_config,
    initial_knobs_for_agent,
    select_knobs,
    summarize_knobs,
)
from commsci.artifacts import (
    agent_artifact_dir,
    agent_workspace_dir,
    ensure_dir,
    environment_info,
    git_commit_hash,
    read_json,
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
    parser.add_argument("--runner", choices=["tinyworlds_command", "ai_scientist_v2", "codex_scientist"], default=None)
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
    ai_status = (
        {"status": "not_required", "path": config["paths"].get("ai_scientist_v2_dir")}
        if use_codex_scientist_runner(config, args.dry_run)
        else ensure_ai_scientist_v2(config, args.dry_run)
    )
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
    if use_codex_scientist_runner(config, args.dry_run):
        run_codex_agent_steps_parallel(
            step=1,
            output_dir=output_dir,
            condition=args.condition,
            num_agents=num_agents,
            run_id=run_id,
            roles=roles,
            config=config,
            dry_run=args.dry_run,
            seed=seed,
            client=client,
        )
    else:
        for agent_index in range(num_agents):
            run_agent_first_step(output_dir, args.condition, agent_index, run_id, roles, config, args.dry_run, seed)
    if use_codex_scientist_runner(config, args.dry_run):
        wait_for_codex_live_overrides(output_dir, args.condition, num_agents, seed, config)
        run_codex_critique_round(
            output_dir=output_dir,
            condition=args.condition,
            num_agents=num_agents,
            seed=seed,
            config=config,
            roles=roles,
        )
    else:
        run_critique_round(
            output_dir=output_dir,
            condition=args.condition,
            num_agents=num_agents,
            seed=seed,
            max_prompt_tokens=int(config["model"]["max_prompt_tokens"]),
            client=client,
            roles=roles,
        )
    if use_codex_scientist_runner(config, args.dry_run):
        run_codex_agent_steps_parallel(
            step=2,
            output_dir=output_dir,
            condition=args.condition,
            num_agents=num_agents,
            run_id=run_id,
            roles=roles,
            config=config,
            dry_run=args.dry_run,
            seed=seed,
            client=client,
        )
    else:
        for agent_index in range(num_agents):
            run_agent_second_step(output_dir, args.condition, agent_index, config, args.dry_run, seed, client)
    aggregate_run(output_dir)
    print(f"Completed {args.condition} run in {output_dir}")
    return 0


def wait_for_codex_live_overrides(
    output_dir: Path,
    condition: str,
    num_agents: int,
    seed: int,
    config: dict[str, Any],
) -> None:
    experiment = config.get("experiment", {})
    if not experiment.get("codex_scientist_wait_for_live_overrides"):
        return
    required = expected_live_override_paths(condition, num_agents, seed, experiment)
    if not required:
        return
    timeout = float(experiment.get("codex_scientist_live_override_timeout_seconds", 1800))
    poll = float(experiment.get("codex_scientist_live_override_poll_seconds", 5))
    deadline = time.monotonic() + timeout
    marker = ensure_dir(output_dir / "global") / f"waiting_for_live_overrides_{condition}.json"
    write_json(
        marker,
        {
            "condition": condition,
            "started_at": utc_now(),
            "required": {name: [str(path) for path in alternatives] for name, alternatives in required.items()},
        },
    )
    print(f"Waiting for live Codex override artifacts for {condition}. Marker: {marker}", flush=True)
    while True:
        missing = {
            name: alternatives
            for name, alternatives in required.items()
            if not any(path.exists() for path in alternatives)
        }
        if not missing:
            write_json(
                marker,
                {
                    "condition": condition,
                    "completed_at": utc_now(),
                    "status": "ready",
                    "required": {name: [str(path) for path in alternatives] for name, alternatives in required.items()},
                },
            )
            print(f"Live Codex override artifacts are ready for {condition}.", flush=True)
            return
        if time.monotonic() >= deadline:
            missing_text = "\n".join(
                f"{name}: one of {[str(path) for path in alternatives]}"
                for name, alternatives in sorted(missing.items())
            )
            raise TimeoutError(f"Timed out waiting for live Codex overrides:\n{missing_text}")
        print(f"Still waiting for {len(missing)} live override groups for {condition}.", flush=True)
        time.sleep(max(1.0, poll))


def expected_live_override_paths(
    condition: str,
    num_agents: int,
    seed: int,
    experiment: dict[str, Any],
) -> dict[str, list[Path]]:
    required: dict[str, list[Path]] = {}
    action_override_dir = experiment.get("codex_scientist_action_overrides_dir")
    critique_override_dir = experiment.get("codex_scientist_critique_overrides_dir")
    decision_override_dir = experiment.get("codex_scientist_decision_overrides_dir")
    action_dir = Path(str(action_override_dir)).expanduser() if action_override_dir else None
    critique_dir = Path(str(critique_override_dir)).expanduser() if critique_override_dir else None
    decision_dir = Path(str(decision_override_dir)).expanduser() if decision_override_dir else None
    del seed  # Role assignment is encoded in critic prompts; filenames stay condition/agent based.
    for idx in range(num_agents):
        target_id = f"agent_{idx}"
        critic_id = target_id if condition == "self_critique" or num_agents == 1 else f"agent_{(idx + 1) % num_agents}"
        if critique_dir:
            required[f"{target_id}_critique"] = [
                critique_dir / f"{target_id}_critique{suffix}" for suffix in (".json", ".md", ".txt")
            ] + [
                critique_dir / f"{critic_id}_to_{target_id}{suffix}" for suffix in (".json", ".md", ".txt")
            ] + [
                critique_dir / f"{condition}_{target_id}_critique{suffix}" for suffix in (".json", ".md", ".txt")
            ]
        if decision_dir:
            required[f"{target_id}_decision"] = [
                decision_dir / f"{target_id}_decision{suffix}" for suffix in (".json", ".md", ".txt")
            ] + [
                decision_dir / f"{target_id}_step_2_decision{suffix}" for suffix in (".json", ".md", ".txt")
            ]
        if action_dir:
            required[f"{target_id}_step_2_action"] = [action_dir / f"{target_id}_step_2.json"]
    return required


def run_codex_agent_steps_parallel(
    *,
    step: int,
    output_dir: Path,
    condition: str,
    num_agents: int,
    run_id: str,
    roles: dict[str, str],
    config: dict[str, Any],
    dry_run: bool,
    seed: int,
    client: OpenAICompatibleClient,
) -> None:
    max_workers = int(config["experiment"].get("codex_scientist_parallel_workers", num_agents))
    max_workers = max(1, min(num_agents, max_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for agent_index in range(num_agents):
            if step == 1:
                future = pool.submit(
                    run_agent_first_step,
                    output_dir,
                    condition,
                    agent_index,
                    run_id,
                    roles,
                    config,
                    dry_run,
                    seed,
                )
            elif step == 2:
                future = pool.submit(
                    run_agent_second_step,
                    output_dir,
                    condition,
                    agent_index,
                    config,
                    dry_run,
                    seed,
                    client,
                )
            else:
                raise ValueError(f"Unsupported Codex-Scientist step {step}")
            futures[future] = agent_index
        for future in as_completed(futures):
            agent_index = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"Codex-Scientist agent_{agent_index} step {step} failed") from exc


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
    if use_codex_scientist_runner(config, dry_run):
        workspace_mode = "codex_scientist"
    elif use_ai_scientist_runner(config, dry_run):
        workspace_mode = "ai_scientist_v2"
    else:
        workspace_mode = prepare_agent_workspace(
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
    initial_env: dict[str, str] = {}
    initial_applied: dict[str, Any] = {}
    if use_ai_scientist_runner(config, dry_run):
        allowlist = allowlist_from_config(config)
        initial_env, initial_applied, initial_dropped = initial_knobs_for_agent(agent_index, allowlist)
        write_json(
            artifact_dir / "initial_knobs.json",
            {"applied": initial_applied, "env": initial_env, "dropped": initial_dropped},
        )
    expansion = {}
    try:
        if use_codex_scientist_runner(config, dry_run):
            expansion = run_codex_scientist_branch_expansion(
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
        elif use_ai_scientist_runner(config, dry_run):
            expansion = run_ai_scientist_branch_expansion(
                artifact_dir=artifact_dir,
                agent_id=agent_id,
                branch_id=f"{condition}_{agent_id}",
                step=1,
                config=config,
                seed=seed,
                critique_context=None,
                knob_overrides=initial_env,
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
        "codex_node": expansion.get("codex_node") if expansion else None,
        "action": expansion.get("action") if expansion else None,
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
    if use_codex_scientist_runner(config, dry_run):
        metrics1_for_decision = json.loads((artifact_dir / "metrics_experiment_1.json").read_text(encoding="utf-8"))
        decision = codex_decision(critique, metrics1_for_decision, config, agent_id)
        write_text(artifact_dir / "prompts" / "decision_change.txt", decision_prompt)
        write_text(artifact_dir / "completions" / "decision_change.txt", json.dumps(decision, indent=2))
    else:
        decision_response = client.complete(decision_prompt, condition, artifact_dir, "decision_change")
        decision = parse_decision(decision_response.text)
    revised_plan = decision["revised_experiment_plan"]
    if not isinstance(revised_plan, str):
        revised_plan = json.dumps(revised_plan, indent=2)
    write_text(artifact_dir / "experiment_plan_2.md", revised_plan + "\n")
    knob_overrides: dict[str, str] = {}
    if use_ai_scientist_runner(config, dry_run):
        allowlist = allowlist_from_config(config)
        # Start from this agent's step-1 configuration so the critique change builds on
        # the agent's own branch rather than resetting to defaults.
        initial_path = artifact_dir / "initial_knobs.json"
        initial = read_json(initial_path) if initial_path.exists() else {}
        initial_env = initial.get("env", {}) if isinstance(initial, dict) else {}
        initial_applied = initial.get("applied", {}) if isinstance(initial, dict) else {}
        critique_env, critique_applied, dropped_knobs = select_knobs(
            client, critique, revised_plan, allowlist, condition, artifact_dir,
            current_knobs=initial_applied,
        )
        knob_overrides = {**initial_env, **critique_env}
        applied_knobs = {**initial_applied, **critique_applied}
        decision["applied_knobs"] = applied_knobs
        decision["initial_knobs"] = initial_applied
        decision["critique_knob_changes"] = critique_applied
        write_json(
            artifact_dir / "applied_knobs.json",
            {
                "applied": applied_knobs,
                "env": knob_overrides,
                "initial": initial_applied,
                "critique_changes": critique_applied,
                "dropped": dropped_knobs,
            },
        )
        write_text(artifact_dir / "applied_knobs.md", summarize_knobs(applied_knobs) + "\n")
    expansion = {}
    try:
        if use_codex_scientist_runner(config, dry_run):
            previous_action = read_latest_codex_action(artifact_dir, f"{condition}_{agent_id}", 1)
            expansion = run_codex_scientist_branch_expansion(
                artifact_dir=artifact_dir,
                agent_id=agent_id,
                branch_id=f"{condition}_{agent_id}",
                step=2,
                config=config,
                seed=seed,
                critique_context=critique,
                revised_plan=revised_plan,
                previous_action=previous_action,
            )
            metrics2, logs2 = expansion["metrics"], expansion["logs"]
            if expansion.get("code_diff"):
                write_text(artifact_dir / "git_diff.patch", expansion["code_diff"])
        elif use_ai_scientist_runner(config, dry_run):
            expansion = run_ai_scientist_branch_expansion(
                artifact_dir=artifact_dir,
                agent_id=agent_id,
                branch_id=f"{condition}_{agent_id}",
                step=2,
                config=config,
                seed=seed,
                critique_context=critique,
                revised_plan=revised_plan,
                knob_overrides=knob_overrides,
            )
            metrics2, logs2 = expansion["metrics"], expansion["logs"]
            if expansion.get("code_diff"):
                write_text(artifact_dir / "git_diff.patch", expansion["code_diff"])
        else:
            metrics2, logs2 = run_experiment(workspace_dir, artifact_dir, agent_index, 2, config, dry_run, seed)
    except Exception as exc:
        metrics2, logs2 = {"experiment_success": False, "error": str(exc)}, str(exc)
    metrics1 = json.loads((artifact_dir / "metrics_experiment_1.json").read_text(encoding="utf-8"))
    if expansion:
        step2_action = expansion.get("action") or {}
        inheritance = step2_action.get("inheritance") or {}
        if inheritance:
            decision["cultural_operator"] = inheritance.get("mode", decision.get("cultural_operator"))
            decision["source_agent_ids"] = inheritance.get("source_agent_ids", decision.get("source_agent_ids", []))
            decision["source_node_ids"] = inheritance.get("source_node_ids", decision.get("source_node_ids", []))
            decision["copied_recipe_id"] = inheritance.get("copied_recipe_id", decision.get("copied_recipe_id"))
            decision["recombined_recipe_ids"] = inheritance.get(
                "recombined_recipe_ids", decision.get("recombined_recipe_ids", [])
            )
            decision["rejected_recipe_id"] = inheritance.get("rejected_recipe_id", decision.get("rejected_recipe_id"))
    decision["later_helped"], decision["evidence"] = helped_evidence(metrics1, metrics2, decision)
    write_json(artifact_dir / "decision_change.json", decision)
    write_json(artifact_dir / "metrics_experiment_2.json", metrics2)
    write_text(artifact_dir / "logs_experiment_2.txt", logs2)
    note = research_note(artifact_dir, decision, metrics1, metrics2, critique)
    note_path = artifact_dir / "research_note.md"
    write_text(note_path, note)
    if use_codex_scientist_runner(config, dry_run):
        codex_reviewer(note_path, artifact_dir / "review.json")
        write_json(artifact_dir / "artifact_completeness.json", check_artifact_completeness(artifact_dir))
    else:
        run_reviewer(note_path, artifact_dir / "review.json", dry_run, config)


def summarize_diff(diff: str) -> str:
    lines = [line for line in diff.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    return "\n".join(lines[:40]) if lines else "No code diff captured."


def use_ai_scientist_runner(config: dict[str, Any], dry_run: bool) -> bool:
    return (not dry_run) and config["experiment"].get("runner") == "ai_scientist_v2"


def use_codex_scientist_runner(config: dict[str, Any], dry_run: bool) -> bool:
    return (not dry_run) and config["experiment"].get("runner") == "codex_scientist"


def read_latest_codex_action(artifact_dir: Path, branch_id: str, step: int) -> dict[str, Any]:
    path = artifact_dir / "codex_scientist" / "nodes" / f"{branch_id}_node_{step}" / "action.json"
    return read_json(path) if path.exists() else {}


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
        "cultural_operator": data.get("cultural_operator", data.get("inheritance_mode")),
        "source_agent_ids": data.get("source_agent_ids", []),
        "source_node_ids": data.get("source_node_ids", []),
        "copied_recipe_id": data.get("copied_recipe_id", data.get("source_recipe_id")),
        "recombined_recipe_ids": data.get("recombined_recipe_ids", []),
        "rejected_recipe_id": data.get("rejected_recipe_id"),
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
