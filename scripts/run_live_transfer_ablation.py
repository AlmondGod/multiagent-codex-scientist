#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any


CONDITIONS = [("self", "self_critique"), ("peer", "peer_critique")]

STEP1_ACTIONS: dict[str, dict[str, Any]] = {
    "agent_0": {
        "recipe_id": "agent_0_invent_action_grad",
        "patch_recipe_id": "action_grad_dynamics",
        "inheritance_mode": "invent",
        "knobs": {"use_env_actions": 1, "action_supervision_weight": 0.35},
        "rationale": "Initial invented action-gradient dynamics branch.",
    },
    "agent_1": {
        "recipe_id": "agent_1_invent_smooth_l1",
        "patch_recipe_id": "smooth_l1_dynamics_pixel",
        "inheritance_mode": "invent",
        "knobs": {"dynamics_pixel_loss_weight": 6, "motion_prior_weight": 1.5},
        "rationale": "Initial invented robust Smooth-L1 dynamics-pixel branch.",
    },
    "agent_2": {
        "recipe_id": "agent_2_invent_fast_schedule",
        "patch_recipe_id": "dynamics_first_schedule",
        "inheritance_mode": "invent",
        "knobs": {"depth": 2, "dynamics_change_weight": 2},
        "rationale": "Initial invented early-dynamics schedule branch.",
    },
}

SELF_MUTATIONS: dict[str, dict[str, Any]] = {
    "agent_0": {
        "recipe_id": "agent_0_mutate_full_budget_action_supervision_step_2",
        "patch_recipe_id": "full_budget_action_supervision",
        "inheritance_mode": "mutate",
        "source_agent_ids": ["agent_0"],
        "knobs": {"depth": 1, "use_env_actions": 1, "action_supervision_weight": 0.8},
        "rationale": "Self-only mutation: preserve action conditioning and test persistent action supervision.",
    },
    "agent_1": {
        "recipe_id": "agent_1_mutate_smooth_l1_weights_step_2",
        "patch_recipe_id": "smooth_l1_dynamics_pixel",
        "inheritance_mode": "mutate",
        "source_agent_ids": ["agent_1"],
        "knobs": {"dynamics_pixel_loss_weight": 8.0, "motion_prior_weight": 1.0},
        "rationale": "Self-only mutation: keep Smooth-L1 dynamics and adjust robust reconstruction weights.",
    },
    "agent_2": {
        "recipe_id": "agent_2_mutate_action_grad_step_2",
        "patch_recipe_id": "action_grad_dynamics",
        "inheritance_mode": "mutate",
        "source_agent_ids": ["agent_2"],
        "knobs": {
            "depth": 2,
            "use_env_actions": 1,
            "dynamics_change_weight": 2.0,
            "action_supervision_weight": 0.5,
        },
        "rationale": "Self-only mutation: keep larger early-dynamics branch but add explicit action-gradient conditioning.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Codex-Scientist live transfer ablation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8, 9, 10])
    parser.add_argument("--tinyworlds_dir", required=True)
    parser.add_argument("--output_root", default="runs")
    parser.add_argument("--tmp_root", default="/tmp")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max_runtime_minutes_per_experiment", type=int, default=3)
    parser.add_argument("--max_tokens_per_critique", type=int, default=1000)
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--timeout_seconds", type=float, default=3600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    tinyworlds_dir = Path(args.tinyworlds_dir).expanduser().resolve()
    for seed in args.seeds:
        for short_condition, condition in CONDITIONS:
            run_condition(repo, tinyworlds_dir, args, seed, short_condition, condition)
    return 0


def run_condition(
    repo: Path,
    tinyworlds_dir: Path,
    args: argparse.Namespace,
    seed: int,
    short_condition: str,
    condition: str,
) -> None:
    tmp_root = Path(args.tmp_root).expanduser() / f"multiagent_codex_scientist_seed{seed}"
    output_dir = repo / args.output_root / f"live_cultural_lineage_seed{seed}_{condition}"
    config_path = tmp_root / f"config_{condition}.yaml"
    for name in ("actions", "critiques", "decisions"):
        (tmp_root / short_condition / name).mkdir(parents=True, exist_ok=True)
    config_path.write_text(render_config(seed, tmp_root, short_condition, tinyworlds_dir, output_dir), encoding="utf-8")
    for agent_id, action in STEP1_ACTIONS.items():
        write_json(tmp_root / short_condition / "actions" / f"{agent_id}_step_1.json", action)

    log_path = tmp_root / f"{condition}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    cmd = [
        args.python,
        "run_ablation.py",
        "--runner",
        "codex_scientist",
        "--condition",
        condition,
        "--num_agents",
        "3",
        "--config",
        str(config_path),
        "--tinyworlds_dir",
        str(tinyworlds_dir),
        "--ai_scientist_data_dir",
        str(tinyworlds_dir),
        "--output_dir",
        str(output_dir.relative_to(repo)),
        "--max_runtime_minutes_per_experiment",
        str(args.max_runtime_minutes_per_experiment),
        "--max_tokens_per_critique",
        str(args.max_tokens_per_critique),
        "--write_full_paper",
        "false",
        "--seed",
        str(seed),
    ]
    print(f"starting seed={seed} condition={condition} log={log_path}", flush=True)
    process = subprocess.Popen(cmd, cwd=repo, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    start = time.time()
    wrote_overrides = False
    try:
        while process.poll() is None:
            rows = read_step1_rows(output_dir, condition)
            if rows is not None and not wrote_overrides:
                write_live_overrides(tmp_root, short_condition, condition, seed, rows)
                wrote_overrides = True
            if time.time() - start > args.timeout_seconds:
                process.terminate()
                raise TimeoutError(f"Timed out seed={seed} condition={condition}")
            time.sleep(args.poll_seconds)
    finally:
        log_handle.close()
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"Failed seed={seed} condition={condition} rc={process.returncode}\n{tail}")
    print(f"completed seed={seed} condition={condition}", flush=True)


def render_config(seed: int, tmp_root: Path, short_condition: str, tinyworlds_dir: Path, output_dir: Path) -> str:
    return dedent(
        f"""
        base_system:
          repo_url: https://github.com/SakanaAI/AI-Scientist-v2
          use_existing_tree_search: true
          use_existing_reviewer: true
          write_full_paper: false
          reviewer_enabled: true
          reviewer_backend: codex_scientist_local
          reviewer_model: gpt-4o-mini
          reviewer_model_url: null
          reviewer_num_reflections: 1
          reviewer_num_fs_examples: 0
          reviewer_temperature: 0.2
        model:
          backend: openai_compatible
          model_url: http://localhost:1234/v1
          default_model: qwen3-coder-30b-a3b-instruct
          fallback_models:
          - qwen2.5-coder-14b
          - qwen2.5-coder-7b
          temperature: 0.2
          seed: {seed}
          max_prompt_tokens: 6000
          max_completion_tokens: 1000
          mock_model: true
        compute:
          reduced_mode: true
          num_agents: 3
          max_depth_per_agent: 2
          max_experiments_per_agent: 2
          max_total_experiments: 6
          max_training_steps: 1000
          max_runtime_minutes_per_experiment: 3
          max_tokens_per_critique: 1000
        experiment:
          substrate: TinyWorlds
          runner: codex_scientist
          conditions:
          - self_critique
          - peer_critique
          task_spec: 'Live-authored cultural transfer run: after step 1, choose copy/mutate/recombine/reject/invent from observed TinyWorlds branch summaries.'
          primary_metric: primary_score
          ai_scientist_data_dir: {tinyworlds_dir}
          ai_scientist_timeout_seconds: 600
          ai_scientist_time_budget_seconds: 180
          ai_scientist_copy_data: false
          codex_scientist_timeout_seconds: 240
          codex_scientist_time_budget_seconds: 180
          codex_scientist_shared_memory: false
          codex_scientist_backend: supervised_thread_tools
          codex_scientist_parallel_workers: 3
          codex_scientist_action_overrides_dir: {tmp_root / short_condition / 'actions'}
          codex_scientist_critique_overrides_dir: {tmp_root / short_condition / 'critiques'}
          codex_scientist_decision_overrides_dir: {tmp_root / short_condition / 'decisions'}
          codex_scientist_wait_for_live_overrides: true
          codex_scientist_population_context: true
          codex_scientist_live_override_timeout_seconds: 3600
          codex_scientist_live_override_poll_seconds: 5
          codex_scientist_patch_recipes:
          - baseline_no_patch
          - dynamics_first_schedule
          - action_grad_dynamics
          - smooth_l1_dynamics_pixel
          - sharpen_change_weights
          - full_budget_action_supervision
          tinyworlds_baseline_knobs:
            TW_DATASET: minigrid
            TW_DEPTH: '1'
        paths:
          tinyworlds_dir: {tinyworlds_dir}
          ai_scientist_v2_dir: null
          output_dir: {output_dir}
        """
    ).lstrip()


def read_step1_rows(output_dir: Path, condition: str) -> list[dict[str, Any]] | None:
    rows = []
    for index in range(3):
        agent_id = f"agent_{index}"
        path = output_dir / condition / agent_id / "artifacts" / "branch_summary.json"
        if not path.exists():
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics_experiment_1") or {}
        codex_node = summary.get("codex_node") or {}
        action = codex_node.get("action") or summary.get("action") or {}
        patch_recipe = action.get("patch_recipe") or {}
        rows.append(
            {
                "agent_id": agent_id,
                "node_id": codex_node.get("node_id") or f"{condition}_{agent_id}_node_1",
                "primary_score": metrics.get("primary_score"),
                "patch_recipe_id": patch_recipe.get("id") if isinstance(patch_recipe, dict) else None,
                "recipe_id": action.get("recipe_id"),
                "knobs": action.get("knobs") or {},
            }
        )
    return sorted(rows, key=lambda row: row["primary_score"] if isinstance(row["primary_score"], (int, float)) else -1, reverse=True)


def write_live_overrides(
    tmp_root: Path,
    short_condition: str,
    condition: str,
    seed: int,
    rows: list[dict[str, Any]],
) -> None:
    best = rows[0]
    population = "; ".join(
        f"{row['agent_id']} score={row.get('primary_score')} recipe={row.get('patch_recipe_id')}" for row in rows
    )
    for index in range(3):
        agent_id = f"agent_{index}"
        row = next(item for item in rows if item["agent_id"] == agent_id)
        if condition == "self_critique":
            action = dict(SELF_MUTATIONS[agent_id])
            action["source_node_ids"] = [row["node_id"]]
            critique = {
                "author_id": agent_id,
                "target_id": agent_id,
                "condition": condition,
                "assessment": f"Self critique for {agent_id}; mutate only its own branch.",
                "recommended_inheritance_mode": "mutate",
                "recommended_patch_recipe_id": action["patch_recipe_id"],
            }
        else:
            action = peer_action(agent_id, best)
            critique = {
                "author_id": f"agent_{(index + 1) % 3}",
                "target_id": agent_id,
                "condition": condition,
                "population_summary": population,
                "assessment": f"Observed population after step 1: {population}. Recommended operator: {action['inheritance_mode']}.",
                "recommended_inheritance_mode": action["inheritance_mode"],
                "recommended_patch_recipe_id": action["patch_recipe_id"],
            }
        decision = {
            "decision_changed": True,
            "change_type": "live_codex_selected_transfer_action",
            "cultural_operator": action["inheritance_mode"],
            "source_agent_ids": action.get("source_agent_ids", []),
            "source_node_ids": action.get("source_node_ids", []),
            "copied_recipe_id": action.get("copied_recipe_id"),
            "reason": critique["assessment"],
            "revised_experiment_plan": action["rationale"],
            "later_helped": None,
        }
        write_json(tmp_root / short_condition / "critiques" / f"{agent_id}_critique.json", critique)
        write_json(tmp_root / short_condition / "decisions" / f"{agent_id}_decision.json", decision)
        write_json(tmp_root / short_condition / "actions" / f"{agent_id}_step_2.json", action)
    print(
        f"wrote overrides seed={seed} condition={condition} best={best['agent_id']} "
        f"recipe={best.get('patch_recipe_id')} score={best.get('primary_score')}",
        flush=True,
    )


def peer_action(agent_id: str, best: dict[str, Any]) -> dict[str, Any]:
    if agent_id == best["agent_id"]:
        recipe = best.get("patch_recipe_id") or "baseline_no_patch"
        if recipe == "smooth_l1_dynamics_pixel":
            knobs = {"dynamics_pixel_loss_weight": 8.0, "motion_prior_weight": 1.0}
        elif recipe == "action_grad_dynamics":
            knobs = {"use_env_actions": 1, "action_supervision_weight": 0.8, "dynamics_change_weight": 1.0}
        elif recipe == "dynamics_first_schedule":
            knobs = {"depth": 2, "dynamics_change_weight": 1.0, "dynamics_pixel_loss_weight": 1.0}
        else:
            knobs = dict(best.get("knobs") or {})
        return {
            "recipe_id": f"{agent_id}_mutate_own_leader_step_2",
            "patch_recipe_id": recipe,
            "inheritance_mode": "mutate",
            "cultural_operator": "mutate",
            "source_agent_ids": [agent_id],
            "source_node_ids": [best["node_id"]],
            "knobs": knobs,
            "rationale": f"Live peer transfer: mutate this agent's own population-leading branch {recipe}.",
        }
    return {
        "recipe_id": f"{agent_id}_copy_{best['agent_id']}_{best.get('patch_recipe_id')}_step_2",
        "patch_recipe_id": best.get("patch_recipe_id") or "baseline_no_patch",
        "inheritance_mode": "copy",
        "cultural_operator": "copy",
        "source_agent_ids": [best["agent_id"]],
        "source_node_ids": [best["node_id"]],
        "copied_recipe_id": best.get("recipe_id"),
        "knobs": dict(best.get("knobs") or {}),
        "rationale": f"Live peer transfer: copy observed population leader {best['agent_id']}.",
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
