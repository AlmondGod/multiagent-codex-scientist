#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from commsci.artifacts import ensure_dir, write_json, write_text
from commsci.codex_scientist.runner import run_codex_scientist_branch_expansion
from commsci.config import DEFAULT_CONFIG, deep_merge, write_config


PATCH_RECIPES = [
    "baseline_no_patch",
    "dynamics_first_schedule",
    "action_grad_dynamics",
    "smooth_l1_dynamics_pixel",
    "sharpen_change_weights",
    "full_budget_action_supervision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live Codex cultural-evolution tree for paper generation.")
    parser.add_argument("--output_dir", default="runs/cultural_paper_tree_seed0")
    parser.add_argument("--tinyworlds_dir", required=True)
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--time_budget_seconds", type=int, default=120)
    parser.add_argument("--timeout_seconds", type=int, default=240)
    parser.add_argument("--parallel_workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wait_for_actions", action="store_true")
    parser.add_argument("--poll_seconds", type=float, default=5)
    parser.add_argument("--action_timeout_seconds", type=float, default=3600)
    parser.add_argument("--init_default_actions", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    action_root = ensure_dir(output_dir / "live_actions")
    config = build_config(args, action_root)
    write_config(output_dir / "config.yaml", config)
    write_json(
        output_dir / "run_spec.json",
        {
            "seed": args.seed,
            "generations": args.generations,
            "num_agents": args.num_agents,
            "time_budget_seconds": args.time_budget_seconds,
            "mode": "live_high_variance_cultural_tree",
        },
    )
    if args.init_default_actions:
        write_initial_actions(action_root, args.num_agents)
    for generation in range(args.generations):
        action_dir = ensure_dir(action_root / f"generation_{generation:02d}")
        write_generation_prompts(output_dir, action_dir, generation, args.num_agents)
        if args.wait_for_actions:
            wait_for_actions(action_dir, generation, args.num_agents, args.action_timeout_seconds, args.poll_seconds)
        missing = missing_actions(action_dir, generation, args.num_agents)
        if missing:
            raise FileNotFoundError(
                "Missing live action files:\n"
                + "\n".join(str(path) for path in missing)
                + f"\nPrompts are in {output_dir / 'live_prompts' / f'generation_{generation:02d}'}"
            )
        config["experiment"]["codex_scientist_action_overrides_dir"] = str(action_dir)
        run_generation(output_dir, generation, args.num_agents, config, args.seed, args.parallel_workers)
        summary = build_population_summary(output_dir, generation, args.num_agents)
        write_json(output_dir / f"population_summary_generation_{generation:02d}.json", summary)
        write_json(output_dir / "latest_population_summary.json", summary)
        write_lineage_graph(output_dir, generation)
        print(f"Completed generation {generation}; best_score={summary[0].get('primary_score') if summary else None}", flush=True)
    write_paper(output_dir, args.generations)
    print(f"Wrote paper draft to {output_dir / 'paper.md'}")
    return 0


def build_config(args: argparse.Namespace, action_root: Path) -> dict[str, Any]:
    config = deep_merge(
        DEFAULT_CONFIG,
        {
            "paths": {
                "tinyworlds_dir": args.tinyworlds_dir,
                "output_dir": str(args.output_dir),
            },
            "compute": {
                "num_agents": args.num_agents,
                "max_total_experiments": args.num_agents * args.generations,
            },
            "model": {"mock_model": True, "seed": args.seed},
            "base_system": {
                "write_full_paper": True,
                "reviewer_enabled": False,
            },
            "experiment": {
                "runner": "codex_scientist",
                "task_spec": (
                    "Explore high-variance TinyWorlds architecture/training ideas through cultural evolution. "
                    "Agents should produce paper-worthy ideas with explicit lineage: copy, mutate, recombine, reject, or invent."
                ),
                "ai_scientist_data_dir": args.tinyworlds_dir,
                "codex_scientist_time_budget_seconds": args.time_budget_seconds,
                "codex_scientist_timeout_seconds": args.timeout_seconds,
                "codex_scientist_parallel_workers": args.parallel_workers,
                "codex_scientist_population_context": True,
                "codex_scientist_action_overrides_dir": str(action_root / "generation_00"),
                "codex_scientist_patch_recipes": PATCH_RECIPES,
                "tinyworlds_baseline_knobs": {"TW_DATASET": "minigrid", "TW_DEPTH": "1"},
                "allowed_files": ["train.py", "models.py"],
                "primary_metric": "primary_score",
            },
        },
    )
    return config


def write_initial_actions(action_root: Path, num_agents: int) -> None:
    action_dir = ensure_dir(action_root / "generation_00")
    defaults = [
        {
            "recipe_id": "g0_agent0_auxiliary_action_contrast",
            "patch_recipe_id": "action_grad_dynamics",
            "inheritance_mode": "invent",
            "knobs": {"use_env_actions": 1, "action_supervision_weight": 0.35},
            "rationale": "Invent an action-sensitive dynamics branch that lets action representations receive dynamics gradients.",
        },
        {
            "recipe_id": "g0_agent1_robust_decoder_loss",
            "patch_recipe_id": "smooth_l1_dynamics_pixel",
            "inheritance_mode": "invent",
            "knobs": {"dynamics_pixel_loss_weight": 6.0, "motion_prior_weight": 1.5},
            "rationale": "Invent a robust reconstruction objective for dynamics decoding.",
        },
        {
            "recipe_id": "g0_agent2_short_budget_curriculum",
            "patch_recipe_id": "dynamics_first_schedule",
            "inheritance_mode": "invent",
            "knobs": {"depth": 2, "dynamics_change_weight": 2.0},
            "rationale": "Invent a short-budget curriculum that reaches dynamics training much earlier.",
        },
    ]
    for agent_index in range(num_agents):
        payload = defaults[agent_index % len(defaults)]
        (action_dir / f"agent_{agent_index}_step_1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_generation_prompts(output_dir: Path, action_dir: Path, generation: int, num_agents: int) -> None:
    prompt_dir = ensure_dir(output_dir / "live_prompts" / f"generation_{generation:02d}")
    previous_summary = output_dir / f"population_summary_generation_{generation - 1:02d}.json"
    visible = previous_summary.read_text(encoding="utf-8") if previous_summary.exists() else "[]"
    for agent_index in range(num_agents):
        path = prompt_dir / f"agent_{agent_index}_action_prompt.md"
        write_text(
            path,
            f"""# Live Codex Cultural Tree Action

Target action file:

```text
{action_dir / f"agent_{agent_index}_step_{generation + 1}.json"}
```

You are agent_{agent_index} in generation {generation} of a 15-generation
Codex-Scientist cultural-evolution run.

Goal: produce a paper-worthy, high-variance TinyWorlds idea. You may use:

- curated `patch_recipe_id`: {", ".join(PATCH_RECIPES)}
- exact `file_edits` against only `train.py` and `models.py`
- validated knobs: depth, use_env_actions, dynamics_change_weight,
  dynamics_pixel_loss_weight, motion_loss_weight, motion_change_weight,
  motion_prior_weight, action_supervision_weight

Every non-initial generation must choose `inheritance_mode`: copy, mutate,
recombine, reject, or invent. Include source_agent_ids and source_node_ids when
using prior ideas.

Population summary from the previous generation:

```json
{visible[:7000]}
```

Write only valid JSON. The JSON may include:

```json
{{
  "recipe_id": "short_unique_name",
  "inheritance_mode": "mutate",
  "source_agent_ids": ["agent_1"],
  "source_node_ids": ["cultural_evolution_agent_1_node_{generation}"],
  "patch_recipe_id": "baseline_no_patch",
  "knobs": {{}},
  "file_edits": [
    {{
      "path": "models.py",
      "description": "what this edit does",
      "find": "exact existing text",
      "replace": "replacement text"
    }}
  ],
  "rationale": "why this is a high-variance idea worth trying"
}}
```
""",
        )


def wait_for_actions(action_dir: Path, generation: int, num_agents: int, timeout: float, poll: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        missing = missing_actions(action_dir, generation, num_agents)
        if not missing:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Timed out waiting for actions:\n" + "\n".join(str(path) for path in missing))
        print(f"Waiting for {len(missing)} live action files in {action_dir}", flush=True)
        time.sleep(max(1.0, poll))


def missing_actions(action_dir: Path, generation: int, num_agents: int) -> list[Path]:
    return [
        action_dir / f"agent_{agent_index}_step_{generation + 1}.json"
        for agent_index in range(num_agents)
        if not (action_dir / f"agent_{agent_index}_step_{generation + 1}.json").exists()
    ]


def run_generation(
    output_dir: Path,
    generation: int,
    num_agents: int,
    config: dict[str, Any],
    seed: int,
    parallel_workers: int,
) -> None:
    condition_dir = ensure_dir(output_dir / "cultural_evolution")
    with ThreadPoolExecutor(max_workers=max(1, min(num_agents, parallel_workers))) as pool:
        futures = {}
        for agent_index in range(num_agents):
            agent_id = f"agent_{agent_index}"
            artifact_dir = ensure_dir(condition_dir / agent_id / "artifacts")
            if node_metrics_path(output_dir, agent_index, generation).exists():
                print(f"Skipping {agent_id} generation {generation}; metrics already exist.", flush=True)
                continue
            previous_action = read_action(output_dir, agent_index, generation) if generation > 0 else None
            future = pool.submit(
                run_codex_scientist_branch_expansion,
                artifact_dir,
                agent_id,
                f"cultural_evolution_{agent_id}",
                generation + 1,
                config,
                seed + generation,
                generation_context(output_dir, generation),
                None,
                previous_action,
            )
            futures[future] = agent_id
        for future in as_completed(futures):
            agent_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{agent_id} generation {generation} failed") from exc


def node_metrics_path(output_dir: Path, agent_index: int, generation: int) -> Path:
    return (
        output_dir
        / "cultural_evolution"
        / f"agent_{agent_index}"
        / "artifacts"
        / "codex_scientist"
        / "nodes"
        / f"cultural_evolution_agent_{agent_index}_node_{generation + 1}"
        / "metrics.json"
    )


def generation_context(output_dir: Path, generation: int) -> str | None:
    if generation == 0:
        return None
    summary = output_dir / f"population_summary_generation_{generation - 1:02d}.json"
    return summary.read_text(encoding="utf-8") if summary.exists() else None


def read_action(output_dir: Path, agent_index: int, generation: int) -> dict[str, Any]:
    path = (
        output_dir
        / "cultural_evolution"
        / f"agent_{agent_index}"
        / "artifacts"
        / "codex_scientist"
        / "nodes"
        / f"cultural_evolution_agent_{agent_index}_node_{generation}"
        / "action.json"
    )
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_population_summary(output_dir: Path, generation: int, num_agents: int) -> list[dict[str, Any]]:
    rows = []
    for agent_index in range(num_agents):
        node_id = f"cultural_evolution_agent_{agent_index}_node_{generation + 1}"
        node_dir = (
            output_dir
            / "cultural_evolution"
            / f"agent_{agent_index}"
            / "artifacts"
            / "codex_scientist"
            / "nodes"
            / node_id
        )
        action = json.loads((node_dir / "action.json").read_text(encoding="utf-8"))
        metrics = json.loads((node_dir / "metrics.json").read_text(encoding="utf-8"))
        inheritance = action.get("inheritance") or {}
        rows.append(
            {
                "generation": generation,
                "agent_id": f"agent_{agent_index}",
                "node_id": node_id,
                "parent_id": f"cultural_evolution_agent_{agent_index}_node_{generation}" if generation > 0 else None,
                "primary_score": metrics.get("primary_score"),
                "val_mse": metrics.get("val_mse"),
                "experiment_success": metrics.get("experiment_success"),
                "recipe_id": action.get("recipe_id"),
                "patch_recipe_id": (action.get("patch_recipe") or {}).get("id"),
                "file_edit_count": len(action.get("file_edits") or []),
                "knobs": action.get("knobs"),
                "inheritance_mode": inheritance.get("mode"),
                "source_agent_ids": inheritance.get("source_agent_ids", []),
                "source_node_ids": inheritance.get("source_node_ids", []),
                "rationale": action.get("subagent_rationale") or inheritance.get("rationale"),
            }
        )
    return sorted(rows, key=lambda row: row["primary_score"] if isinstance(row["primary_score"], (int, float)) else -1, reverse=True)


def write_lineage_graph(output_dir: Path, latest_generation: int) -> None:
    lines = ["flowchart TD"]
    for generation in range(latest_generation + 1):
        path = output_dir / f"population_summary_generation_{generation:02d}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            node = row["node_id"]
            label = (
                f"{row['agent_id']} g{generation}<br/>{row.get('inheritance_mode')} "
                f"{row.get('patch_recipe_id')}<br/>score {row.get('primary_score')}"
            )
            lines.append(f'    {safe_id(node)}["{label}"]')
            for source in row.get("source_node_ids") or []:
                lines.append(f"    {safe_id(source)} --> {safe_id(node)}")
            if row.get("parent_id"):
                lines.append(f"    {safe_id(row['parent_id'])} -.-> {safe_id(node)}")
    write_text(output_dir / "lineage_graph.md", "```mermaid\n" + "\n".join(lines) + "\n```\n")


def safe_id(value: str) -> str:
    return value.replace("-", "_").replace(".", "_")


def write_paper(output_dir: Path, generations: int) -> None:
    summaries = []
    for generation in range(generations):
        path = output_dir / f"population_summary_generation_{generation:02d}.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    flat = [row for summary in summaries for row in summary]
    best = max(flat, key=lambda row: row.get("primary_score") or -1) if flat else {}
    best_by_generation = [
        max(summary, key=lambda row: row.get("primary_score") or -1)
        for summary in summaries
        if summary
    ]
    operator_counts: dict[str, int] = {}
    for row in flat:
        operator_counts[row.get("inheritance_mode") or "unknown"] = operator_counts.get(row.get("inheritance_mode") or "unknown", 0) + 1
    scores = [row["primary_score"] for row in flat if isinstance(row.get("primary_score"), (int, float))]
    table = "\n".join(
        f"| {row['generation']} | {row['agent_id']} | {row['primary_score']} | {row['patch_recipe_id']} | {row['inheritance_mode']} | {row['recipe_id']} |"
        for row in best_by_generation
    )
    write_text(
        output_dir / "paper.md",
        f"""# Cultural Evolution in a Live Codex-Scientist TinyWorlds Population

## Abstract

We ran a 3-agent, {generations}-generation exploratory Codex-Scientist tree in
which agents proposed high-variance TinyWorlds training and architecture changes,
executed real experiments, and used explicit cultural operators over prior
branches. The best observed node reached primary_score={best.get('primary_score')}.

## Methods

Each generation contained one node per agent. Nodes ran the canonical TinyWorlds
`train.py` harness in isolated workspaces under a fixed time budget. Actions could
use curated patch recipes, validated `TW_*` knobs, and exact source edits against
allowlisted files (`train.py`, `models.py`). Each non-initial action recorded an
inheritance operator: copy, mutate, recombine, reject, or invent.

## Results

- total nodes: {len(flat)}
- mean score: {round(mean(scores), 6) if scores else None}
- best score: {best.get('primary_score')}
- best node: {best.get('node_id')}
- operator counts: {json.dumps(operator_counts, sort_keys=True)}

| Generation | Agent | Best score | Patch recipe | Operator | Recipe |
| ---: | --- | ---: | --- | --- | --- |
{table}

## Lineage Analysis

The lineage graph is stored in `lineage_graph.md`. Successful ideas should be
read as branches that spread by copy or recombination and continue improving
across later generations.

## Limitations

This is a single exploratory run, not a controlled ablation. Its purpose is to
produce a paper-style case study and discover high-variance ideas worth testing
under stricter replicated conditions.

## Conclusion

This run provides an auditable case study of cultural evolution over live
Codex-Scientist research nodes. The next step is to compare the discovered
lineage against self-only and random-transfer controls.
""",
    )


if __name__ == "__main__":
    raise SystemExit(main())
