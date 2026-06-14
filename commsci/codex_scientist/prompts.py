from __future__ import annotations

import json
from typing import Any


def worker_task_prompt(
    *,
    node_id: str,
    parent_id: str | None,
    agent_id: str,
    branch_id: str,
    step: int,
    task_spec: str,
    action: dict[str, Any],
    critique_context: str | None,
    memory: list[str],
) -> str:
    return f"""# Codex-Scientist Worker Task

You are a live Codex node worker in a supervised tree-search science run.

## Node
- node_id: {node_id}
- parent_id: {parent_id}
- agent_id: {agent_id}
- branch_id: {branch_id}
- depth/step: {step}

## Scientific Objective
{task_spec}

## Assigned Action
```json
{json.dumps(action, indent=2, sort_keys=True)}
```

## Execution Contract
- Run the real TinyWorlds harness through the canonical runfile.
- Do not invent a synthetic replacement experiment.
- Keep the compute budget fixed.
- Save metrics from `working/metrics.json`.
- Record all changes as validated knobs or curated patch recipes.
- Curated patch recipes are allowed when assigned in the action JSON. Available
  recipe ids are: `baseline_no_patch`, `dynamics_first_schedule`,
  `action_grad_dynamics`, `smooth_l1_dynamics_pixel`,
  `sharpen_change_weights`, and `full_budget_action_supervision`.

## Visible Memory
{format_memory(memory)}

## Communication Context
{critique_context or "None. This is the first branch expansion."}
"""


def critic_prompt(
    *,
    author_id: str,
    target_summary: dict[str, Any],
    condition: str,
    role: str | None,
    visible_context: str,
) -> str:
    role_text = f"\nRole prior: {role}\n" if role else ""
    return f"""# Codex-Scientist Critique Task

author_id: {author_id}
condition: {condition}{role_text}

You may only use the branch summary and visible context below. Produce a critique
that can guide exactly one second branch expansion under the same TinyWorlds budget.
When recommending a next action, classify the cultural operator as one of:
`copy`, `mutate`, `recombine`, `reject`, or `invent`. Prefer explicit
copy/mutate/recombine recommendations when visible peer outcomes justify them.

## Target Branch Summary
```json
{json.dumps(target_summary, indent=2, sort_keys=True)}
```

## Visible Context
{visible_context}
"""


def format_memory(memory: list[str]) -> str:
    if not memory:
        return "No prior node-local memory."
    return "\n".join(f"- {item}" for item in memory)
