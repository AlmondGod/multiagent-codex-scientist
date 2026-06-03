from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .artifacts import read_json, write_json, write_text
from .model_client import OpenAICompatibleClient, approx_tokens


ROLES = [
    ("critic", "find flaws, confounds, invalid claims, and missing controls"),
    ("ablator", "suggest the smallest decisive ablation or control experiment"),
    ("optimizer", "suggest the most likely improvement under the fixed compute budget"),
    ("explorer", "suggest an unusual but still feasible alternative direction"),
]


def assign_roles(num_agents: int, seed: int) -> dict[str, str]:
    offset = seed % len(ROLES)
    return {f"agent_{i}": ROLES[(offset + i) % len(ROLES)][0] for i in range(num_agents)}


def peer_source_for(agent_index: int, num_agents: int, seed: int) -> str:
    return f"agent_{(agent_index + 1 + seed % max(1, num_agents - 1)) % num_agents}"


def build_critique_prompt(
    target_summary: dict[str, Any],
    critic_summary: dict[str, Any],
    condition: str,
    role: str | None,
    role_prior: str | None,
    max_prompt_tokens: int,
) -> tuple[str, bool]:
    target_json = json.dumps(target_summary, indent=2, sort_keys=True)
    critic_json = json.dumps(critic_summary, indent=2, sort_keys=True)
    role_text = f"\nRole prior: {role} - {role_prior}.\n" if role and role_prior else ""
    prompt = f"""You are critiquing a TinyWorlds research branch under matched budget constraints.
Use the same critique standard regardless of condition. Only identity/context differs.
Condition: {condition}.{role_text}

Target branch summary:
{target_json}

Critic local context:
{critic_json}

Return concise Markdown plus a JSON-compatible interpretation if useful. Address exactly:
1. Strongest concern about the hypothesis or interpretation.
2. Missing control, baseline, or ablation.
3. Metric or evaluation risk.
4. Implementation/debug risk.
5. Suggested next experiment under the fixed budget.
6. What result would falsify the current hypothesis?
7. Whether the proposed next experiment is worth running or should be changed.
"""
    if approx_tokens(prompt) <= max_prompt_tokens:
        return prompt, False
    compressed_target = deterministic_compress(target_summary)
    compressed_critic = deterministic_compress(critic_summary)
    prompt = f"""You are critiquing a TinyWorlds research branch under matched budget constraints.
Use the same critique standard regardless of condition. Only identity/context differs.
Condition: {condition}.{role_text}

Target branch summary, deterministically compressed:
{json.dumps(compressed_target, indent=2, sort_keys=True)}

Critic local context, deterministically compressed:
{json.dumps(compressed_critic, indent=2, sort_keys=True)}

Return concise Markdown. Address exactly:
1. Strongest concern about the hypothesis or interpretation.
2. Missing control, baseline, or ablation.
3. Metric or evaluation risk.
4. Implementation/debug risk.
5. Suggested next experiment under the fixed budget.
6. What result would falsify the current hypothesis?
7. Whether the proposed next experiment is worth running or should be changed.
"""
    return prompt, True


def deterministic_compress(summary: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "run_id",
        "condition",
        "agent_id",
        "branch_id",
        "role",
        "hypothesis",
        "experiment_plan_1",
        "code_diff_summary",
        "baseline_metrics",
        "metrics_experiment_1",
        "failure_notes",
        "current_interpretation",
        "proposed_next_experiment",
    ]
    compressed = {key: _shorten(summary.get(key)) for key in keep}
    raw_hash = hashlib.sha256(json.dumps(summary, sort_keys=True).encode("utf-8")).hexdigest()
    compressed["original_sha256"] = raw_hash
    return compressed


def _shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 900:
        return value[:900] + "...[truncated]"
    return value


def run_critique_round(
    output_dir: Path,
    condition: str,
    num_agents: int,
    seed: int,
    max_prompt_tokens: int,
    client: OpenAICompatibleClient,
    roles: dict[str, str],
) -> None:
    role_lookup = dict(ROLES)
    for i in range(num_agents):
        target_id = f"agent_{i}"
        target_dir = output_dir / condition / target_id / "artifacts"
        target_summary = read_json(target_dir / "branch_summary.json")
        if condition == "self_critique":
            critic_id = target_id
        else:
            critic_id = peer_source_for(i, num_agents, seed)
        critic_dir = output_dir / condition / critic_id / "artifacts"
        critic_summary = read_json(critic_dir / "branch_summary.json")
        role = roles.get(critic_id) if condition == "peer_critique_with_roles" else None
        role_prior = role_lookup.get(role) if role else None
        prompt, truncated = build_critique_prompt(
            target_summary,
            critic_summary,
            condition,
            role,
            role_prior,
            max_prompt_tokens,
        )
        response = client.complete(prompt, condition, target_dir, "critique")
        critique_json = {
            "condition": condition,
            "target_agent_id": target_id,
            "critic_agent_id": critic_id,
            "role": role,
            "role_prior": role_prior,
            "max_prompt_tokens": max_prompt_tokens,
            "max_completion_tokens": client.max_completion_tokens,
            "temperature": client.temperature,
            "prompt_truncated": truncated,
            "critique": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }
        write_text(target_dir / "critique.md", response.text)
        write_json(target_dir / "critique.json", critique_json)
