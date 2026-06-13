from __future__ import annotations

import json
from typing import Any

from commsci.tinyworlds_knobs import allowlist_from_config, initial_knobs_for_agent, validate_knobs


PATCH_RECIPES: dict[str, dict[str, Any]] = {
    "baseline_no_patch": {
        "id": "baseline_no_patch",
        "kind": "patch_recipe",
        "description": "No source patch; rely on validated TinyWorlds environment knobs.",
        "files": [],
    },
}


def initial_action(agent_index: int, config: dict[str, Any]) -> dict[str, Any]:
    allowlist = allowlist_from_config(config)
    env, applied, dropped = initial_knobs_for_agent(agent_index, allowlist)
    return {
        "kind": "knob_recipe",
        "recipe_id": f"initial_agent_{agent_index}",
        "knobs": applied,
        "env": env,
        "dropped": dropped,
        "patch_recipe": PATCH_RECIPES["baseline_no_patch"],
    }


def normalize_action(raw: dict[str, Any], config: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    allowlist = allowlist_from_config(config)
    raw_knobs = raw.get("knobs", raw.get("env", raw)) if isinstance(raw, dict) else {}
    env, applied, dropped = validate_knobs(raw_knobs, allowlist)
    return {
        "kind": "knob_recipe",
        "recipe_id": str(raw.get("recipe_id", recipe_id)) if isinstance(raw, dict) else recipe_id,
        "knobs": applied,
        "env": env,
        "dropped": dropped,
        "subagent_rationale": raw.get("rationale", "") if isinstance(raw, dict) else "",
        "patch_recipe": PATCH_RECIPES["baseline_no_patch"],
    }


def revise_action(
    *,
    agent_index: int,
    config: dict[str, Any],
    current_action: dict[str, Any],
    critique: str,
    revised_plan: str,
) -> dict[str, Any]:
    allowlist = allowlist_from_config(config)
    current_knobs = current_action.get("knobs", {}) if isinstance(current_action, dict) else {}
    raw = _heuristic_knob_revision(agent_index, critique, revised_plan, current_knobs)
    env, applied, dropped = validate_knobs(raw, allowlist)
    merged_knobs = {**current_knobs, **applied}
    merged_env, merged_applied, merge_dropped = validate_knobs(merged_knobs, allowlist)
    return {
        "kind": "knob_recipe",
        "recipe_id": f"revision_agent_{agent_index}",
        "knobs": merged_applied,
        "env": merged_env,
        "dropped": [*dropped, *merge_dropped],
        "critique_basis": critique[:1200],
        "revised_plan": revised_plan,
        "patch_recipe": PATCH_RECIPES["baseline_no_patch"],
    }


def action_diff(action: dict[str, Any]) -> str:
    env = action.get("env") or {}
    if not env:
        return "Codex-Scientist action: baseline TinyWorlds configuration.\n"
    return "Codex-Scientist TinyWorlds knob overrides:\n" + "\n".join(
        f"{key}={value}" for key, value in sorted(env.items())
    ) + "\n"


def action_summary(action: dict[str, Any]) -> str:
    knobs = action.get("knobs") or {}
    if not knobs:
        return "Codex-Scientist baseline TinyWorlds run; no source patch."
    return "Codex-Scientist knobs: " + json.dumps(knobs, sort_keys=True)


def _heuristic_knob_revision(
    agent_index: int,
    critique: str,
    revised_plan: str,
    current_knobs: dict[str, Any],
) -> dict[str, Any]:
    text = f"{critique}\n{revised_plan}".lower()
    if "action" in text or "supervision" in text:
        return {"use_env_actions": 1, "action_supervision_weight": 0.5}
    if "motion" in text:
        return {"motion_loss_weight": 1.0 + (agent_index % 3), "motion_change_weight": 1.0}
    if "pixel" in text or "reconstruction" in text:
        return {"dynamics_pixel_loss_weight": 1.0}
    if "dynamics" in text:
        return {"dynamics_change_weight": 1.0}
    depth = int(float(current_knobs.get("depth", 1))) if str(current_knobs.get("depth", "1")).replace(".", "", 1).isdigit() else 1
    return {"depth": min(6, max(1, depth + 1))}
