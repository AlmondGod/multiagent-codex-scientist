from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from commsci.tinyworlds_knobs import allowlist_from_config, initial_knobs_for_agent, validate_knobs


PATCH_RECIPES: dict[str, dict[str, Any]] = {
    "baseline_no_patch": {
        "id": "baseline_no_patch",
        "kind": "patch_recipe",
        "description": "No source patch; rely on validated TinyWorlds environment knobs.",
        "files": [],
    },
    "dynamics_first_schedule": {
        "id": "dynamics_first_schedule",
        "kind": "patch_recipe",
        "description": "Move TinyWorlds training into actions/dynamics phases earlier so short runs spend less time only on tokenizer warmup.",
        "files": ["train.py"],
    },
    "action_grad_dynamics": {
        "id": "action_grad_dynamics",
        "kind": "patch_recipe",
        "description": "Allow the learned action tokenizer to receive gradient through dynamics training instead of detaching actions.",
        "files": ["models.py"],
    },
    "smooth_l1_dynamics_pixel": {
        "id": "smooth_l1_dynamics_pixel",
        "kind": "patch_recipe",
        "description": "Use smooth-L1 instead of MSE for dynamics pixel reconstruction to reduce outlier sensitivity.",
        "files": ["models.py"],
    },
    "sharpen_change_weights": {
        "id": "sharpen_change_weights",
        "kind": "patch_recipe",
        "description": "Square-normalize patch-change weights so dynamics loss focuses harder on moving/changing patches.",
        "files": ["models.py"],
    },
    "full_budget_action_supervision": {
        "id": "full_budget_action_supervision",
        "kind": "patch_recipe",
        "description": "Keep action supervision active for the whole training budget when action supervision weight is nonzero.",
        "files": ["train.py"],
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
    patch_recipe = normalize_patch_recipe(raw if isinstance(raw, dict) else {}, config)
    return {
        "kind": "code_recipe" if patch_recipe["id"] != "baseline_no_patch" else "knob_recipe",
        "recipe_id": str(raw.get("recipe_id", recipe_id)) if isinstance(raw, dict) else recipe_id,
        "knobs": applied,
        "env": env,
        "dropped": dropped,
        "subagent_rationale": raw.get("rationale", "") if isinstance(raw, dict) else "",
        "patch_recipe": patch_recipe,
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
    patch_diff = action.get("code_diff") or ""
    parts: list[str] = []
    if env:
        parts.append("Codex-Scientist TinyWorlds knob overrides:\n" + "\n".join(
            f"{key}={value}" for key, value in sorted(env.items())
        ) + "\n")
    else:
        parts.append("Codex-Scientist action: baseline TinyWorlds knob configuration.\n")
    recipe = (action.get("patch_recipe") or {}).get("id", "baseline_no_patch")
    parts.append(f"Patch recipe: {recipe}\n")
    if patch_diff:
        parts.append("\n" + patch_diff)
    return "".join(parts)


def action_summary(action: dict[str, Any]) -> str:
    knobs = action.get("knobs") or {}
    recipe = (action.get("patch_recipe") or {}).get("id", "baseline_no_patch")
    bits = []
    if knobs:
        bits.append("knobs: " + json.dumps(knobs, sort_keys=True))
    if recipe != "baseline_no_patch":
        bits.append(f"patch_recipe: {recipe}")
    if not bits:
        return "Codex-Scientist baseline TinyWorlds run; no source patch."
    return "Codex-Scientist " + "; ".join(bits)


def normalize_patch_recipe(raw: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    requested = (
        raw.get("patch_recipe_id")
        or raw.get("patch_recipe")
        or raw.get("code_recipe")
        or "baseline_no_patch"
    )
    if isinstance(requested, dict):
        requested = requested.get("id", "baseline_no_patch")
    recipe_id = str(requested)
    allowed = config.get("experiment", {}).get("codex_scientist_patch_recipes")
    if allowed and recipe_id not in {str(item) for item in allowed}:
        return {
            **PATCH_RECIPES["baseline_no_patch"],
            "dropped_patch_recipe": f"{recipe_id}: not enabled in codex_scientist_patch_recipes",
        }
    recipe = PATCH_RECIPES.get(recipe_id)
    if recipe is None:
        recipe = {
            **PATCH_RECIPES["baseline_no_patch"],
            "dropped_patch_recipe": f"{recipe_id}: not an allowed patch recipe",
        }
    return dict(recipe)


def apply_patch_recipe(workspace: Path, action: dict[str, Any]) -> dict[str, Any]:
    recipe = action.get("patch_recipe") or PATCH_RECIPES["baseline_no_patch"]
    recipe_id = recipe.get("id", "baseline_no_patch")
    if recipe_id == "baseline_no_patch":
        return {"patch_applied": False, "patch_recipe_id": recipe_id, "code_diff": ""}
    before: dict[Path, str] = {}
    for filename in recipe.get("files", []):
        path = workspace / filename
        before[path] = path.read_text(encoding="utf-8")
    if recipe_id == "dynamics_first_schedule":
        replace_once(
            workspace / "train.py",
            '    if progress < 0.25:\n        return "tokenizer"\n    if progress < 0.45:\n        return "actions"\n    return "dynamics"\n',
            '    if progress < 0.10:\n        return "tokenizer"\n    if progress < 0.20:\n        return "actions"\n    return "dynamics"\n',
        )
    elif recipe_id == "action_grad_dynamics":
        replace_once(
            workspace / "models.py",
            "        actions = self.action_conditioning(frames, env_actions=env_actions, detach=True)\n",
            "        actions = self.action_conditioning(frames, env_actions=env_actions, detach=False)\n",
        )
        replace_once(
            workspace / "models.py",
            "            actions.detach(),\n",
            "            actions,\n",
        )
    elif recipe_id == "smooth_l1_dynamics_pixel":
        replace_once(
            workspace / "models.py",
            "        return F.mse_loss(pred[:, 0], target)\n",
            "        return F.smooth_l1_loss(pred[:, 0], target)\n",
        )
    elif recipe_id == "sharpen_change_weights":
        replace_once(
            workspace / "models.py",
            "        normalized = patch_change / patch_change.mean(dim=1, keepdim=True).clamp_min(1e-6)\n",
            "        normalized = patch_change / patch_change.mean(dim=1, keepdim=True).clamp_min(1e-6)\n        normalized = normalized.square()\n",
        )
    elif recipe_id == "full_budget_action_supervision":
        replace_once(
            workspace / "train.py",
            "        sup_weight = ACTION_SUPERVISION_WEIGHT if total_training_time < ACTION_SUPERVISION_SECONDS else 0.0\n",
            "        sup_weight = ACTION_SUPERVISION_WEIGHT if ACTION_SUPERVISION_WEIGHT > 0 else 0.0\n",
        )
    else:
        raise ValueError(f"Unsupported patch recipe {recipe_id!r}")
    diffs = []
    changed_files = []
    for path, old_text in before.items():
        new_text = path.read_text(encoding="utf-8")
        if new_text == old_text:
            continue
        changed_files.append(path.name)
        diffs.extend(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
            )
        )
    return {
        "patch_applied": bool(changed_files),
        "patch_recipe_id": recipe_id,
        "changed_files": changed_files,
        "code_diff": "".join(diffs),
    }


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"Patch target not found in {path.name}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


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
