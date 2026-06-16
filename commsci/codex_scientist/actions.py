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

INHERITANCE_MODES = {"invent", "copy", "mutate", "recombine", "reject"}


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
        "inheritance": {
            "mode": "invent",
            "source_agent_ids": [],
            "source_node_ids": [],
            "rationale": "Initial branch invention before any communication checkpoint.",
        },
    }


def normalize_action(raw: dict[str, Any], config: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    allowlist = allowlist_from_config(config)
    raw_knobs = raw.get("knobs", raw.get("env", raw)) if isinstance(raw, dict) else {}
    env, applied, dropped = validate_knobs(raw_knobs, allowlist)
    patch_recipe = normalize_patch_recipe(raw if isinstance(raw, dict) else {}, config)
    inheritance = normalize_inheritance(raw if isinstance(raw, dict) else {}, default_mode="invent")
    file_edits, dropped_file_edits = normalize_file_edits(raw if isinstance(raw, dict) else {}, config)
    return {
        "kind": "code_recipe" if patch_recipe["id"] != "baseline_no_patch" or file_edits else "knob_recipe",
        "recipe_id": str(raw.get("recipe_id", recipe_id)) if isinstance(raw, dict) else recipe_id,
        "knobs": applied,
        "env": env,
        "dropped": [*dropped, *dropped_file_edits],
        "subagent_rationale": raw.get("rationale", "") if isinstance(raw, dict) else "",
        "patch_recipe": patch_recipe,
        "file_edits": file_edits,
        "inheritance": inheritance,
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
        "inheritance": {
            "mode": "mutate",
            "source_agent_ids": [f"agent_{agent_index}"],
            "source_node_ids": [],
            "rationale": "Local fallback revision mutates this worker's previous branch after critique.",
        },
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
    edits = action.get("file_edits") or []
    if edits:
        bits.append(f"file_edits: {len(edits)}")
    inheritance = action.get("inheritance") or {}
    mode = inheritance.get("mode")
    if mode:
        bits.append(f"inheritance: {mode}")
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


def normalize_inheritance(raw: dict[str, Any], default_mode: str) -> dict[str, Any]:
    nested = raw.get("inheritance") if isinstance(raw.get("inheritance"), dict) else {}
    mode = str(
        raw.get(
            "inheritance_mode",
            raw.get("cultural_operator", raw.get("operator", nested.get("mode", default_mode))),
        )
    ).strip().lower()
    if mode not in INHERITANCE_MODES:
        mode = default_mode
    source_agent_ids = raw.get(
        "source_agent_ids",
        raw.get("source_agents", raw.get("source_agent_id", nested.get("source_agent_ids", []))),
    )
    source_node_ids = raw.get(
        "source_node_ids",
        raw.get("source_nodes", raw.get("source_node_id", nested.get("source_node_ids", []))),
    )
    if isinstance(source_agent_ids, str):
        source_agent_ids = [source_agent_ids]
    if isinstance(source_node_ids, str):
        source_node_ids = [source_node_ids]
    copied_recipe_id = raw.get("copied_recipe_id", raw.get("source_recipe_id", nested.get("copied_recipe_id")))
    rejected_recipe_id = raw.get("rejected_recipe_id", nested.get("rejected_recipe_id"))
    recombined_recipe_ids = raw.get("recombined_recipe_ids", nested.get("recombined_recipe_ids", []))
    if isinstance(recombined_recipe_ids, str):
        recombined_recipe_ids = [recombined_recipe_ids]
    return {
        "mode": mode,
        "source_agent_ids": [str(item) for item in source_agent_ids],
        "source_node_ids": [str(item) for item in source_node_ids],
        "copied_recipe_id": str(copied_recipe_id) if copied_recipe_id is not None else None,
        "recombined_recipe_ids": [str(item) for item in recombined_recipe_ids],
        "rejected_recipe_id": str(rejected_recipe_id) if rejected_recipe_id is not None else None,
        "rationale": str(raw.get("inheritance_rationale", raw.get("rationale", nested.get("rationale", "")))),
    }


def normalize_file_edits(raw: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    raw_edits = raw.get("file_edits") or raw.get("edits") or []
    if not isinstance(raw_edits, list):
        return [], ["file_edits: expected list"]
    allowed = config.get("experiment", {}).get("allowed_files") or ["train.py", "models.py"]
    allowed_set = {str(item) for item in allowed}
    edits: list[dict[str, str]] = []
    dropped: list[str] = []
    for index, edit in enumerate(raw_edits):
        if not isinstance(edit, dict):
            dropped.append(f"file_edits[{index}]: expected object")
            continue
        filename = str(edit.get("path", edit.get("file", ""))).strip()
        find = edit.get("find", edit.get("old"))
        replace = edit.get("replace", edit.get("new"))
        if filename not in allowed_set:
            dropped.append(f"file_edits[{index}].path={filename!r}: not allowed")
            continue
        if not isinstance(find, str) or not isinstance(replace, str) or not find:
            dropped.append(f"file_edits[{index}]: find/replace must be non-empty strings")
            continue
        edits.append(
            {
                "path": filename,
                "find": find,
                "replace": replace,
                "description": str(edit.get("description", "")),
            }
        )
    return edits, dropped


def apply_patch_recipe(workspace: Path, action: dict[str, Any]) -> dict[str, Any]:
    recipe = action.get("patch_recipe") or PATCH_RECIPES["baseline_no_patch"]
    recipe_id = recipe.get("id", "baseline_no_patch")
    file_edits = action.get("file_edits") or []
    if recipe_id == "baseline_no_patch" and not file_edits:
        return {"patch_applied": False, "patch_recipe_id": recipe_id, "file_edits_applied": 0, "code_diff": ""}
    before: dict[Path, str] = {}
    for filename in recipe.get("files", []):
        path = workspace / filename
        before[path] = path.read_text(encoding="utf-8")
    for edit in file_edits:
        path = workspace / edit["path"]
        before.setdefault(path, path.read_text(encoding="utf-8"))
    if recipe_id == "baseline_no_patch":
        pass
    elif recipe_id == "dynamics_first_schedule":
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
    applied_file_edits = 0
    for edit in file_edits:
        replace_once(workspace / edit["path"], edit["find"], edit["replace"])
        applied_file_edits += 1
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
        "file_edits_applied": applied_file_edits,
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
