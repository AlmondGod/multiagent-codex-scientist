"""Tier-1 controlled modification surface for TinyWorlds branch expansion.

TinyWorlds' train.py reads its entire config from TW_* environment variables, so a
"code modification" reduces to choosing a validated set of knob values and injecting
them as env vars into the canonical runfile.py.  This keeps branch expansion fully
deterministic and attributable: experiment 2 differs from experiment 1 only by the
knobs the critique drove, never by free-form generated code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Friendly knob name -> spec. Only knobs that run on the default minigrid+tinyworlds
# path are enabled by default; widen via experiment.tinyworlds_knob_allowlist in config.
# TW_TIME_BUDGET and TW_DATASET are intentionally excluded: compute budget and task
# must stay fixed across the ablation.
TINYWORLDS_KNOBS: dict[str, dict[str, Any]] = {
    "depth": {"env": "TW_DEPTH", "type": "int", "min": 1, "max": 6,
              "note": "model size; higher = bigger model, fewer steps in the fixed budget"},
    # context_length is coupled to the architecture: the default tinyworlds model is
    # built for context_length=2 and a different value triggers a tensor-shape mismatch
    # in the attention/FiLM path. Only valid alongside model_type=dit, so default_off.
    "context_length": {"env": "TW_CONTEXT_LENGTH", "type": "int", "min": 1, "max": 8, "default_off": True,
                       "note": "frames of temporal context (only safe with model_type=dit)"},
    "use_env_actions": {"env": "TW_USE_ENV_ACTIONS", "type": "bool01",
                        "note": "0 = learned latent actions, 1 = ground-truth env actions"},
    "dynamics_change_weight": {"env": "TW_DYNAMICS_CHANGE_WEIGHT", "type": "float", "min": 0.0, "max": 10.0,
                               "note": "weight on dynamics change loss"},
    "dynamics_pixel_loss_weight": {"env": "TW_DYNAMICS_PIXEL_LOSS_WEIGHT", "type": "float", "min": 0.0, "max": 10.0,
                                   "note": "weight on dynamics pixel-reconstruction loss"},
    "motion_loss_weight": {"env": "TW_MOTION_LOSS_WEIGHT", "type": "float", "min": 0.0, "max": 10.0,
                           "note": "weight on motion loss"},
    "motion_change_weight": {"env": "TW_MOTION_CHANGE_WEIGHT", "type": "float", "min": 0.0, "max": 10.0,
                             "note": "weight on motion change loss"},
    "motion_prior_weight": {"env": "TW_MOTION_PRIOR_WEIGHT", "type": "float", "min": 0.0, "max": 10.0,
                            "note": "weight on motion prior"},
    "action_supervision_weight": {"env": "TW_ACTION_SUPERVISION_WEIGHT", "type": "float", "min": 0.0, "max": 1.0,
                                  "note": "weight on optional action supervision"},
    # Off by default (only relevant to the diffusion 'dit' model path):
    "model_type": {"env": "TW_MODEL", "type": "choice", "choices": ["tinyworlds", "dit"], "default_off": True,
                   "note": "architecture; 'dit' switches to the diffusion code path"},
    "diffusion_steps": {"env": "TW_DIFFUSION_STEPS", "type": "int", "min": 4, "max": 128, "default_off": True,
                        "note": "diffusion steps (only used when model_type=dit)"},
}


def allowlist_from_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the active knob allowlist.

    Default = every knob without default_off. Override with
    experiment.tinyworlds_knob_allowlist: a list of friendly knob names.
    """
    override = (config.get("experiment") or {}).get("tinyworlds_knob_allowlist")
    if override:
        return {name: TINYWORLDS_KNOBS[name] for name in override if name in TINYWORLDS_KNOBS}
    return {name: spec for name, spec in TINYWORLDS_KNOBS.items() if not spec.get("default_off")}


def _coerce_one(name: str, spec: dict[str, Any], value: Any) -> tuple[str | None, str | None]:
    """Coerce + validate a single knob value. Returns (env_string, drop_reason)."""
    kind = spec["type"]
    try:
        if kind == "int":
            v = int(round(float(value)))
            v = max(spec["min"], min(spec["max"], v))
            return str(v), None
        if kind == "float":
            v = float(value)
            v = max(spec["min"], min(spec["max"], v))
            return repr(v), None
        if kind == "bool01":
            if isinstance(value, bool):
                v = 1 if value else 0
            elif isinstance(value, (int, float)):
                v = 1 if value else 0
            else:
                v = 1 if str(value).strip().lower() in {"1", "true", "yes", "on"} else 0
            return str(v), None
        if kind == "choice":
            v = str(value).strip().lower()
            if v not in spec["choices"]:
                return None, f"{name}={value!r} not in {spec['choices']}"
            return v, None
    except (TypeError, ValueError):
        return None, f"{name}={value!r} not coercible to {kind}"
    return None, f"{name}: unknown knob type {kind}"


def validate_knobs(
    raw: dict[str, Any], allowlist: dict[str, dict[str, Any]]
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Validate a raw knob dict against the allowlist.

    Accepts both friendly names (depth) and env names (TW_DEPTH).
    Returns (env_overrides {TW_*: str}, applied {friendly: value}, dropped [reasons]).
    """
    env_by_name = {spec["env"]: name for name, spec in allowlist.items()}
    env_overrides: dict[str, str] = {}
    applied: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in (raw or {}).items():
        name = key if key in allowlist else env_by_name.get(key)
        if name is None:
            dropped.append(f"{key}: not an allowed knob")
            continue
        spec = allowlist[name]
        env_value, reason = _coerce_one(name, spec, value)
        if env_value is None:
            dropped.append(reason or f"{key}: invalid")
            continue
        env_overrides[spec["env"]] = env_value
        applied[name] = env_value
    return env_overrides, applied, dropped


def _extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of the first JSON object from model output."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # strip code fences, then grab the first balanced-looking {...}
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(cleaned[start : i + 1])
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    return {}
    return {}


def build_selection_prompt(
    critique: str, revised_plan: str, allowlist: dict[str, dict[str, Any]]
) -> str:
    lines = []
    for name, spec in allowlist.items():
        if spec["type"] == "choice":
            rng = f"one of {spec['choices']}"
        elif spec["type"] == "bool01":
            rng = "0 or 1"
        else:
            rng = f"{spec['type']} in [{spec['min']}, {spec['max']}]"
        lines.append(f"- {name}: {rng} ({spec['note']})")
    menu = "\n".join(lines)
    return (
        "You are configuring the NEXT TinyWorlds experiment to act on the revised plan below.\n"
        "You may ONLY adjust these knobs:\n"
        f"{menu}\n\n"
        "Output ONLY a JSON object mapping knob names to their NEW values. Include only the\n"
        "knobs you want to change relative to the previous run. Use {} for no change.\n"
        "Do not add commentary, code, or any key not listed above.\n\n"
        f"Revised plan:\n{revised_plan}\n\n"
        f"Critique that motivated it:\n{critique[:1500]}\n"
    )


def select_knobs(
    client: Any,
    critique: str,
    revised_plan: str,
    allowlist: dict[str, dict[str, Any]],
    condition: str,
    artifact_dir: Path,
) -> tuple[dict[str, str], dict[str, Any], list[str]]:
    """Ask the communication model to pick validated TinyWorlds knob overrides.

    Returns (env_overrides {TW_*: str}, applied {friendly: value}, dropped [reasons]).
    On any model/parse failure, returns empty overrides (baseline re-run) rather than raising.
    """
    if not allowlist:
        return {}, {}, []
    prompt = build_selection_prompt(critique, revised_plan, allowlist)
    try:
        response = client.complete(prompt, condition, artifact_dir, "knob_selection")
        raw = _extract_json_object(response.text)
    except Exception as exc:  # noqa: BLE001 - degrade to baseline, never block the run
        return {}, {}, [f"knob selection failed: {exc}"]
    return validate_knobs(raw, allowlist)


def summarize_knobs(applied: dict[str, Any]) -> str:
    if not applied:
        return "No TinyWorlds knob changes (baseline configuration re-run)."
    return "Applied TinyWorlds knob overrides:\n" + "\n".join(
        f"  {name} = {value}" for name, value in sorted(applied.items())
    )
