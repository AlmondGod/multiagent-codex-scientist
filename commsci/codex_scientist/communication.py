from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from commsci.artifacts import agent_artifact_dir, read_json, write_json, write_text
from commsci.critique import assign_roles

from .prompts import critic_prompt
from .schemas import REQUIRED_AGENT_ARTIFACTS, REQUIRED_NODE_ARTIFACTS


def run_codex_critique_round(
    *,
    output_dir: Path,
    condition: str,
    num_agents: int,
    seed: int,
    config: dict[str, Any],
    roles: dict[str, str] | None = None,
) -> None:
    roles = roles or assign_roles(num_agents, seed)
    summaries = {
        f"agent_{idx}": read_json(agent_artifact_dir(output_dir, condition, f"agent_{idx}") / "branch_summary.json")
        for idx in range(num_agents)
    }
    population_summary = build_population_summary(summaries)
    write_json(output_dir / condition / "population_summary_step_1.json", population_summary)
    for idx in range(num_agents):
        target_id = f"agent_{idx}"
        critic_id = target_id if condition == "self_critique" or num_agents == 1 else f"agent_{(idx + 1) % num_agents}"
        role = roles.get(critic_id) if condition == "peer_critique_with_roles" else None
        target_dir = agent_artifact_dir(output_dir, condition, target_id)
        visible_context = build_visible_context(summaries, target_id, critic_id, condition, config)
        prompt = critic_prompt(
            author_id=critic_id,
            target_summary=summaries[target_id],
            condition=condition,
            role=role,
            visible_context=visible_context,
        )
        override = load_critique_override(config, target_id, critic_id, condition)
        if override:
            critique = str(override.get("critique", "")).strip()
            if not critique:
                critique = local_critique(summaries[target_id], critic_id, role)
            author_id = str(override.get("author_id", critic_id))
            recommended_next_action = str(
                override.get(
                    "recommended_next_action",
                    "Use the paired step-2 action override if present; otherwise change exactly one validated TinyWorlds knob.",
                )
            )
            critique_backend = override.get("backend", "live_codex_override")
            override_path = override.get("override_path")
        else:
            critique = local_critique(summaries[target_id], critic_id, role)
            author_id = critic_id
            recommended_next_action = "Change exactly one validated TinyWorlds knob while preserving the harness."
            critique_backend = "codex_scientist_local"
            override_path = None
        write_text(target_dir / "prompts" / "critique.txt", prompt)
        write_text(target_dir / "completions" / "critique.txt", critique)
        write_text(target_dir / "critique.md", critique)
        write_json(
            target_dir / "critique.json",
            {
                "author_id": author_id,
                "target_id": target_id,
                "condition": condition,
                "role": role,
                "visible_context": visible_context,
                "population_summary_step_1": population_summary if condition != "self_critique" else None,
                "critique": critique,
                "recommended_next_action": recommended_next_action,
                "critique_backend": critique_backend,
                "override_path": override_path,
            },
        )


def codex_decision(
    critique: str,
    metrics1: dict[str, Any],
    config: dict[str, Any] | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    override = load_decision_override(config or {}, agent_id or "")
    if override:
        decision = normalize_decision_override(override)
        decision.setdefault("input_received", critique)
        decision.setdefault("decision_backend", "live_codex_override")
        decision.setdefault("override_path", override.get("override_path"))
        return decision
    failed = not metrics1.get("experiment_success")
    return {
        "input_received": critique,
        "decision_changed": True,
        "change_type": "debugged_code" if failed else "changed_hyperparameter",
        "cultural_operator": "reject" if failed else "mutate",
        "source_agent_ids": [agent_id] if agent_id else [],
        "source_node_ids": [],
        "copied_recipe_id": None,
        "recombined_recipe_ids": [],
        "rejected_recipe_id": None,
        "reason": (
            "First node failed, so the second node should use a simpler validated action."
            if failed
            else "Critique recommended a controlled one-variable follow-up under the same TinyWorlds budget."
        ),
        "revised_experiment_plan": (
            "Run a second Codex-Scientist node using the canonical TinyWorlds harness and one validated knob change."
        ),
        "decision_backend": "codex_scientist_local",
        "later_helped": None,
        "evidence": None,
    }


def codex_reviewer(note_path: Path, review_path: Path) -> dict[str, Any]:
    text = note_path.read_text(encoding="utf-8")
    unsupported = sum(text.lower().count(marker) for marker in ["unsupported", "no evidence", "unclear"])
    success_mentions = text.count('"experiment_success": true')
    score = 6 if success_mentions >= 2 else 4 if success_mentions == 1 else 3
    review = {
        "reviewer_score": score,
        "unsupported_claim_count": unsupported,
        "ablation_quality": round(score / 8, 3),
        "reviewer_backend": "codex_scientist_local",
        "reviewer_model": "supervised_codex_artifact_rubric",
        "raw_review": {
            "Summary": "Local Codex-Scientist rubric review of the two-node research note.",
            "Strengths": ["Uses the real TinyWorlds harness", "Records node actions and metrics"],
            "Weaknesses": [] if success_mentions else ["No successful TinyWorlds metric was recorded"],
            "Overall": score,
        },
        "note_path": str(note_path),
    }
    write_json(review_path, review)
    return review


def check_artifact_completeness(artifact_dir: Path) -> dict[str, Any]:
    missing_agent = [name for name in REQUIRED_AGENT_ARTIFACTS if not (artifact_dir / name).exists()]
    node_dirs = sorted((artifact_dir / "codex_scientist" / "nodes").glob("*"))
    node_missing = {
        node.name: [name for name in REQUIRED_NODE_ARTIFACTS if not (node / name).exists()]
        for node in node_dirs
        if node.is_dir()
    }
    node_missing = {key: value for key, value in node_missing.items() if value}
    return {
        "complete": not missing_agent and not node_missing,
        "missing_agent_artifacts": missing_agent,
        "missing_node_artifacts": node_missing,
        "node_count": len(node_dirs),
    }


def build_visible_context(
    summaries: dict[str, dict[str, Any]],
    target_id: str,
    critic_id: str,
    condition: str,
    config: dict[str, Any] | None = None,
) -> str:
    # Self-critique is the isolation baseline: no population information leaks in.
    if condition == "self_critique":
        return "Self critique: critic sees only its own branch summary."

    # Peer conditions expose a compact scoreboard of first-step branch outcomes.
    population_enabled = (config or {}).get("experiment", {}).get("codex_scientist_population_context", True)
    population_text = ""
    if population_enabled:
        population_text = "Population step-1 scoreboard:\n" + json.dumps(
            build_population_summary(summaries), indent=2, sort_keys=True
        )[:3500] + "\n\n"

    # The peer critic also sees its own branch, enabling comparison against the target.
    if critic_id in summaries:
        return population_text + "Peer critique: critic also knows its own first branch outcome:\n" + json.dumps(
            summaries[critic_id], indent=2, sort_keys=True
        )[:2500]

    # Fallback keeps the peer condition valid even if a critic branch is missing.
    return population_text + "Peer critique: no extra critic context was available."


def build_population_summary(summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for agent_id, summary in sorted(summaries.items()):
        metrics = summary.get("metrics_experiment_1", {})
        codex_node = summary.get("codex_node") or {}
        action = codex_node.get("action") or summary.get("action") or {}
        patch_recipe = (action.get("patch_recipe") or {}).get("id")
        inheritance = action.get("inheritance") or {}
        rows.append(
            {
                "agent_id": agent_id,
                "branch_id": summary.get("branch_id"),
                "node_id": codex_node.get("node_id"),
                "primary_score": metrics.get("primary_score"),
                "val_mse": metrics.get("val_mse"),
                "experiment_success": metrics.get("experiment_success"),
                "recipe_id": action.get("recipe_id"),
                "patch_recipe_id": patch_recipe,
                "knobs": action.get("knobs"),
                "inheritance_mode": inheritance.get("mode"),
            }
        )
    return sorted(rows, key=lambda row: row["primary_score"] if isinstance(row["primary_score"], (int, float)) else -1, reverse=True)


def local_critique(summary: dict[str, Any], critic_id: str, role: str | None) -> str:
    metrics = summary.get("metrics_experiment_1", {})
    success = bool(metrics.get("experiment_success"))
    score = metrics.get("primary_score")
    role_line = f"Role lens: {role}.\n" if role else ""
    concern = (
        "The first branch failed or lacks a parsed metric, so the second node should reduce risk."
        if not success
        else f"The first branch succeeded with primary_score={score}; the next node needs a controlled comparison."
    )
    return (
        f"{role_line}"
        f"Critic: {critic_id}\n"
        f"1. Strongest concern: {concern}\n"
        "2. Missing control or ablation: compare exactly one TinyWorlds knob against the current branch.\n"
        "3. Metric risk: preserve score, val_mse, runtime, and failure status from metrics.json.\n"
        "4. Implementation risk: do not replace the canonical TinyWorlds harness or invent synthetic data.\n"
        "5. Suggested next experiment: choose one validated knob change and rerun the node in an isolated workspace.\n"
        "6. Falsification: the change fails, lacks metrics, or worsens primary_score under the same budget.\n"
        "7. Recommendation: change the second expansion but keep the action surface bounded.\n"
    )


def load_critique_override(
    config: dict[str, Any],
    target_id: str,
    critic_id: str,
    condition: str,
) -> dict[str, Any] | None:
    override_dir = config.get("experiment", {}).get("codex_scientist_critique_overrides_dir")
    if not override_dir:
        return None
    base = Path(override_dir).expanduser()
    stems = [
        f"{target_id}_critique",
        f"{critic_id}_to_{target_id}",
        f"{condition}_{target_id}_critique",
    ]
    for stem in stems:
        for suffix in (".json", ".md", ".txt"):
            path = base / f"{stem}{suffix}"
            if path.exists():
                return read_override_file(path, default_key="critique")
    return None


def load_decision_override(config: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    override_dir = config.get("experiment", {}).get("codex_scientist_decision_overrides_dir")
    if not override_dir or not agent_id:
        return None
    base = Path(override_dir).expanduser()
    stems = [f"{agent_id}_decision", f"{agent_id}_step_2_decision"]
    for stem in stems:
        for suffix in (".json", ".md", ".txt"):
            path = base / f"{stem}{suffix}"
            if path.exists():
                return read_override_file(path, default_key="revised_experiment_plan")
    return None


def read_override_file(path: Path, default_key: str) -> dict[str, Any]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected object JSON in {path}")
    else:
        data = {default_key: path.read_text(encoding="utf-8")}
    data["override_path"] = str(path)
    return data


def normalize_decision_override(raw: dict[str, Any]) -> dict[str, Any]:
    plan = raw.get("revised_experiment_plan", raw.get("plan", raw.get("decision", "")))
    if not isinstance(plan, str):
        plan = json.dumps(plan, indent=2, sort_keys=True)
    return {
        "decision_changed": bool(raw.get("decision_changed", True)),
        "change_type": str(raw.get("change_type", "live_codex_selected_action")),
        "cultural_operator": str(raw.get("cultural_operator", raw.get("inheritance_mode", "mutate"))),
        "source_agent_ids": raw.get("source_agent_ids", []),
        "source_node_ids": raw.get("source_node_ids", []),
        "copied_recipe_id": raw.get("copied_recipe_id", raw.get("source_recipe_id")),
        "recombined_recipe_ids": raw.get("recombined_recipe_ids", []),
        "rejected_recipe_id": raw.get("rejected_recipe_id"),
        "reason": str(raw.get("reason", "Live Codex critic selected the second branch action.")),
        "revised_experiment_plan": plan,
        "recommended_next_action": raw.get("recommended_next_action"),
        "critic_author_id": raw.get("critic_author_id"),
        "target_id": raw.get("target_id"),
        "later_helped": raw.get("later_helped"),
        "evidence": raw.get("evidence"),
        "override_path": raw.get("override_path"),
    }
