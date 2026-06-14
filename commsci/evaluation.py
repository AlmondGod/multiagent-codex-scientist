from __future__ import annotations

import csv
import json
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from .artifacts import read_json, write_json, write_text


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    rows = []
    for condition_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name != "global"):
        condition = condition_dir.name
        agent_rows = []
        summaries = []
        for artifact_dir in sorted(condition_dir.glob("agent_*/artifacts")):
            row = summarize_agent(condition, artifact_dir)
            rows.append(row)
            agent_rows.append(row)
            summaries.append(read_json(artifact_dir / "branch_summary.json"))
        duplicate_rate = duplicate_experiment_rate(summaries)
        for row in agent_rows:
            row["duplicate_experiment_rate"] = duplicate_rate
    condition_summaries = summarize_conditions(rows)
    write_outputs(run_dir, rows, condition_summaries)
    return {"rows": rows, "conditions": condition_summaries}


def summarize_agent(condition: str, artifact_dir: Path) -> dict[str, Any]:
    branch = read_json(artifact_dir / "branch_summary.json")
    metrics1 = read_json(artifact_dir / "metrics_experiment_1.json")
    metrics2 = read_json(artifact_dir / "metrics_experiment_2.json")
    decision = read_json(artifact_dir / "decision_change.json")
    review = read_json(artifact_dir / "review.json")
    step1_action = read_codex_action(artifact_dir, 1)
    step2_action = read_codex_action(artifact_dir, 2)
    step2_inheritance = step2_action.get("inheritance", {}) if isinstance(step2_action, dict) else {}
    step2_patch_recipe = patch_recipe_id(step2_action)
    primary1 = _num(metrics1.get("primary_score"))
    primary2 = _num(metrics2.get("primary_score"))
    improvement = None if primary1 is None or primary2 is None else primary2 - primary1
    later_helped = decision.get("later_helped")
    return {
        "condition": condition,
        "agent_id": branch["agent_id"],
        "decision_changed": bool(decision.get("decision_changed")),
        "later_helped": later_helped,
        "communication_value": 1 if decision.get("decision_changed") and later_helped is True else 0,
        "metric_improvement": improvement,
        "final_metric_score": primary2,
        "step1_patch_recipe_id": patch_recipe_id(step1_action),
        "step2_patch_recipe_id": step2_patch_recipe,
        "cultural_operator": step2_inheritance.get("mode") or decision.get("cultural_operator"),
        "source_agent_ids": json.dumps(step2_inheritance.get("source_agent_ids", decision.get("source_agent_ids", []))),
        "source_node_ids": json.dumps(step2_inheritance.get("source_node_ids", decision.get("source_node_ids", []))),
        "copied_recipe_id": step2_inheritance.get("copied_recipe_id") or decision.get("copied_recipe_id"),
        "recombined_recipe_ids": json.dumps(step2_inheritance.get("recombined_recipe_ids", decision.get("recombined_recipe_ids", []))),
        "rejected_recipe_id": step2_inheritance.get("rejected_recipe_id") or decision.get("rejected_recipe_id"),
        "cross_agent_transfer": cross_agent_transfer(branch["agent_id"], step2_inheritance, decision),
        "unsupported_claim_count": review.get("unsupported_claim_count"),
        "ablation_quality": review.get("ablation_quality"),
        "failure_avoidance": 1 if not metrics1.get("experiment_success") and metrics2.get("experiment_success") else 0,
        "experiment_success_rate": mean([1 if metrics1.get("experiment_success") else 0, 1 if metrics2.get("experiment_success") else 0]),
        "reviewer_score": review.get("reviewer_score"),
        "runtime": _sum_nums(metrics1.get("runtime_seconds"), metrics2.get("runtime_seconds")),
        "prompt_tokens": token_sum(artifact_dir, "prompt_tokens"),
        "completion_tokens": token_sum(artifact_dir, "completion_tokens"),
        "duplicate_experiment_rate": None,
    }


def summarize_conditions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        out.append(
            {
                "condition": condition,
                "agents": len(subset),
                "useful_decision_changes": sum(row["communication_value"] for row in subset),
                "fraction_decision_changed": avg_bool(row["decision_changed"] for row in subset),
                "fraction_later_helped": avg_bool(row["later_helped"] is True for row in subset),
                "communication_value": avg(row["communication_value"] for row in subset),
                "metric_improvement": avg(row["metric_improvement"] for row in subset),
                "final_metric_score": avg(row["final_metric_score"] for row in subset),
                "copy_rate": operator_rate(subset, "copy"),
                "mutate_rate": operator_rate(subset, "mutate"),
                "recombine_rate": operator_rate(subset, "recombine"),
                "reject_rate": operator_rate(subset, "reject"),
                "invent_rate": operator_rate(subset, "invent"),
                "cross_agent_transfer_rate": avg_bool(row["cross_agent_transfer"] for row in subset),
                "unsupported_claim_count": avg(row["unsupported_claim_count"] for row in subset),
                "duplicate_experiment_rate": avg(row["duplicate_experiment_rate"] for row in subset),
                "ablation_quality": avg(row["ablation_quality"] for row in subset),
                "failure_avoidance": avg(row["failure_avoidance"] for row in subset),
                "experiment_success_rate": avg(row["experiment_success_rate"] for row in subset),
                "reviewer_score": avg(row["reviewer_score"] for row in subset),
                "runtime": avg(row["runtime"] for row in subset),
                "prompt_tokens": sum_none(row["prompt_tokens"] for row in subset),
                "completion_tokens": sum_none(row["completion_tokens"] for row in subset),
            }
        )
    return out


def write_outputs(run_dir: Path, rows: list[dict[str, Any]], condition_summaries: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(run_dir / "results.json", {"agents": rows, "conditions": condition_summaries})
    lines = ["# Results Summary", ""]
    for summary in condition_summaries:
        lines.append(f"## {summary['condition']}")
        lines.append(f"- agents: {summary['agents']}")
        lines.append(f"- fraction_decision_changed: {summary['fraction_decision_changed']}")
        lines.append(f"- fraction_later_helped: {summary['fraction_later_helped']}")
        lines.append(f"- communication_value: {summary['communication_value']}")
        lines.append(f"- final_metric_score: {summary['final_metric_score']}")
        lines.append(f"- copy_rate: {summary['copy_rate']}")
        lines.append(f"- mutate_rate: {summary['mutate_rate']}")
        lines.append(f"- recombine_rate: {summary['recombine_rate']}")
        lines.append(f"- cross_agent_transfer_rate: {summary['cross_agent_transfer_rate']}")
        lines.append(f"- reviewer_score: {summary['reviewer_score']}")
        lines.append("")
    write_text(run_dir / "summary.md", "\n".join(lines))


def duplicate_experiment_rate(summaries: list[dict[str, Any]]) -> float:
    if len(summaries) < 2:
        return 0.0
    texts = [
        " ".join(
            str(summary.get(key, ""))
            for key in ("hypothesis", "experiment_plan_1", "code_diff_summary", "proposed_next_experiment")
        ).lower()
        for summary in summaries
    ]
    pairs = 0
    duplicates = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            pairs += 1
            if SequenceMatcher(None, texts[i], texts[j]).ratio() >= 0.82:
                duplicates += 1
    return duplicates / pairs if pairs else 0.0


def read_codex_action(artifact_dir: Path, step: int) -> dict[str, Any]:
    node_root = artifact_dir / "codex_scientist" / "nodes"
    matches = sorted(node_root.glob(f"*_node_{step}/action.json"))
    if not matches:
        return {}
    return read_json(matches[-1])


def patch_recipe_id(action: dict[str, Any]) -> str | None:
    if not isinstance(action, dict):
        return None
    recipe = action.get("patch_recipe") or {}
    return recipe.get("id") if isinstance(recipe, dict) else None


def cross_agent_transfer(agent_id: str, inheritance: dict[str, Any], decision: dict[str, Any]) -> bool:
    sources = inheritance.get("source_agent_ids") if isinstance(inheritance, dict) else None
    if sources is None:
        sources = decision.get("source_agent_ids", [])
    if isinstance(sources, str):
        sources = [sources]
    return any(str(source) != agent_id for source in (sources or []))


def operator_rate(rows: list[dict[str, Any]], operator: str) -> float:
    return avg_bool(row.get("cultural_operator") == operator for row in rows)


def token_sum(artifact_dir: Path, key: str) -> int | None:
    values = []
    for path in (artifact_dir / "model_calls").glob("*.json"):
        value = read_json(path).get(key)
        if isinstance(value, int):
            values.append(value)
    return sum(values) if values else None


def avg(values: Any) -> float | None:
    clean = [_num(value) for value in values]
    clean = [value for value in clean if value is not None]
    return round(mean(clean), 6) if clean else None


def avg_bool(values: Any) -> float:
    clean = list(values)
    return round(mean(1 if value else 0 for value in clean), 6) if clean else 0.0


def sum_none(values: Any) -> int | None:
    clean = [value for value in values if isinstance(value, int)]
    return sum(clean) if clean else None


def _num(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sum_nums(*values: Any) -> float | None:
    clean = [_num(value) for value in values]
    clean = [value for value in clean if value is not None]
    return round(sum(clean), 6) if clean else None
