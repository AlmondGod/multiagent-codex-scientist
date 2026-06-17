#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


CONTROLLED_SEEDS = [0, 1, 2, 4, 5]
LIVE_SEEDS = [6, 7, 8, 9, 10]

CONTROLLED_CONDITIONS = [
    ("self critique", "self_critique"),
    ("peer critique", "peer_critique"),
    ("peer + roles", "peer_critique_with_roles"),
]
LIVE_CONDITIONS = [
    ("self critique", "self_critique"),
    ("peer critique", "peer_critique"),
]

METRICS = [
    ("Mean Final Score", "final_metric_score"),
    ("Mean Improvement", "metric_improvement"),
    ("Decision Changed", "fraction_decision_changed"),
    ("Cross-Agent Transfer", "cross_agent_transfer_rate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final paper result tables from run artifacts.")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--docs_output", default="docs/results.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    controlled = summarize_table(
        runs_dir=runs_dir,
        run_prefix="cultural_lineage_seed",
        seeds=CONTROLLED_SEEDS,
        conditions=CONTROLLED_CONDITIONS,
    )
    live = summarize_table(
        runs_dir=runs_dir,
        run_prefix="live_cultural_lineage_seed",
        seeds=LIVE_SEEDS,
        conditions=LIVE_CONDITIONS,
    )

    write_csv(results_dir / "controlled_critique_5seed.csv", controlled)
    write_csv(results_dir / "live_transfer_5seed.csv", live)
    write_docs(Path(args.docs_output), controlled, live)
    return 0


def summarize_table(
    *,
    runs_dir: Path,
    run_prefix: str,
    seeds: list[int],
    conditions: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for label, condition in conditions:
        seed_rows = []
        for seed in seeds:
            path = runs_dir / f"{run_prefix}{seed}_{condition}" / "results.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            seed_rows.append(data["conditions"][0])
        row: dict[str, Any] = {"condition": label, "seeds": " ".join(str(seed) for seed in seeds)}
        for _, key in METRICS:
            values = [float(seed_row.get(key) or 0.0) for seed_row in seed_rows]
            row[key] = round(mean(values), 6)
            row[f"{key}_stddev"] = round(stdev(values), 6) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "seeds",
        "final_metric_score",
        "final_metric_score_stddev",
        "metric_improvement",
        "metric_improvement_stddev",
        "fraction_decision_changed",
        "fraction_decision_changed_stddev",
        "cross_agent_transfer_rate",
        "cross_agent_transfer_rate_stddev",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_docs(path: Path, controlled: list[dict[str, Any]], live: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# Results",
            "",
            "All standard deviations are sample standard deviations across seeds.",
            "Decision Changed is the fraction of agents whose second action changed after the communication checkpoint.",
            "",
            "## Controlled Critique Ablation",
            "",
            "Seeds: 0, 1, 2, 4, 5.",
            "",
            markdown_table(controlled),
            "",
            controlled_delta_text(controlled),
            "",
            "## Live Transfer Ablation",
            "",
            "Seeds: 6, 7, 8, 9, 10.",
            "",
            markdown_table(live),
            "",
            live_delta_text(live),
            "",
            "## Compact CSVs",
            "",
            "- `results/controlled_critique_5seed.csv`",
            "- `results/live_transfer_5seed.csv`",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Condition | Mean Final Score | Mean Improvement | Decision Changed | Cross-Agent Transfer |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {final} +/- {final_sd} | {improvement} +/- {improvement_sd} | {changed} +/- {changed_sd} | {transfer} +/- {transfer_sd} |".format(
                condition=row["condition"],
                final=format_float(row["final_metric_score"]),
                final_sd=format_float(row["final_metric_score_stddev"]),
                improvement=format_float(row["metric_improvement"]),
                improvement_sd=format_float(row["metric_improvement_stddev"]),
                changed=format_float(row["fraction_decision_changed"]),
                changed_sd=format_float(row["fraction_decision_changed_stddev"]),
                transfer=format_float(row["cross_agent_transfer_rate"]),
                transfer_sd=format_float(row["cross_agent_transfer_rate_stddev"]),
            )
        )
    return "\n".join(lines)


def controlled_delta_text(rows: list[dict[str, Any]]) -> str:
    self_row = by_condition(rows, "self critique")
    peer_row = by_condition(rows, "peer critique")
    final_delta = peer_row["final_metric_score"] - self_row["final_metric_score"]
    improvement_delta = peer_row["metric_improvement"] - self_row["metric_improvement"]
    return (
        "Peer critique improved final score by "
        f"+{final_delta:.6f} absolute on average, about "
        f"{relative_percent(peer_row['final_metric_score'], self_row['final_metric_score']):.2f}% relative. "
        "Mean improvement increased by "
        f"+{improvement_delta:.6f}, about "
        f"{relative_percent(peer_row['metric_improvement'], self_row['metric_improvement']):.1f}% relative."
    )


def live_delta_text(rows: list[dict[str, Any]]) -> str:
    self_row = by_condition(rows, "self critique")
    peer_row = by_condition(rows, "peer critique")
    final_delta = peer_row["final_metric_score"] - self_row["final_metric_score"]
    improvement_delta = peer_row["metric_improvement"] - self_row["metric_improvement"]
    return (
        "Live peer transfer improved final score by "
        f"+{final_delta:.6f} absolute on average, about "
        f"{relative_percent(peer_row['final_metric_score'], self_row['final_metric_score']):.2f}% relative. "
        "Mean improvement increased by "
        f"+{improvement_delta:.6f}, about "
        f"{relative_percent(peer_row['metric_improvement'], self_row['metric_improvement']):.1f}% relative."
    )


def by_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    for row in rows:
        if row["condition"] == condition:
            return row
    raise KeyError(condition)


def relative_percent(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def format_float(value: float) -> str:
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
