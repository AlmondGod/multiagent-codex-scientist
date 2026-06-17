from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

from commsci.artifacts import ensure_dir, write_json, write_text
from commsci.codex_scientist.paper import load_population_rows, score_key, write_domain_latex_from_run, write_domain_paper_from_run

from .schemas import RichCodexNode, StageReport, V2_STAGES


STAGE_GOALS = {
    "initial_implementation": [
        "Establish a working TinyWorlds baseline and verify the harness.",
        "Prefer simple executable interventions over novelty.",
    ],
    "baseline_tuning": [
        "Tune or compare compact baseline variants without changing the core architecture.",
        "Identify whether simple schedules or losses already explain the gains.",
    ],
    "creative_research": [
        "Explore higher-variance world-model ideas with bounded patches.",
        "Preserve cultural lineage through copy, mutate, recombine, reject, or invent.",
    ],
    "ablation_studies": [
        "Compare the best idea against direct negative and positive controls.",
        "Avoid dumping the full search trace into the paper.",
    ],
    "plot_aggregation": [
        "Generate a small number of final figures that clarify the paper claim.",
    ],
    "paper_writeup": [
        "Write a workshop-style domain paper about the discovered scientific idea.",
    ],
    "review": [
        "Review the paper for soundness, clarity, contribution, and missing controls.",
    ],
}


def run_codex_scientistv2(args: Any) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    if not getattr(args, "skip_experiments", False):
        run_population_tree(args, output_dir)
    return write_v2_outputs(output_dir, doctrine_doc=getattr(args, "doctrine_doc", None))


def run_population_tree(args: Any, output_dir: Path) -> None:
    if getattr(args, "live_actions_src", None):
        src = Path(args.live_actions_src).expanduser().resolve()
        dst = output_dir / "live_actions"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    cmd = [
        sys.executable,
        "scripts/run_cultural_paper_tree.py",
        "--output_dir",
        str(output_dir),
        "--tinyworlds_dir",
        str(Path(args.tinyworlds_dir).expanduser()),
        "--generations",
        str(args.generations),
        "--num_agents",
        str(args.num_agents),
        "--time_budget_seconds",
        str(args.time_budget_seconds),
        "--timeout_seconds",
        str(args.timeout_seconds),
        "--parallel_workers",
        str(args.parallel_workers),
        "--seed",
        str(args.seed),
        "--doctrine_doc",
        str(args.doctrine_doc),
    ]
    if getattr(args, "init_default_actions", False):
        cmd.append("--init_default_actions")
    subprocess.run(cmd, check=True)


def write_v2_outputs(output_dir: Path, doctrine_doc: str | None = None) -> Path:
    rows = load_population_rows(output_dir)
    if not rows:
        raise FileNotFoundError(f"No population summaries found in {output_dir}")
    v2_dir = ensure_dir(output_dir / "codex_scientistv2")
    nodes = build_rich_nodes(output_dir, rows)
    write_jsonl(v2_dir / "rich_nodes.jsonl", [node.to_dict() for node in nodes])
    stage_reports = build_stage_reports(nodes)
    ensure_dir(v2_dir / "stage_reports")
    for report in stage_reports:
        write_json(v2_dir / "stage_reports" / f"{report.name}.json", report.to_dict())
    write_json(v2_dir / "run_manifest.json", build_manifest(output_dir, rows, doctrine_doc))
    write_json(v2_dir / "ablation_report.json", build_ablation_report(rows))
    write_literature_seed(v2_dir)
    write_figures(output_dir, rows)
    write_tree_visualizations(output_dir, rows)
    write_codex_tasks(output_dir, rows, doctrine_doc)
    write_domain_paper_from_run(output_dir, output_dir / "paper.md")
    write_latex_workshop_stub(output_dir)
    write_review_scaffold(output_dir)
    return v2_dir


def build_rich_nodes(output_dir: Path, rows: list[dict[str, Any]]) -> list[RichCodexNode]:
    nodes = []
    max_generation = max(int(row["generation"]) for row in rows)
    for row in sorted(rows, key=lambda item: (int(item["generation"]), item["agent_id"])):
        node_root = (
            output_dir
            / "cultural_evolution"
            / row["agent_id"]
            / "artifacts"
            / "codex_scientist"
            / "nodes"
            / row["node_id"]
        )
        metrics_path = node_root / "metrics.json"
        action_path = node_root / "action.json"
        logs_path = node_root / "logs.txt"
        diff_path = node_root / "code_diff.patch"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        metrics = {
            **metrics,
            "primary_score": metrics.get("primary_score", row.get("primary_score")),
            "val_mse": metrics.get("val_mse", row.get("val_mse")),
            "experiment_success": metrics.get("experiment_success", row.get("experiment_success")),
        }
        nodes.append(
            RichCodexNode(
                node_id=row["node_id"],
                parent_id=row.get("parent_id"),
                agent_id=row["agent_id"],
                generation=int(row["generation"]),
                stage=stage_for_generation(int(row["generation"]), max_generation),
                recipe_id=row.get("recipe_id"),
                inheritance_mode=row.get("inheritance_mode"),
                source_node_ids=list(row.get("source_node_ids") or []),
                patch_recipe_id=row.get("patch_recipe_id"),
                knobs=row.get("knobs") or {},
                metrics=metrics,
                action_path=str(action_path),
                metrics_path=str(metrics_path),
                logs_path=str(logs_path),
                code_diff_path=str(diff_path) if diff_path.exists() else None,
                rationale=row.get("rationale"),
                analysis=analysis_for_row(row),
                is_buggy=not bool(row.get("experiment_success")),
                ablation_name=ablation_name_for_row(row),
            )
        )
    return nodes


def stage_for_generation(generation: int, max_generation: int) -> str:
    if generation == 0:
        return "initial_implementation"
    if generation == 1:
        return "baseline_tuning"
    if generation >= max(2, max_generation - 2):
        return "ablation_studies"
    return "creative_research"


def analysis_for_row(row: dict[str, Any]) -> str:
    score = row.get("primary_score")
    patch = row.get("patch_recipe_id")
    mode = row.get("inheritance_mode")
    return f"{mode} node using {patch}; primary_score={score}, val_mse={row.get('val_mse')}."


def ablation_name_for_row(row: dict[str, Any]) -> str | None:
    mode = row.get("inheritance_mode")
    patch = row.get("patch_recipe_id")
    if mode == "reject":
        return "negative_result_or_simplification_control"
    if patch == "full_budget_action_supervision":
        return "persistent_action_supervision"
    if mode in {"copy", "mutate", "recombine"}:
        return f"{mode}_transfer_control"
    return None


def build_stage_reports(nodes: list[RichCodexNode]) -> list[StageReport]:
    reports = []
    for stage in V2_STAGES:
        stage_nodes = [node for node in nodes if node.stage == stage]
        if stage_nodes:
            best = max(stage_nodes, key=lambda node: float(node.metrics.get("primary_score", -1)))
            best_score = best.metrics.get("primary_score")
            findings = [
                f"Best node {best.node_id} used {best.patch_recipe_id} with score {best_score}.",
                f"Stage contained {len(stage_nodes)} node experiments.",
            ]
        else:
            best = None
            best_score = None
            findings = ["Stage scaffold present; no executable node experiments assigned to this stage yet."]
        reports.append(
            StageReport(
                name=stage,
                goals=STAGE_GOALS[stage],
                total_nodes=len(stage_nodes),
                best_node_id=best.node_id if best else None,
                best_score=float(best_score) if isinstance(best_score, (int, float)) else None,
                findings=findings,
            )
        )
    return reports


def build_manifest(output_dir: Path, rows: list[dict[str, Any]], doctrine_doc: str | None) -> dict[str, Any]:
    best = max(rows, key=score_key)
    return {
        "mode": "codex_scientistv2",
        "output_dir": str(output_dir),
        "doctrine_doc": doctrine_doc,
        "features": {
            "cultural_operators": True,
            "multiagent_population": True,
            "communication_checkpoints": True,
            "rich_node_index": True,
            "stage_reports": True,
            "focused_ablations": True,
            "plot_aggregation": True,
            "workshop_paper_markdown": True,
            "latex_workshop_paper": True,
            "citation_seed": True,
            "codex_review_prompts": True,
            "llm_vlm_review_scaffold": True,
            "noninteractive_codex_backend": False,
        },
        "total_nodes": len(rows),
        "best_node": best.get("node_id"),
        "best_score": best.get("primary_score"),
        "paper": str(output_dir / "latex" / "paper.tex"),
        "paper_markdown_companion": str(output_dir / "paper.md"),
        "search_report": str(output_dir / "search_report.md"),
    }


def build_ablation_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=score_key)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("primary_score")
        if isinstance(score, (int, float)):
            groups[str(row.get("patch_recipe_id") or "unknown")].append(float(score))
    return {
        "best_method": best,
        "focused_comparisons": [
            row for row in sorted(rows, key=score_key, reverse=True)
            if row.get("recipe_id") in {
                "g0_agent1_robust_decoder_loss",
                "g0_agent0_auxiliary_action_contrast",
                "g0_agent2_short_budget_curriculum",
                "g1_agent1_imagination_action_cycle",
                "g2_agent1_counterfactual_imagination_bijection",
                "g4_agent2_reject_complexity_full_budget_actions",
                "g8_agent2_reject_complexity_full_budget_actions",
                "g10_agent2_recombine_best_with_action_grounding",
            }
        ],
        "patch_family_summary": {
            key: {
                "n": len(values),
                "best_score": max(values),
                "mean_score": mean(values),
            }
            for key, values in sorted(groups.items(), key=lambda item: max(item[1]), reverse=True)
        },
    }


def write_literature_seed(v2_dir: Path) -> None:
    lit_dir = ensure_dir(v2_dir / "literature")
    references = [
        {
            "key": "ha2018worldmodels",
            "title": "World Models",
            "authors": "Ha and Schmidhuber",
            "year": 2018,
            "relevance": "Foundational latent world-model framing.",
        },
        {
            "key": "prelar2024",
            "title": "World Model Pre-training with Learnable Action Representation",
            "authors": "Anonymous/ECVV listing",
            "year": 2024,
            "relevance": "Action representation motivation.",
        },
        {
            "key": "aiscientist2024",
            "title": "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery",
            "authors": "Lu et al.",
            "year": 2024,
            "relevance": "Automated research loop precedent.",
        },
        {
            "key": "aiscientistv2_2025",
            "title": "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search",
            "authors": "Yamada et al.",
            "year": 2025,
            "relevance": "Staged tree search and workshop paper generation.",
        },
    ]
    write_json(lit_dir / "literature_seed.json", references)
    write_text(
        lit_dir / "references.bib",
        "\n\n".join(
            [
                "@article{ha2018worldmodels,\n  title={World Models},\n  author={Ha, David and Schmidhuber, Juergen},\n  year={2018}\n}",
                "@inproceedings{prelar2024,\n  title={World Model Pre-training with Learnable Action Representation},\n  booktitle={ECCV},\n  year={2024}\n}",
                "@article{aiscientist2024,\n  title={The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery},\n  author={Lu et al.},\n  journal={arXiv preprint arXiv:2408.06292},\n  year={2024}\n}",
                "@article{aiscientistv2_2025,\n  title={The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search},\n  author={Yamada et al.},\n  journal={arXiv preprint arXiv:2504.08066},\n  year={2025}\n}",
            ]
        )
        + "\n",
    )


def write_figures(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig_dir = ensure_dir(output_dir / "figures")
    best_by_generation = []
    for generation in sorted({int(row["generation"]) for row in rows}):
        best_by_generation.append(max([row for row in rows if int(row["generation"]) == generation], key=score_key))
    write_svg_line_chart(
        fig_dir / "best_score_by_generation.svg",
        [int(row["generation"]) for row in best_by_generation],
        [float(row["primary_score"]) for row in best_by_generation],
        title="Best TinyWorlds score by generation",
        y_label="primary score",
    )
    patch_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("primary_score")
        if isinstance(score, (int, float)):
            patch_groups[str(row.get("patch_recipe_id") or "unknown")].append(float(score))
    write_svg_bar_chart(
        fig_dir / "patch_family_mean_scores.svg",
        {key: mean(values) for key, values in patch_groups.items()},
        title="Mean score by patch family",
    )
    write_matplotlib_pdf_figures(fig_dir, best_by_generation, patch_groups)


def write_matplotlib_pdf_figures(fig_dir: Path, best_by_generation: list[dict[str, Any]], patch_groups: dict[str, list[float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_text(fig_dir / "pdf_figure_generation.log", f"Skipped PDF figure generation: {exc}\n")
        return

    xs = [int(row["generation"]) for row in best_by_generation]
    ys = [float(row["primary_score"]) for row in best_by_generation]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xs, ys, marker="o", linewidth=2)
    ax.set_title("Best TinyWorlds score by generation")
    ax.set_xlabel("generation")
    ax.set_ylabel("primary score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_score_by_generation.pdf")
    plt.close(fig)

    items = sorted(
        [(key, mean(values)) for key, values in patch_groups.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(range(len(values)), values, color="#2ca02c")
    ax.set_title("Mean score by patch family")
    ax.set_ylabel("mean primary score")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "patch_family_mean_scores.pdf")
    plt.close(fig)


def write_tree_visualizations(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig_dir = ensure_dir(output_dir / "figures")
    write_tree_svg(fig_dir / "cultural_tree.svg", rows)
    write_tree_pdf(fig_dir / "cultural_tree.pdf", rows)
    write_json(fig_dir / "cultural_tree_index.json", {"nodes": rows, "edge_semantics": {
        "solid": "explicit cultural source node",
        "dotted": "same-agent parent node",
    }})


def write_tree_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    max_generation = max(int(row["generation"]) for row in rows)
    agents = sorted({str(row["agent_id"]) for row in rows})
    width = max(1200, 220 * (max_generation + 1) + 180)
    height = 150 * len(agents) + 140
    x_for_generation = lambda generation: 110 + int(generation) * 220
    y_for_agent = {agent: 95 + index * 150 for index, agent in enumerate(agents)}
    by_id = {row["node_id"]: row for row in rows}
    best_id = max(rows, key=score_key)["node_id"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#666"/></marker></defs>',
        '<text x="24" y="30" font-family="Arial" font-size="20" font-weight="700">Codex-Scientist cultural tree</text>',
        '<text x="24" y="52" font-family="Arial" font-size="12" fill="#555">solid = cultural source transfer; dotted = same-agent parent lineage; gold = best node</text>',
    ]
    for generation in range(max_generation + 1):
        x = x_for_generation(generation)
        elements.append(f'<text x="{x}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="12">g{generation}</text>')
        elements.append(f'<line x1="{x}" y1="65" x2="{x}" y2="{height - 45}" stroke="#eee"/>')
    for agent, y in y_for_agent.items():
        elements.append(f'<text x="18" y="{y + 5}" font-family="Arial" font-size="12" fill="#333">{escape(agent)}</text>')
        elements.append(f'<line x1="80" y1="{y}" x2="{width - 40}" y2="{y}" stroke="#f5f5f5"/>')
    for row in rows:
        target_x = x_for_generation(int(row["generation"]))
        target_y = y_for_agent[str(row["agent_id"])]
        parent_id = row.get("parent_id")
        if parent_id in by_id:
            parent = by_id[parent_id]
            source_x = x_for_generation(int(parent["generation"]))
            source_y = y_for_agent[str(parent["agent_id"])]
            elements.append(
                f'<line x1="{source_x + 58}" y1="{source_y}" x2="{target_x - 58}" y2="{target_y}" '
                'stroke="#999" stroke-width="1.2" stroke-dasharray="4 4" marker-end="url(#arrow)"/>'
            )
        for source_id in row.get("source_node_ids") or []:
            if source_id in by_id:
                source = by_id[source_id]
                source_x = x_for_generation(int(source["generation"]))
                source_y = y_for_agent[str(source["agent_id"])]
                elements.append(
                    f'<line x1="{source_x + 58}" y1="{source_y}" x2="{target_x - 58}" y2="{target_y}" '
                    'stroke="#555" stroke-width="1.4" opacity="0.65" marker-end="url(#arrow)"/>'
                )
    for row in rows:
        x = x_for_generation(int(row["generation"]))
        y = y_for_agent[str(row["agent_id"])]
        fill = color_for_patch(row.get("patch_recipe_id"))
        stroke = "#d18f00" if row["node_id"] == best_id else "#444"
        stroke_width = 3 if row["node_id"] == best_id else 1
        label_1 = f"{row['agent_id']} n{int(row['generation']) + 1}"
        label_2 = f"{row.get('inheritance_mode')} / {row.get('patch_recipe_id')}"
        label_3 = f"score {row.get('primary_score')}"
        elements.append(
            f'<rect x="{x - 62}" y="{y - 34}" width="124" height="68" rx="6" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
        elements.append(f'<text x="{x}" y="{y - 13}" text-anchor="middle" font-family="Arial" font-size="10" font-weight="700">{escape(label_1)}</text>')
        elements.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-family="Arial" font-size="8">{escape(label_2[:28])}</text>')
        elements.append(f'<text x="{x}" y="{y + 20}" text-anchor="middle" font-family="Arial" font-size="9">{escape(label_3)}</text>')
    elements.append("</svg>\n")
    write_text(path, "\n".join(elements))


def write_tree_pdf(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception as exc:
        write_text(path.with_suffix(".pdf.log"), f"Skipped tree PDF generation: {exc}\n")
        return
    max_generation = max(int(row["generation"]) for row in rows)
    agents = sorted({str(row["agent_id"]) for row in rows})
    fig, ax = plt.subplots(figsize=(max(12, 1.2 * (max_generation + 1)), 1.5 * len(agents) + 1.2))
    y_for_agent = {agent: len(agents) - index for index, agent in enumerate(agents)}
    by_id = {row["node_id"]: row for row in rows}
    best_id = max(rows, key=score_key)["node_id"]
    for row in rows:
        x = int(row["generation"])
        y = y_for_agent[str(row["agent_id"])]
        parent_id = row.get("parent_id")
        if parent_id in by_id:
            parent = by_id[parent_id]
            ax.annotate("", xy=(x - 0.33, y), xytext=(int(parent["generation"]) + 0.33, y_for_agent[str(parent["agent_id"])]),
                        arrowprops={"arrowstyle": "->", "color": "#999", "linestyle": "dotted", "lw": 1.0})
        for source_id in row.get("source_node_ids") or []:
            if source_id in by_id:
                source = by_id[source_id]
                ax.annotate("", xy=(x - 0.33, y), xytext=(int(source["generation"]) + 0.33, y_for_agent[str(source["agent_id"])]),
                            arrowprops={"arrowstyle": "->", "color": "#555", "lw": 1.0, "alpha": 0.6})
    for row in rows:
        x = int(row["generation"])
        y = y_for_agent[str(row["agent_id"])]
        patch = FancyBboxPatch(
            (x - 0.32, y - 0.25),
            0.64,
            0.5,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=2.0 if row["node_id"] == best_id else 0.8,
            edgecolor="#d18f00" if row["node_id"] == best_id else "#444",
            facecolor=color_for_patch(row.get("patch_recipe_id")),
        )
        ax.add_patch(patch)
        ax.text(x, y + 0.08, f"{row['agent_id']} g{row['generation']}", ha="center", va="center", fontsize=6, weight="bold")
        ax.text(x, y - 0.06, str(row.get("inheritance_mode")), ha="center", va="center", fontsize=5)
        ax.text(x, y - 0.18, f"{float(row.get('primary_score') or 0):.3f}", ha="center", va="center", fontsize=5)
    ax.set_xlim(-0.8, max_generation + 0.8)
    ax.set_ylim(0.4, len(agents) + 0.8)
    ax.set_yticks(list(y_for_agent.values()))
    ax.set_yticklabels(list(y_for_agent.keys()))
    ax.set_xticks(range(max_generation + 1))
    ax.set_xlabel("generation")
    ax.set_title("Codex-Scientist cultural lineage tree")
    ax.grid(True, axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def color_for_patch(patch_recipe_id: Any) -> str:
    palette = {
        "full_budget_action_supervision": "#ffe08a",
        "smooth_l1_dynamics_pixel": "#cde7ff",
        "action_grad_dynamics": "#d4f4dd",
        "dynamics_first_schedule": "#ead7ff",
        "sharpen_change_weights": "#ffd6cc",
        "baseline_no_patch": "#eeeeee",
    }
    return palette.get(str(patch_recipe_id), "#f7f7f7")


def write_svg_line_chart(path: Path, xs: list[int], ys: list[float], title: str, y_label: str) -> None:
    width, height = 760, 420
    pad = 60
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1, max_x - min_x)
    span_y = max(1e-9, max_y - min_y)
    points = []
    for x, y in zip(xs, ys):
        px = pad + (x - min_x) / span_x * (width - 2 * pad)
        py = height - pad - (y - min_y) / span_y * (height - 2 * pad)
        points.append((px, py))
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "\n".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f77b4"/>' for x, y in points)
    safe_title = escape(title)
    safe_y_label = escape(y_label)
    write_text(
        path,
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{safe_title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="13">generation</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="13">{safe_y_label}</text>
<polyline fill="none" stroke="#1f77b4" stroke-width="3" points="{polyline}"/>
{circles}
<text x="{pad}" y="{height-pad+22}" font-family="Arial" font-size="11">{min_x}</text>
<text x="{width-pad}" y="{height-pad+22}" text-anchor="end" font-family="Arial" font-size="11">{max_x}</text>
<text x="{pad-8}" y="{height-pad}" text-anchor="end" font-family="Arial" font-size="11">{min_y:.3f}</text>
<text x="{pad-8}" y="{pad+4}" text-anchor="end" font-family="Arial" font-size="11">{max_y:.3f}</text>
</svg>
""",
    )


def write_svg_bar_chart(path: Path, values: dict[str, float], title: str) -> None:
    width, height = 900, 460
    pad = 70
    items = sorted(values.items(), key=lambda item: item[1], reverse=True)
    max_v = max(values.values()) if values else 1.0
    bar_w = (width - 2 * pad) / max(1, len(items))
    bars = []
    for i, (label, value) in enumerate(items):
        x = pad + i * bar_w + bar_w * 0.15
        h = (value / max_v) * (height - 2 * pad)
        y = height - pad - h
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.7:.1f}" height="{h:.1f}" fill="#2ca02c"/>')
        bars.append(f'<text x="{x + bar_w*0.35:.1f}" y="{height-pad+14}" text-anchor="middle" font-family="Arial" font-size="10" transform="rotate(25 {x + bar_w*0.35:.1f} {height-pad+14})">{escape(label)}</text>')
        bars.append(f'<text x="{x + bar_w*0.35:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3f}</text>')
    safe_title = escape(title)
    write_text(
        path,
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{safe_title}</text>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>
{chr(10).join(bars)}
</svg>
""",
    )


def write_codex_tasks(output_dir: Path, rows: list[dict[str, Any]], doctrine_doc: str | None) -> None:
    task_dir = ensure_dir(output_dir / "codex_scientistv2" / "codex_tasks")
    best = max(rows, key=score_key)
    shared = f"""Run: {output_dir}
Best node: {best.get('node_id')}
Best recipe: {best.get('recipe_id')}
Best score: {best.get('primary_score')}
Doctrine: {doctrine_doc}
"""
    tasks = {
        "literature_search.md": "Find 6-10 current papers directly relevant to the domain paper claim. Return BibTeX and one-sentence relevance notes.",
        "plot_aggregation.md": "Inspect the run summaries and generated SVGs. Propose at most 3 final workshop figures and say which should appear in the main paper.",
        "paper_reflection.md": "Review latex/paper.tex as a workshop submission. Improve claims, ablations, citations, and limitations without inventing results.",
        "llm_review.md": "Write a NeurIPS-style review JSON for latex/paper.tex: strengths, weaknesses, soundness, presentation, contribution, questions, decision.",
        "vlm_review.md": "Review figures/*.svg for clarity and whether they support the paper claims. Recommend edits or omissions.",
    }
    for filename, instruction in tasks.items():
        write_text(task_dir / filename, f"# Codex-Scientist-v2 Task\n\n{shared}\n\n{instruction}\n")


def write_latex_workshop_stub(output_dir: Path) -> None:
    latex_dir = ensure_dir(output_dir / "latex")
    paper_text = (output_dir / "paper.md").read_text(encoding="utf-8") if (output_dir / "paper.md").exists() else ""
    refs_src = output_dir / "codex_scientistv2" / "literature" / "references.bib"
    if refs_src.exists():
        shutil.copyfile(refs_src, latex_dir / "references.bib")
    paper_tex = write_domain_latex_from_run(output_dir, latex_dir / "paper.tex")
    compile_latex_paper(paper_tex)
    write_text(
        latex_dir / "template.tex",
        r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\title{Persistent Action Supervision Improves Short-Budget TinyWorlds World Models}
\author{Codex-Scientist-v2}
\begin{document}
\maketitle
\begin{abstract}
This workshop-style template is a compile-ready starting point. Convert paper.md into LaTeX here during the Codex paper reflection/writeup stage.
\end{abstract}
\section{Markdown Source}
\begin{verbatim}
"""
        + paper_text[:12000]
        + r"""
\end{verbatim}
\bibliography{references}
\bibliographystyle{plain}
\end{document}
""",
    )


def compile_latex_paper(paper_tex: Path) -> None:
    latex_dir = paper_tex.parent
    engine = shutil.which("pdflatex") or shutil.which("xelatex")
    log_path = latex_dir / "compile.log"
    if not engine:
        write_text(
            log_path,
            "Skipped LaTeX compilation: no pdflatex or xelatex executable was found on PATH.\n"
            f"Manuscript written to {paper_tex}.\n",
        )
        return
    commands = [
        [engine, "-interaction=nonstopmode", paper_tex.name],
        [engine, "-interaction=nonstopmode", paper_tex.name],
    ]
    logs = []
    for command in commands:
        result = subprocess.run(command, cwd=latex_dir, capture_output=True, text=True, check=False)
        logs.append(f"$ {' '.join(command)}\n")
        logs.append(result.stdout)
        logs.append(result.stderr)
        if result.returncode != 0:
            logs.append(f"\nCommand failed with return code {result.returncode}.\n")
            break
    write_text(log_path, "\n".join(logs))


def write_review_scaffold(output_dir: Path) -> None:
    review_dir = ensure_dir(output_dir / "review")
    paper_path = output_dir / "latex" / "paper.tex"
    write_json(
        review_dir / "review_scaffold.json",
        {
            "status": "pending_live_codex_review",
            "paper": str(paper_path),
            "llm_review_prompt": str(output_dir / "codex_scientistv2" / "codex_tasks" / "llm_review.md"),
            "vlm_review_prompt": str(output_dir / "codex_scientistv2" / "codex_tasks" / "vlm_review.md"),
        },
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
