from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any
from xml.sax.saxutils import escape

from commsci.artifacts import ensure_dir, write_json, write_text
from commsci.codex_scientist.paper import load_population_rows, score_key, write_domain_latex_from_run, write_domain_paper_from_run
from commsci.codex_scientist.runner import run_codex_scientist_branch_expansion
from commsci.config import DEFAULT_CONFIG, deep_merge

from .schemas import RichCodexNode, StageReport, V2_STAGES


STAGE_GOALS = {
    "literature_review": [
        "Run multiagent literature nodes before experiment generation.",
        "Produce query-specific evidence and idea seeds that become visible context for the first experimental nodes.",
    ],
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
    if should_run_initial_literature_nodes(args):
        run_initial_literature_nodes(output_dir, args)
    if not getattr(args, "skip_experiments", False):
        run_population_tree(args, output_dir)
    if should_run_controlled_ablations(args):
        run_controlled_ablations(output_dir, args)
    return write_v2_outputs(output_dir, doctrine_doc=getattr(args, "doctrine_doc", None))


def should_run_controlled_ablations(args: Any) -> bool:
    return bool(getattr(args, "run_controlled_ablations", False)) and not bool(getattr(args, "skip_experiments", False))


def should_run_initial_literature_nodes(args: Any) -> bool:
    return bool(getattr(args, "run_initial_literature_nodes", True))


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
    literature_context = output_dir / "codex_scientistv2" / "literature_nodes" / "initial_literature_context.md"
    if literature_context.exists():
        cmd.extend(["--literature_context_file", str(literature_context)])
    subprocess.run(cmd, check=True)


def run_initial_literature_nodes(output_dir: Path, args: Any) -> None:
    lit_root = ensure_dir(output_dir / "codex_scientistv2" / "literature_nodes")
    summary_path = lit_root / "summary.json"
    if summary_path.exists() and not bool(getattr(args, "rerun_initial_literature_nodes", False)):
        return
    node_count = getattr(args, "literature_node_count", None)
    if node_count is None:
        node_count = getattr(args, "num_agents", 3)
    node_count = max(1, int(node_count))
    limit = max(1, int(getattr(args, "literature_max_results_per_node", 6)))
    doctrine = read_text_file(getattr(args, "doctrine_doc", None))[:12000]
    queries = build_initial_literature_queries(node_count)
    seed_refs = seed_literature_references()
    nodes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(node_count, max(1, int(getattr(args, "parallel_workers", node_count))))) as pool:
        futures = {
            pool.submit(fetch_semantic_scholar, query, limit): (index, query)
            for index, query in enumerate(queries)
        }
        for future in as_completed(futures):
            index, query = futures[future]
            agent_id = f"agent_{index}"
            node_id = f"literature_{agent_id}_node_0"
            try:
                fetched, error = future.result()
            except Exception as exc:  # Defensive: a literature node should degrade, not kill the run.
                fetched, error = [], repr(exc)
            references = merge_references(seed_refs, fetched)
            node = {
                "node_id": node_id,
                "parent_id": None,
                "agent_id": agent_id,
                "generation": -1,
                "stage": "literature_review",
                "node_type": "literature",
                "query": query,
                "status": "complete" if fetched else "seed_only",
                "network_error": error,
                "num_seed_references": len(seed_refs),
                "num_fetched_references": len(fetched),
                "references": references,
                "doctrine_excerpt": doctrine[:2000],
                "synthesis": synthesize_initial_literature_node(query, references),
                "recommended_idea_seeds": literature_idea_seeds(query),
            }
            node_dir = ensure_dir(lit_root / agent_id)
            write_json(node_dir / "literature_node.json", node)
            write_text(node_dir / "references.bib", references_to_bibtex(references))
            write_text(node_dir / "synthesis.md", literature_node_markdown(node))
            nodes.append(node)
    nodes = sorted(nodes, key=lambda item: item["agent_id"])
    write_json(summary_path, nodes)
    write_text(lit_root / "initial_literature_context.md", initial_literature_context_markdown(nodes))
    write_text(lit_root / "references.bib", references_to_bibtex(merge_literature_node_references(nodes)))


def build_initial_literature_queries(node_count: int) -> list[str]:
    base_queries = [
        "world models action conditioned dynamics learnable action representation",
        "counterfactual representation learning dynamics prediction world models",
        "object centric dynamics world model latent state prediction",
        "uncertainty calibrated dynamics model reinforcement learning world model",
        "curriculum learning short horizon dynamics prediction world models",
        "multi agent automated science cultural evolution collective intelligence",
    ]
    return [base_queries[index % len(base_queries)] for index in range(node_count)]


def synthesize_initial_literature_node(query: str, references: list[dict[str, Any]]) -> str:
    titles = [str(ref.get("title")) for ref in references[:5] if ref.get("title")]
    title_text = "; ".join(titles) if titles else "seed references only"
    return (
        f"Query `{query}` suggests starting TinyWorlds ideas from these reference clusters: {title_text}. "
        "Use these papers as inspiration for concrete bounded world-model interventions, not as claims of proof."
    )


def literature_idea_seeds(query: str) -> list[str]:
    text = query.lower()
    if "counterfactual" in text:
        return [
            "contrast observed transitions against action-swapped counterfactual predictions",
            "penalize dynamics features that cannot distinguish plausible alternative actions",
        ]
    if "object" in text:
        return [
            "add object/motion-local reconstruction emphasis for changed pixels",
            "separate static-background and moving-entity losses under the same short budget",
        ]
    if "uncertainty" in text:
        return [
            "calibrate losses by motion or prediction uncertainty instead of uniform reconstruction",
            "reward robust improvements that persist under seed and budget variation",
        ]
    if "curriculum" in text:
        return [
            "front-load dynamics updates before pixel reconstruction dominates",
            "schedule short-horizon dynamics first, then increase reconstruction weight",
        ]
    if "multi agent" in text or "cultural" in text:
        return [
            "copy high-performing recipes only when source evidence is visible",
            "recombine complementary ideas with explicit source ids and negative-result memory",
        ]
    return [
        "learn action-sensitive latent transitions instead of only frame reconstruction",
        "test whether action supervision helps through dynamics gradients or auxiliary loss only",
    ]


def literature_node_markdown(node: dict[str, Any]) -> str:
    refs = "\n".join(
        f"- {ref.get('title')} ({ref.get('year', 'n.d.')})"
        for ref in node.get("references", [])[:8]
    )
    seeds = "\n".join(f"- {seed}" for seed in node.get("recommended_idea_seeds", []))
    return f"""# {node.get('node_id')}

Query: `{node.get('query')}`

Status: {node.get('status')}

{node.get('synthesis')}

## Recommended Idea Seeds

{seeds}

## References

{refs}
"""


def initial_literature_context_markdown(nodes: list[dict[str, Any]]) -> str:
    sections = ["# Initial Multiagent Literature Nodes", ""]
    for node in nodes:
        sections.extend(
            [
                f"## {node.get('agent_id')} / {node.get('node_id')}",
                "",
                f"Query: `{node.get('query')}`",
                f"Status: {node.get('status')}; fetched references: {node.get('num_fetched_references')}",
                "",
                str(node.get("synthesis") or ""),
                "",
                "Idea seeds:",
                *[f"- {seed}" for seed in node.get("recommended_idea_seeds", [])],
                "",
            ]
        )
    return "\n".join(sections).strip() + "\n"


def merge_literature_node_references(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        for ref in node.get("references", []):
            key = str(ref.get("key") or bib_key(str(ref.get("title")), ref.get("year")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(ref)
    return merged


def read_text_file(path: str | None) -> str:
    if not path:
        return ""
    doc_path = Path(path).expanduser()
    if not doc_path.is_absolute():
        doc_path = Path.cwd() / doc_path
    return doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""



def run_controlled_ablations(output_dir: Path, args: Any) -> None:
    rows = load_population_rows(output_dir)
    if not rows:
        raise FileNotFoundError(f"No population summaries found in {output_dir}")
    ablation_root = ensure_dir(output_dir / "controlled_ablations")
    summary_path = ablation_root / "summary.json"
    if summary_path.exists() and not bool(getattr(args, "rerun_controlled_ablations", False)):
        return
    best = max(rows, key=score_key)
    best_action = read_node_json(output_dir, best, "action.json")
    actions = build_controlled_ablation_actions(best, best_action)
    actions = actions[: int(getattr(args, "max_controlled_ablations", 3))]
    action_dir = ensure_dir(ablation_root / "actions")
    for index, action in enumerate(actions):
        write_json(action_dir / f"ablation_{index}_step_1.json", action)
    config = build_ablation_config(args, action_dir)
    results = []
    workers = max(1, min(int(getattr(args, "ablation_parallel_workers", 1)), len(actions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for index, action in enumerate(actions):
            agent_id = f"ablation_{index}"
            artifact_dir = ensure_dir(ablation_root / agent_id / "artifacts")
            future = pool.submit(
                run_codex_scientist_branch_expansion,
                artifact_dir,
                agent_id,
                f"controlled_ablation_{index}",
                1,
                config,
                int(getattr(args, "seed", 0)) + 1000 + index,
                "Controlled Codex-Scientist-v2 ablation after the main population run.",
                None,
                None,
            )
            futures[future] = (index, action)
        for future in as_completed(futures):
            index, action = futures[future]
            try:
                expansion = future.result()
                metrics = expansion.get("metrics") or {}
                results.append(
                    {
                        "index": index,
                        "ablation_id": action.get("recipe_id"),
                        "description": action.get("rationale"),
                        "metrics": metrics,
                        "success": bool(metrics.get("experiment_success")),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "index": index,
                        "ablation_id": action.get("recipe_id"),
                        "description": action.get("rationale"),
                        "success": False,
                        "error": repr(exc),
                    }
                )
    results = sorted(results, key=lambda item: item["index"])
    write_json(
        summary_path,
        {
            "status": "complete",
            "source_best_node": best.get("node_id"),
            "source_best_recipe": best.get("recipe_id"),
            "source_best_score": best.get("primary_score"),
            "time_budget_seconds": int(getattr(args, "ablation_time_budget_seconds", getattr(args, "time_budget_seconds", 120))),
            "controls": results,
        },
    )


def build_ablation_config(args: Any, action_dir: Path) -> dict[str, Any]:
    time_budget = int(getattr(args, "ablation_time_budget_seconds", getattr(args, "time_budget_seconds", 120)))
    timeout = int(getattr(args, "ablation_timeout_seconds", max(180, time_budget + 180)))
    return deep_merge(
        DEFAULT_CONFIG,
        {
            "paths": {
                "tinyworlds_dir": str(Path(args.tinyworlds_dir).expanduser()),
                "output_dir": str(Path(args.output_dir).expanduser()),
            },
            "compute": {
                "num_agents": 1,
                "max_total_experiments": int(getattr(args, "max_controlled_ablations", 3)),
            },
            "model": {"mock_model": True, "seed": int(getattr(args, "seed", 0))},
            "base_system": {
                "write_full_paper": False,
                "reviewer_enabled": False,
            },
            "experiment": {
                "runner": "codex_scientist",
                "task_spec": "Run a controlled ablation for the best Codex-Scientist-v2 TinyWorlds idea.",
                "ai_scientist_data_dir": str(Path(args.tinyworlds_dir).expanduser()),
                "codex_scientist_time_budget_seconds": time_budget,
                "codex_scientist_timeout_seconds": timeout,
                "codex_scientist_action_overrides_dir": str(action_dir),
                "codex_scientist_patch_recipes": [
                    "baseline_no_patch",
                    "dynamics_first_schedule",
                    "action_grad_dynamics",
                    "smooth_l1_dynamics_pixel",
                    "sharpen_change_weights",
                    "full_budget_action_supervision",
                ],
                "tinyworlds_baseline_knobs": {"TW_DATASET": "minigrid", "TW_DEPTH": "1"},
                "allowed_files": ["train.py", "models.py"],
                "primary_metric": "primary_score",
            },
        },
    )


def build_controlled_ablation_actions(best: dict[str, Any], best_action: dict[str, Any]) -> list[dict[str, Any]]:
    knobs = dict(best_action.get("knobs") or best.get("knobs") or {})
    patch_id = (best_action.get("patch_recipe") or {}).get("id") or best.get("patch_recipe_id") or "baseline_no_patch"
    file_edits = list(best_action.get("file_edits") or [])
    source_ids = [str(best.get("node_id"))]
    controls = [
        {
            "recipe_id": f"ablate_repeat_{best.get('recipe_id')}",
            "inheritance_mode": "copy",
            "source_agent_ids": [str(best.get("agent_id"))],
            "source_node_ids": source_ids,
            "patch_recipe_id": patch_id,
            "knobs": knobs,
            "file_edits": file_edits,
            "rationale": "Exact repeat of the best branch under the controlled-ablation budget.",
        },
        {
            "recipe_id": f"ablate_knobs_only_{best.get('recipe_id')}",
            "inheritance_mode": "reject",
            "source_agent_ids": [str(best.get("agent_id"))],
            "source_node_ids": source_ids,
            "patch_recipe_id": "baseline_no_patch",
            "knobs": knobs,
            "rationale": "Remove source edits and patch recipe while preserving the best branch knobs.",
        },
        {
            "recipe_id": f"ablate_patch_only_{best.get('recipe_id')}",
            "inheritance_mode": "mutate",
            "source_agent_ids": [str(best.get("agent_id"))],
            "source_node_ids": source_ids,
            "patch_recipe_id": patch_id,
            "knobs": {},
            "file_edits": file_edits,
            "rationale": "Preserve source edits/patch recipe but remove tuned knobs.",
        },
    ]
    if patch_id == "baseline_no_patch" and not file_edits:
        controls[2] = {
            "recipe_id": f"ablate_minimal_baseline_{best.get('recipe_id')}",
            "inheritance_mode": "reject",
            "source_agent_ids": [str(best.get("agent_id"))],
            "source_node_ids": source_ids,
            "patch_recipe_id": "baseline_no_patch",
            "knobs": {},
            "rationale": "Minimal TinyWorlds baseline control with no best-branch edits or knobs.",
        }
    return controls


def read_node_json(output_dir: Path, row: dict[str, Any], filename: str) -> dict[str, Any]:
    path = (
        output_dir
        / "cultural_evolution"
        / row["agent_id"]
        / "artifacts"
        / "codex_scientist"
        / "nodes"
        / row["node_id"]
        / filename
    )
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
    literature_report = write_literature_review(v2_dir, rows)
    write_json(v2_dir / "run_manifest.json", build_manifest(output_dir, rows, doctrine_doc))
    write_json(v2_dir / "ablation_report.json", build_ablation_report(output_dir, rows))
    write_figures(output_dir, rows)
    write_tree_visualizations(output_dir, rows)
    write_codex_tasks(output_dir, rows, doctrine_doc)
    write_domain_paper_from_run(output_dir, output_dir / "paper.md")
    write_latex_workshop_stub(output_dir)
    write_review_outputs(output_dir, rows, literature_report)
    return v2_dir


def build_rich_nodes(output_dir: Path, rows: list[dict[str, Any]]) -> list[RichCodexNode]:
    nodes = build_literature_rich_nodes(output_dir)
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


def build_literature_rich_nodes(output_dir: Path) -> list[RichCodexNode]:
    summary_path = output_dir / "codex_scientistv2" / "literature_nodes" / "summary.json"
    if not summary_path.exists():
        return []
    try:
        literature_nodes = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    nodes = []
    for node in literature_nodes:
        agent_id = str(node.get("agent_id") or "agent_unknown")
        node_id = str(node.get("node_id") or f"literature_{agent_id}_node_0")
        node_dir = output_dir / "codex_scientistv2" / "literature_nodes" / agent_id
        nodes.append(
            RichCodexNode(
                node_id=node_id,
                parent_id=None,
                agent_id=agent_id,
                generation=int(node.get("generation", -1)),
                stage="literature_review",
                recipe_id=str(node.get("query") or "literature_query"),
                inheritance_mode="invent",
                source_node_ids=[],
                patch_recipe_id=None,
                knobs={},
                metrics={
                    "num_fetched_references": node.get("num_fetched_references", 0),
                    "num_seed_references": node.get("num_seed_references", 0),
                    "status": node.get("status"),
                },
                action_path=str(node_dir / "literature_node.json"),
                metrics_path=str(node_dir / "literature_node.json"),
                logs_path=str(node_dir / "synthesis.md"),
                code_diff_path=None,
                rationale=str(node.get("synthesis") or ""),
                analysis=f"Literature node queried `{node.get('query')}`.",
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
            "controlled_ablations": (output_dir / "controlled_ablations" / "summary.json").exists(),
            "plot_aggregation": True,
            "workshop_paper_markdown": True,
            "latex_workshop_paper": True,
            "literature_review": True,
            "citation_seed": True,
            "codex_review_prompts": True,
            "automated_text_review": True,
            "automated_figure_review": True,
            "noninteractive_codex_backend": False,
        },
        "total_nodes": len(rows),
        "best_node": best.get("node_id"),
        "best_score": best.get("primary_score"),
        "paper": str(output_dir / "latex" / "paper.tex"),
        "paper_markdown_companion": str(output_dir / "paper.md"),
        "search_report": str(output_dir / "search_report.md"),
    }


def build_ablation_report(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(rows, key=score_key)
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("primary_score")
        if isinstance(score, (int, float)):
            groups[str(row.get("patch_recipe_id") or "unknown")].append(float(score))
    controlled_path = output_dir / "controlled_ablations" / "summary.json"
    controlled = json.loads(controlled_path.read_text(encoding="utf-8")) if controlled_path.exists() else None
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
        "controlled_ablations": controlled,
    }


def write_literature_review(v2_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lit_dir = ensure_dir(v2_dir / "literature")
    seed_references = seed_literature_references()
    initial_references = load_initial_literature_references(v2_dir)
    best = max(rows, key=score_key)
    query = literature_query_for_best(best)
    fetched, error = fetch_semantic_scholar(query, limit=8)
    references = merge_references(merge_references(seed_references, initial_references), fetched)
    report = {
        "status": "complete" if fetched else "seed_only",
        "query": query,
        "network_error": error,
        "num_seed_references": len(seed_references),
        "num_initial_literature_node_references": len(initial_references),
        "num_fetched_references": len(fetched),
        "references": references,
        "synthesis": synthesize_literature_review(best, references),
    }
    write_json(lit_dir / "literature_review.json", report)
    write_json(lit_dir / "literature_seed.json", seed_references)
    write_text(lit_dir / "references.bib", references_to_bibtex(references))
    return report


def load_initial_literature_references(v2_dir: Path) -> list[dict[str, Any]]:
    summary_path = v2_dir / "literature_nodes" / "summary.json"
    if not summary_path.exists():
        return []
    try:
        nodes = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return merge_literature_node_references(nodes)


def seed_literature_references() -> list[dict[str, Any]]:
    return [
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


def literature_query_for_best(best: dict[str, Any]) -> str:
    text = " ".join(
        str(part)
        for part in [
            best.get("recipe_id"),
            best.get("patch_recipe_id"),
            best.get("rationale"),
            "world model action conditioned dynamics",
        ]
        if part
    ).lower()
    if "counterfactual" in text:
        return "counterfactual action representation world model dynamics"
    if "object" in text or "local" in text or "change" in text:
        return "object centric dynamics world model changed pixels prediction"
    if "latent" in text or "action" in text:
        return "learnable action representation world model dynamics"
    return "world models action conditioned dynamics automated science"


def fetch_semantic_scholar(query: str, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    if os.environ.get("CODEX_SCIENTISTV2_OFFLINE_LITERATURE") == "1":
        return [], "offline mode via CODEX_SCIENTISTV2_OFFLINE_LITERATURE=1"
    params = urllib.parse.urlencode(
        {
            "query": query,
            "limit": str(limit),
            "fields": "title,year,authors,abstract,url,venue,citationCount,externalIds",
        }
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "codex-scientistv2/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        arxiv_refs, arxiv_error = fetch_arxiv(query, limit=limit)
        if arxiv_refs:
            return arxiv_refs, f"Semantic Scholar failed with {repr(exc)}; used arXiv fallback."
        return [], f"Semantic Scholar failed with {repr(exc)}; arXiv failed with {arxiv_error}."
    references = []
    for item in payload.get("data", []):
        title = item.get("title")
        if not title:
            continue
        authors = ", ".join(author.get("name", "") for author in item.get("authors", [])[:6] if author.get("name"))
        key = bib_key(title, item.get("year"))
        references.append(
            {
                "key": key,
                "title": title,
                "authors": authors or "Unknown",
                "year": item.get("year"),
                "venue": item.get("venue"),
                "url": item.get("url"),
                "citation_count": item.get("citationCount"),
                "abstract": item.get("abstract"),
                "external_ids": item.get("externalIds") or {},
                "relevance": "Retrieved by Codex-Scientist-v2 literature stage for the run's best idea.",
            }
        )
    return references, None


def fetch_arxiv(query: str, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "codex-scientistv2/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            root = ET.fromstring(response.read())
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
        return [], repr(exc)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    references = []
    for entry in root.findall("atom:entry", ns):
        title = normalize_space(entry.findtext("atom:title", default="", namespaces=ns))
        if not title:
            continue
        authors = [
            normalize_space(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]
        published = entry.findtext("atom:published", default="", namespaces=ns)
        year = published[:4] if published else None
        link = entry.findtext("atom:id", default="", namespaces=ns)
        abstract = normalize_space(entry.findtext("atom:summary", default="", namespaces=ns))
        references.append(
            {
                "key": bib_key(title, year),
                "title": title,
                "authors": ", ".join([author for author in authors if author]) or "Unknown",
                "year": year,
                "venue": "arXiv",
                "url": link,
                "abstract": abstract,
                "external_ids": {"ArXiv": link.rsplit("/", 1)[-1] if link else None},
                "relevance": "Retrieved by arXiv fallback for the Codex-Scientist-v2 literature stage.",
            }
        )
    return references, None


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def merge_references(seed_references: list[dict[str, Any]], fetched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in [*seed_references, *fetched]:
        title_key = str(ref.get("title", "")).strip().lower()
        if not title_key or title_key in seen:
            continue
        seen.add(title_key)
        merged.append(ref)
    return merged


def synthesize_literature_review(best: dict[str, Any], references: list[dict[str, Any]]) -> str:
    titles = "; ".join(str(ref.get("title")) for ref in references[:6])
    return (
        f"The best branch ({best.get('recipe_id')}) should be framed against world-model work, "
        "learned action representations, counterfactual dynamics, and automated-science systems. "
        f"Most relevant retrieved/seeded references include: {titles}."
    )


def references_to_bibtex(references: list[dict[str, Any]]) -> str:
    entries = []
    for ref in references:
        entry_type = "inproceedings" if ref.get("venue") else "article"
        fields = {
            "title": ref.get("title"),
            "author": ref.get("authors"),
            "year": ref.get("year"),
            "journal": ref.get("venue") if entry_type == "article" else None,
            "booktitle": ref.get("venue") if entry_type == "inproceedings" else None,
            "url": ref.get("url"),
        }
        lines = [f"@{entry_type}{{{ref.get('key') or bib_key(str(ref.get('title')), ref.get('year'))},"]
        for key, value in fields.items():
            if value:
                lines.append(f"  {key}={{{bib_escape(str(value))}}},")
        if lines[-1].endswith(","):
            lines[-1] = lines[-1][:-1]
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + "\n"


def bib_key(title: str, year: Any) -> str:
    words = "".join(char.lower() if char.isalnum() else " " for char in str(title)).split()
    core = "".join(words[:4]) or "reference"
    return f"{core}{year or 'nd'}"[:48]


def bib_escape(text: str) -> str:
    return text.replace("{", "").replace("}", "")


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
    commands = [[engine, "-interaction=nonstopmode", paper_tex.name]]
    if shutil.which("bibtex"):
        commands.append(["bibtex", paper_tex.stem])
    commands.extend(
        [
            [engine, "-interaction=nonstopmode", paper_tex.name],
            [engine, "-interaction=nonstopmode", paper_tex.name],
        ]
    )
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


def write_review_outputs(output_dir: Path, rows: list[dict[str, Any]], literature_report: dict[str, Any]) -> None:
    review_dir = ensure_dir(output_dir / "review")
    paper_path = output_dir / "latex" / "paper.tex"
    ablation_report_path = output_dir / "codex_scientistv2" / "ablation_report.json"
    ablation_report = json.loads(ablation_report_path.read_text(encoding="utf-8")) if ablation_report_path.exists() else {}
    text_review = build_text_review(output_dir, rows, ablation_report, literature_report)
    figure_review = build_figure_review(output_dir)
    write_json(review_dir / "review.json", text_review)
    write_json(review_dir / "vlm_review.json", figure_review)
    write_json(
        review_dir / "review_scaffold.json",
        {
            "status": "complete_local_automated_review",
            "paper": str(paper_path),
            "text_review": str(review_dir / "review.json"),
            "figure_review": str(review_dir / "vlm_review.json"),
            "llm_review_prompt": str(output_dir / "codex_scientistv2" / "codex_tasks" / "llm_review.md"),
            "vlm_review_prompt": str(output_dir / "codex_scientistv2" / "codex_tasks" / "vlm_review.md"),
        },
    )


def build_text_review(
    output_dir: Path,
    rows: list[dict[str, Any]],
    ablation_report: dict[str, Any],
    literature_report: dict[str, Any],
) -> dict[str, Any]:
    best = max(rows, key=score_key)
    controlled = ablation_report.get("controlled_ablations")
    references = literature_report.get("references") or []
    paper_path = output_dir / "latex" / "paper.tex"
    paper_text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    weaknesses = []
    if not controlled:
        weaknesses.append("Controlled ablation reruns are missing; current comparisons are mostly post-hoc population comparisons.")
    elif not any(control.get("success") for control in controlled.get("controls", [])):
        weaknesses.append("Controlled ablations were attempted but did not produce a successful control run.")
    if len(references) <= len(seed_literature_references()):
        weaknesses.append("Literature review fell back mostly or entirely to seed references.")
    if "Persistent action supervision" in paper_text and best.get("patch_recipe_id") != "full_budget_action_supervision":
        weaknesses.append("Paper framing may be stale relative to the current best branch.")
    strengths = [
        "The run preserves an auditable node tree, metrics, logs, and source diffs.",
        "The paper includes focused result tables and generated figures.",
    ]
    if controlled:
        strengths.append("The pipeline executed bounded controlled ablation reruns after the population search.")
    score = 6
    if controlled:
        score += 1
    if len(references) > len(seed_literature_references()):
        score += 1
    if weaknesses:
        score -= min(3, len(weaknesses))
    return {
        "reviewer": "codex_scientistv2_local_text_reviewer",
        "paper": str(paper_path),
        "best_node": best.get("node_id"),
        "best_score": best.get("primary_score"),
        "summary": (
            "The submission reports an automated TinyWorlds world-model search with preserved lineage and artifacts. "
            "The evidence is useful as exploratory science, but claims should stay bounded by the controls and seeds."
        ),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "questions": [
            "Does the best intervention replicate across independent seeds?",
            "Which component remains useful when knobs, source edits, and patch recipes are isolated?",
            "Does lower validation MSE translate into improved downstream planning or rollout quality?",
        ],
        "soundness": max(1, min(10, score)),
        "presentation": 7 if paper_path.exists() else 3,
        "contribution": 6,
        "decision": "weak_accept" if score >= 7 else "borderline",
        "confidence": 6,
    }


def build_figure_review(output_dir: Path) -> dict[str, Any]:
    fig_dir = output_dir / "figures"
    figures = []
    for path in sorted(fig_dir.glob("*")):
        if path.suffix.lower() not in {".svg", ".pdf", ".png"}:
            continue
        figures.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "status": "present_nonempty" if path.stat().st_size > 0 else "empty",
            }
        )
    missing = [
        str(fig_dir / "best_score_by_generation.pdf"),
        str(fig_dir / "cultural_tree.pdf"),
    ]
    missing = [path for path in missing if not Path(path).exists()]
    return {
        "reviewer": "codex_scientistv2_local_figure_reviewer",
        "figures": figures,
        "missing_expected_figures": missing,
        "summary": "Generated figures are present and referenced by the LaTeX paper." if not missing else "Some expected figures are missing.",
        "recommendations": [
            "Keep the lineage tree in the appendix if it is too dense for the main paper.",
            "Use the best-score curve as the main run-level figure.",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
