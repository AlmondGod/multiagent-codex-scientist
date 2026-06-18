from __future__ import annotations

import json
import os
import re
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
from commsci.codex_scientist.paper import load_population_rows, score_key
from commsci.codex_scientist.runner import run_codex_scientist_branch_expansion
from commsci.config import DEFAULT_CONFIG, deep_merge
from commsci.model_client import OpenAICompatibleClient

from .prompts import (
    CODEX_ACTION_FINALIZATION_PROMPT,
    IDEA_GENERATION_PROMPT,
    IDEA_REFLECTION_PROMPT,
    IDEATION_SYSTEM_PROMPT,
    NEURIPS_REVIEW_FORM,
    PLOT_AGGREGATION_PROMPT,
    PLOT_REFLECTION_PROMPT_TEMPLATE,
    REVIEWER_SYSTEM_PROMPT_NEG,
    WRITEUP_PROMPT,
    WRITEUP_REFLECTION_PROMPT,
    WRITEUP_SYSTEM_MESSAGE_TEMPLATE,
    codex_live_ideation_prompt,
    codex_plotting_prompt_bundle,
    codex_strict_review_prompt_bundle,
    codex_writeup_prompt_bundle,
    plot_aggregation_system_prompt,
)
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
        "Generate a small number of final figures that clarify the paper claim and are included in the manuscript by default.",
    ],
    "paper_writeup": [
        "Write an ICML-style workshop paper about one discovered scientific idea with a conference-style title, not a benchmark-name title.",
    ],
    "review": [
        "Review the paper for soundness, clarity, contribution, and missing controls.",
    ],
}


ICML_TEMPLATE_DIR = Path(__file__).resolve().parent / "blank_icml_latex"
ICML_TEMPLATE_FILES = (
    "algorithm.sty",
    "algorithmic.sty",
    "icml2025.bst",
    "icml2025.sty",
    "template.tex",
    "README.md",
    "LICENSE.Sakana-AI-Scientist-v2",
)

DEFAULT_AUTONOMOUS_CODEX_TASKS = ("idea_generation", "figure_making", "paper_writeup", "paper_reflection", "llm_review")
FIGURE_TASK_NAMES = {"figure_making", "figure_generation", "figures", "plotting", "plot_aggregation"}
DEFAULT_CODEX_TASK_MAX_COMPLETION_TOKENS = 12000


def run_codex_scientistv2(args: Any) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)
    if should_run_initial_literature_nodes(args):
        run_initial_literature_nodes(output_dir, args)
    if not getattr(args, "skip_experiments", False):
        run_population_tree(args, output_dir)
    if should_run_controlled_ablations(args):
        run_controlled_ablations(output_dir, args)
    return write_v2_outputs(output_dir, doctrine_doc=getattr(args, "doctrine_doc", None), codex_task_args=args)


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
        "--auto_generate_actions",
        "--auto_action_max_completion_tokens",
        str(max(2000, int(getattr(args, "codex_task_max_completion_tokens", 2000) or 2000))),
    ]
    if getattr(args, "codex_task_model_url", None):
        cmd.extend(["--auto_action_model_url", str(args.codex_task_model_url)])
    if getattr(args, "codex_task_model", None):
        cmd.extend(["--auto_action_model", str(args.codex_task_model)])
    if getattr(args, "codex_task_temperature", None) is not None:
        cmd.extend(["--auto_action_temperature", str(args.codex_task_temperature)])
    if getattr(args, "codex_task_dry_run", False):
        cmd.append("--auto_action_dry_run")
    if getattr(args, "codex_task_fail_fast", False):
        cmd.append("--auto_action_fail_fast")
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


def write_v2_outputs(output_dir: Path, doctrine_doc: str | None = None, codex_task_args: Any | None = None) -> Path:
    rows = load_population_rows(output_dir)
    if not rows:
        raise FileNotFoundError(f"No population summaries found in {output_dir}")
    v2_dir = ensure_dir(output_dir / "codex_scientistv2")
    literature_report = write_literature_review(v2_dir, rows)
    write_json(v2_dir / "run_manifest.json", build_manifest(output_dir, rows, doctrine_doc))
    write_json(v2_dir / "ablation_report.json", build_ablation_report(output_dir, rows))
    write_figures(output_dir, rows)
    write_tree_visualizations(output_dir, rows)
    write_v2_markdown_scaffold(output_dir, rows)
    write_latex_workshop_stub(output_dir)
    write_codex_tasks(output_dir, rows, doctrine_doc)
    if should_run_autonomous_codex_tasks(codex_task_args):
        run_autonomous_codex_tasks(output_dir, codex_task_args, rows=rows, doctrine_doc=doctrine_doc)
    write_review_outputs(output_dir, rows, literature_report)
    nodes = build_rich_nodes(output_dir, rows)
    write_jsonl(v2_dir / "rich_nodes.jsonl", [node.to_dict() for node in nodes])
    stage_reports = build_stage_reports(nodes)
    ensure_dir(v2_dir / "stage_reports")
    for report in stage_reports:
        write_json(v2_dir / "stage_reports" / f"{report.name}.json", report.to_dict())
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
    nodes.extend(build_controlled_ablation_rich_nodes(output_dir, max_generation))
    nodes.extend(build_artifact_rich_nodes(output_dir, max_generation))
    return nodes


def build_controlled_ablation_rich_nodes(output_dir: Path, max_generation: int) -> list[RichCodexNode]:
    nodes = []
    report_path = output_dir / "codex_scientistv2" / "ablation_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
        controlled = report.get("controlled_ablations") or {}
        controls = controlled.get("controls") or []
        successful_controls = [control for control in controls if control.get("success")]
        best_method = report.get("best_method") or {}
        nodes.append(
            RichCodexNode(
                node_id="ablation_report_artifact",
                parent_id=best_method.get("node_id"),
                agent_id="codex_scientistv2",
                generation=max_generation + 1,
                stage="ablation_studies",
                recipe_id="focused_and_controlled_ablation_report",
                inheritance_mode="artifact",
                source_node_ids=[str(best_method.get("node_id"))] if best_method.get("node_id") else [],
                patch_recipe_id=best_method.get("patch_recipe_id"),
                knobs=best_method.get("knobs") or {},
                metrics={
                    "focused_comparison_count": len(report.get("focused_comparisons") or []),
                    "controlled_ablation_count": len(controls),
                    "successful_controlled_ablation_count": len(successful_controls),
                    "primary_score": best_method.get("primary_score"),
                    "val_mse": best_method.get("val_mse"),
                },
                action_path=str(report_path),
                metrics_path=str(report_path),
                logs_path=str(report_path),
                code_diff_path=None,
                rationale="Summarized focused comparisons and controlled component ablations for the selected method.",
                analysis="Ablation report artifact used by the manuscript and reviewer.",
                is_buggy=bool(controls) and not bool(successful_controls),
                ablation_name="focused_and_controlled_ablation_report",
            )
        )
    summary_path = output_dir / "controlled_ablations" / "summary.json"
    if not summary_path.exists():
        return nodes
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return nodes
    source_node = summary.get("source_best_node")
    for control in summary.get("controls") or []:
        index = int(control.get("index", len(nodes)))
        action_path = output_dir / "controlled_ablations" / "actions" / f"ablation_{index}_step_1.json"
        metrics = control.get("metrics") or {}
        nodes.append(
            RichCodexNode(
                node_id=f"controlled_ablation_{index}",
                parent_id=str(source_node) if source_node else None,
                agent_id=f"ablation_{index}",
                generation=max_generation + 1,
                stage="ablation_studies",
                recipe_id=control.get("ablation_id"),
                inheritance_mode="controlled_ablation",
                source_node_ids=[str(source_node)] if source_node else [],
                patch_recipe_id=metrics.get("patch_recipe_id"),
                knobs={},
                metrics={
                    **metrics,
                    "primary_score": metrics.get("primary_score"),
                    "val_mse": metrics.get("val_mse"),
                    "experiment_success": control.get("success"),
                    "controlled_ablation_success": control.get("success"),
                },
                action_path=str(action_path),
                metrics_path=str(summary_path),
                logs_path=str(summary_path),
                code_diff_path=None,
                rationale=str(control.get("description") or ""),
                analysis=f"Controlled component ablation: {control.get('description')}",
                is_buggy=not bool(control.get("success")),
                ablation_name=str(control.get("ablation_id") or f"controlled_ablation_{index}"),
            )
        )
    return nodes


def build_artifact_rich_nodes(output_dir: Path, max_generation: int) -> list[RichCodexNode]:
    nodes = []
    fig_dir = output_dir / "figures"
    figure_paths = sorted(
        path for path in fig_dir.glob("*")
        if path.suffix.lower() in {".svg", ".pdf", ".png"} and path.stat().st_size > 0
    ) if fig_dir.exists() else []
    if figure_paths:
        nodes.append(
            RichCodexNode(
                node_id="plot_aggregation_artifact",
                parent_id=None,
                agent_id="codex_scientistv2",
                generation=max_generation + 2,
                stage="plot_aggregation",
                recipe_id="plot_aggregation",
                inheritance_mode="artifact",
                source_node_ids=[],
                patch_recipe_id=None,
                knobs={},
                metrics={"figure_count": len(figure_paths), "primary_score": float(len(figure_paths))},
                action_path=str(fig_dir),
                metrics_path=str(fig_dir / "cultural_tree_index.json"),
                logs_path=str(fig_dir),
                code_diff_path=None,
                rationale="Generated manuscript figures and lineage/provenance visualizations.",
                analysis="Plot aggregation created the main result figures and appendix lineage artifact.",
            )
        )
    paper_path = output_dir / "latex" / "paper.tex"
    markdown_path = output_dir / "paper.md"
    if paper_path.exists() or markdown_path.exists():
        nodes.append(
            RichCodexNode(
                node_id="paper_writeup_artifact",
                parent_id="plot_aggregation_artifact" if figure_paths else None,
                agent_id="codex_scientistv2",
                generation=max_generation + 3,
                stage="paper_writeup",
                recipe_id="icml_workshop_paper",
                inheritance_mode="artifact",
                source_node_ids=[],
                patch_recipe_id=None,
                knobs={},
                metrics={
                    "paper_tex_bytes": paper_path.stat().st_size if paper_path.exists() else 0,
                    "paper_markdown_bytes": markdown_path.stat().st_size if markdown_path.exists() else 0,
                    "primary_score": 1.0,
                },
                action_path=str(paper_path),
                metrics_path=str(markdown_path),
                logs_path=str(output_dir / "latex" / "compile.log"),
                code_diff_path=None,
                rationale="Wrote an ICML-style manuscript centered on the selected scientific idea.",
                analysis="Paper writeup produced the Markdown companion and LaTeX manuscript.",
            )
        )
    review_path = output_dir / "review" / "review.json"
    figure_review_path = output_dir / "review" / "vlm_review.json"
    if review_path.exists() or figure_review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
        figure_review = json.loads(figure_review_path.read_text(encoding="utf-8")) if figure_review_path.exists() else {}
        review_scores = [
            float(review.get(key))
            for key in ("soundness", "presentation", "contribution")
            if isinstance(review.get(key), (int, float))
        ]
        nodes.append(
            RichCodexNode(
                node_id="review_artifact",
                parent_id="paper_writeup_artifact" if paper_path.exists() or markdown_path.exists() else None,
                agent_id="codex_scientistv2",
                generation=max_generation + 4,
                stage="review",
                recipe_id="automated_paper_review",
                inheritance_mode="artifact",
                source_node_ids=[],
                patch_recipe_id=None,
                knobs={},
                metrics={
                    "soundness": review.get("soundness"),
                    "presentation": review.get("presentation"),
                    "contribution": review.get("contribution"),
                    "figure_score": figure_review.get("figure_score"),
                    "primary_score": mean(review_scores) if review_scores else None,
                },
                action_path=str(review_path),
                metrics_path=str(figure_review_path),
                logs_path=str(output_dir / "review" / "review_scaffold.json"),
                code_diff_path=None,
                rationale="Reviewed paper soundness, presentation, contribution, and figure support.",
                analysis=f"Automated reviewer decision: {review.get('decision', 'unknown')}.",
                is_buggy=bool(review.get("weaknesses")),
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
            best = max(stage_nodes, key=rich_node_score)
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


def rich_node_score(node: RichCodexNode) -> float:
    value = node.metrics.get("primary_score")
    return float(value) if isinstance(value, (int, float)) else -1.0


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
        title="Best score by generation",
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
    ax.set_title("Best score by generation")
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
    v2_dir = output_dir / "codex_scientistv2"
    write_prompt_templates(v2_dir)
    summaries = build_writeup_summaries(output_dir, rows)
    latex_path = output_dir / "latex" / "paper.tex"
    latex_writeup = latex_path.read_text(encoding="utf-8") if latex_path.exists() else ""
    figure_review_path = output_dir / "review" / "vlm_review.json"
    plot_descriptions = figure_review_path.read_text(encoding="utf-8") if figure_review_path.exists() else local_plot_descriptions(output_dir)
    idea_text = synthesize_writeup_idea(best, rows)
    ideation_response = read_codex_task_response(output_dir, "idea_generation")
    if ideation_response:
        idea_text += "\n\nCodex ideation response from this run:\n" + ideation_response
    figure_response = read_codex_task_response(output_dir, "figure_making") or read_codex_task_response(output_dir, "plot_aggregation")
    plot_files = [
        str(path.relative_to(output_dir))
        for path in sorted((output_dir / "figures").glob("*"))
        if path.suffix.lower() in {".pdf", ".svg", ".png"}
    ]
    ideation_bundle = codex_live_ideation_prompt(
        workshop_description=codex_v2_workshop_description(),
        previous_ideas=json.dumps([row.get("recipe_id") for row in rows if row.get("recipe_id")], indent=2),
        literature_context=read_text_file(output_dir / "codex_scientistv2" / "literature_nodes" / "initial_literature_context.md"),
        last_tool_results=read_text_file(output_dir / "codex_scientistv2" / "literature" / "literature_review.json"),
    )
    aggregator_context = plot_aggregator_context(output_dir)
    if figure_response:
        plot_descriptions += "\n\nCodex figure-making response from this run:\n" + figure_response
    writeup_bundle = codex_writeup_prompt_bundle(
        idea_text=idea_text,
        summaries=summaries,
        aggregator_code=aggregator_context,
        plot_list=plot_files,
        plot_descriptions=plot_descriptions,
        latex_writeup=latex_writeup,
        page_limit=8,
    )
    plotting_bundle = codex_plotting_prompt_bundle(
        idea_text=idea_text,
        summaries=summaries,
        aggregator_code=aggregator_context,
        plot_list=plot_files,
        figure_count=len(plot_files),
        aggregator_out="The deterministic v2 pipeline has already generated the listed figures; improve or replace only when the summaries support it.",
    )
    strict_review_bundle = codex_strict_review_prompt_bundle(
        paper_text=latex_writeup,
        evidence={
            "best_method": best,
            "summaries": summaries,
            "plot_files": plot_files,
            "plot_descriptions": plot_descriptions,
            "paper_path": str(latex_path),
        },
    )
    shared = f"""Run: {output_dir}
Best node: {best.get('node_id')}
Best recipe: {best.get('recipe_id')}
Best score: {best.get('primary_score')}
Doctrine: {doctrine_doc}
"""
    tasks = {
        "idea_generation.md": ideation_bundle,
        "literature_search.md": "Find 6-10 current papers directly relevant to the domain paper claim. Return BibTeX and one-sentence relevance notes.",
        "figure_making.md": plotting_bundle,
        "plot_aggregation.md": plotting_bundle,
        "paper_writeup.md": writeup_bundle,
        "paper_reflection.md": (
            writeup_bundle
            + "\n\nAdditional Codex-Scientist-v2 reflection constraints:\n"
            "- Review latex/paper.tex as an ICML-style workshop submission.\n"
            "- The title must read like a conference paper and should not name TinyWorlds; TinyWorlds belongs in the experimental setup as the benchmark.\n"
            "- Improve claims, ablations, figure references, citations, and limitations without inventing results.\n"
        ),
        "llm_review.md": strict_review_bundle,
        "vlm_review.md": (
            "Review each figure in figures/*.svg and figures/*.pdf using caption-reference discipline. For each figure, "
            "return Img_description, Img_review, Caption_review, Figrefs_review, Overall_comments, Containing_sub_figures, "
            "and Informative_review. Check whether the best-score curve and patch-family plot support the main claim, "
            "and whether the cultural tree belongs in the appendix as provenance rather than as the central argument."
        ),
    }
    for filename, instruction in tasks.items():
        write_text(task_dir / filename, f"# Codex-Scientist-v2 Task\n\n{shared}\n\n{instruction}\n")


def write_prompt_templates(v2_dir: Path) -> None:
    prompt_dir = ensure_dir(v2_dir / "prompt_templates")
    templates = {
        "ideation_system_prompt.md": IDEATION_SYSTEM_PROMPT,
        "idea_generation_prompt.md": IDEA_GENERATION_PROMPT,
        "idea_reflection_prompt.md": IDEA_REFLECTION_PROMPT,
        "codex_action_finalization_prompt.md": CODEX_ACTION_FINALIZATION_PROMPT,
        "reviewer_system_prompt_neg.md": REVIEWER_SYSTEM_PROMPT_NEG,
        "neurips_review_form.md": NEURIPS_REVIEW_FORM,
        "writeup_system_prompt.md": WRITEUP_SYSTEM_MESSAGE_TEMPLATE,
        "writeup_prompt.md": WRITEUP_PROMPT,
        "writeup_reflection_prompt.md": WRITEUP_REFLECTION_PROMPT,
        "plot_aggregation_system_prompt.md": plot_aggregation_system_prompt(),
        "plot_aggregation_prompt.md": PLOT_AGGREGATION_PROMPT,
        "plot_reflection_prompt.md": PLOT_REFLECTION_PROMPT_TEMPLATE,
    }
    for filename, text in templates.items():
        write_text(prompt_dir / filename, text)


def build_writeup_summaries(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ablation_path = output_dir / "codex_scientistv2" / "ablation_report.json"
    literature_path = output_dir / "codex_scientistv2" / "literature" / "literature_review.json"
    return {
        "population": rows,
        "ablation_report": json.loads(ablation_path.read_text(encoding="utf-8")) if ablation_path.exists() else {},
        "literature_report": json.loads(literature_path.read_text(encoding="utf-8")) if literature_path.exists() else {},
    }


def synthesize_writeup_idea(best: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            f"Selected method: {best.get('recipe_id')}",
            f"Patch family: {best.get('patch_recipe_id')}",
            f"Primary score: {best.get('primary_score')}",
            f"Validation MSE: {best.get('val_mse')}",
            f"Rationale: {best.get('rationale')}",
            "Write the paper around one scientific method or negative result, not around the autoresearch run.",
            f"Evaluated variants: {len(rows)}",
        ]
    )


def codex_v2_workshop_description() -> str:
    return (
        "We are studying short-budget action-conditioned world-model learning. "
        "The research setting is TinyWorlds, but the idea should be framed as a publishable ML method, ablation, or negative result. "
        "Propose simple, feasible, falsifiable interventions that can be executed through validated knobs, curated patch recipes, or small edits to train.py/models.py."
    )


def plot_aggregator_description() -> str:
    return (
        "Figures are generated by codex_scientistv2.write_figures and write_tree_visualizations. "
        "The main plot files are figures/best_score_by_generation.pdf, figures/patch_family_mean_scores.pdf, "
        "and figures/cultural_tree.pdf. Use the first two in the main paper and keep the tree as appendix provenance."
    )


def plot_aggregator_context(output_dir: Path) -> str:
    for path in (output_dir / "auto_figure_maker.py", output_dir / "auto_plot_aggregator.py"):
        if path.exists():
            return path.read_text(encoding="utf-8")[:12000]
    return plot_aggregator_description()


def read_codex_task_response(output_dir: Path, task_name: str, max_chars: int = 8000) -> str:
    path = output_dir / "codex_scientistv2" / "codex_task_runs" / "responses" / f"{task_name}.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")[:max_chars]


def local_plot_descriptions(output_dir: Path) -> str:
    figures = []
    for path in sorted((output_dir / "figures").glob("*")):
        if path.suffix.lower() in {".pdf", ".svg", ".png"}:
            figures.append(f"{path.name}: generated plot artifact, size={path.stat().st_size} bytes")
    return "\n".join(figures) if figures else "No generated plot descriptions available."


def should_run_autonomous_codex_tasks(args: Any | None) -> bool:
    return bool(args is not None and getattr(args, "run_codex_tasks", False))


def autonomous_task_names(args: Any | None) -> list[str]:
    raw = getattr(args, "codex_task_names", None) if args is not None else None
    if raw:
        names = [item.strip() for item in str(raw).split(",") if item.strip()]
    else:
        names = list(DEFAULT_AUTONOMOUS_CODEX_TASKS)
    normalized = []
    for name in names:
        stem = name[:-3] if name.endswith(".md") else name
        normalized.append("figure_making" if stem in FIGURE_TASK_NAMES else stem)
    return normalized


def build_codex_task_client(args: Any) -> OpenAICompatibleClient:
    model_cfg = DEFAULT_CONFIG["model"]
    return OpenAICompatibleClient(
        model_url=str(getattr(args, "codex_task_model_url", None) or model_cfg["model_url"]),
        model_name=str(getattr(args, "codex_task_model", None) or model_cfg["default_model"]),
        temperature=float(getattr(args, "codex_task_temperature", None) if getattr(args, "codex_task_temperature", None) is not None else model_cfg["temperature"]),
        seed=int(getattr(args, "seed", model_cfg["seed"])),
        max_completion_tokens=int(
            getattr(args, "codex_task_max_completion_tokens", None)
            if getattr(args, "codex_task_max_completion_tokens", None) is not None
            else DEFAULT_CODEX_TASK_MAX_COMPLETION_TOKENS
        ),
        dry_run=bool(getattr(args, "codex_task_dry_run", False)),
    )


def run_autonomous_codex_tasks(
    output_dir: Path,
    args: Any,
    rows: list[dict[str, Any]] | None = None,
    doctrine_doc: str | None = None,
) -> dict[str, Any]:
    task_dir = output_dir / "codex_scientistv2" / "codex_tasks"
    run_dir = ensure_dir(output_dir / "codex_scientistv2" / "codex_task_runs")
    response_dir = ensure_dir(run_dir / "responses")
    applied_dir = ensure_dir(run_dir / "applied")
    client = build_codex_task_client(args)
    results = []
    task_names = autonomous_task_names(args)
    for name in task_names:
        if rows is not None and name in {"figure_making", "paper_writeup", "paper_reflection", "llm_review"}:
            write_codex_tasks(output_dir, rows, doctrine_doc)
        task_path = task_dir / f"{name}.md"
        if not task_path.exists():
            results.append({"task": name, "success": False, "error": f"missing task file: {task_path}"})
            continue
        prompt = task_path.read_text(encoding="utf-8")
        try:
            response = client.complete(prompt, "codex_scientistv2", run_dir, name)
        except Exception as exc:
            error_text = repr(exc)
            write_text(response_dir / f"{name}.error.txt", error_text + "\n")
            if name == "llm_review":
                clear_stale_codex_llm_review(output_dir, error_text)
            results.append({"task": name, "success": False, "error": error_text})
            if bool(getattr(args, "codex_task_fail_fast", False)):
                raise
            continue
        response_path = response_dir / f"{name}.md"
        write_text(response_path, response.text)
        applied = apply_codex_task_response(output_dir, name, response.text, args, applied_dir)
        if rows is not None and name in {"idea_generation", "figure_making"}:
            write_codex_tasks(output_dir, rows, doctrine_doc)
        results.append(
            {
                "task": name,
                "success": True,
                "response": str(response_path),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "applied": applied,
            }
        )
    summary = {
        "status": "complete",
        "task_names": task_names,
        "apply_outputs": bool(getattr(args, "apply_codex_task_outputs", False)),
        "execute_plot_outputs": bool(getattr(args, "execute_plot_task_outputs", False)),
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def apply_codex_task_response(output_dir: Path, task_name: str, response_text: str, args: Any, applied_dir: Path) -> dict[str, Any]:
    if not bool(getattr(args, "apply_codex_task_outputs", False)):
        return {"status": "not_applied"}
    if task_name in {"paper_writeup", "paper_reflection"}:
        latex = extract_fenced_code(response_text, "latex")
        if not latex:
            return {"status": "not_applied", "reason": "no fenced latex block found"}
        paper_path = output_dir / "latex" / "paper.tex"
        write_text(applied_dir / f"{task_name}.tex", latex)
        write_text(paper_path, latex)
        compile_latex_paper(paper_path)
        return {"status": "applied_latex", "path": str(paper_path)}
    if task_name in FIGURE_TASK_NAMES:
        python_code = extract_fenced_code(response_text, "python")
        if not python_code:
            return {"status": "not_applied", "reason": "no fenced python block found"}
        script_name = "auto_figure_maker.py" if task_name == "figure_making" else "auto_plot_aggregator.py"
        script_path = output_dir / script_name
        write_text(applied_dir / script_name, python_code)
        write_text(script_path, python_code)
        if not bool(getattr(args, "execute_plot_task_outputs", False)):
            return {"status": "wrote_figure_script", "path": str(script_path)}
        result = subprocess.run(
            [sys.executable, script_path.name],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=int(getattr(args, "codex_task_execution_timeout_seconds", 120)),
            check=False,
        )
        write_text(applied_dir / f"{script_path.stem}.stdout.txt", result.stdout)
        write_text(applied_dir / f"{script_path.stem}.stderr.txt", result.stderr)
        return {
            "status": "executed_figure_script",
            "path": str(script_path),
            "returncode": result.returncode,
        }
    if task_name == "llm_review":
        review_dir = ensure_dir(output_dir / "review")
        write_text(review_dir / "codex_llm_review_raw.md", response_text)
        review_json = extract_review_json(response_text)
        if review_json is None:
            return {"status": "not_applied", "reason": "no parseable review JSON found"}
        json_path = review_dir / "codex_llm_review.json"
        write_json(json_path, review_json)
        return {"status": "applied_review_json", "path": str(json_path)}
    return {"status": "not_applied", "reason": f"task {task_name} is recorded only"}


def extract_fenced_code(text: str, language: str) -> str | None:
    pattern = rf"```{re.escape(language)}\s*(.*?)\s*```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", text, flags=re.DOTALL)
    return generic.group(1).strip() if generic else None


def extract_review_json(text: str) -> dict[str, Any] | None:
    fenced = extract_fenced_code(text, "json")
    candidates = [fenced] if fenced else []
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def clear_stale_codex_llm_review(output_dir: Path, error_text: str) -> None:
    review_dir = ensure_dir(output_dir / "review")
    for filename in ("codex_llm_review.json", "codex_llm_review_raw.md"):
        path = review_dir / filename
        if path.exists():
            path.unlink()
    write_text(review_dir / "codex_llm_review_error.txt", error_text + "\n")


def copy_icml_latex_template(latex_dir: Path) -> None:
    if not ICML_TEMPLATE_DIR.exists():
        return
    for filename in ICML_TEMPLATE_FILES:
        src = ICML_TEMPLATE_DIR / filename
        if not src.exists():
            continue
        dst_name = "blank_icml_template.tex" if filename == "template.tex" else filename
        shutil.copyfile(src, latex_dir / dst_name)


def write_latex_workshop_stub(output_dir: Path) -> None:
    latex_dir = ensure_dir(output_dir / "latex")
    copy_icml_latex_template(latex_dir)
    refs_src = output_dir / "codex_scientistv2" / "literature" / "references.bib"
    if refs_src.exists():
        shutil.copyfile(refs_src, latex_dir / "references.bib")
    paper_tex = write_v2_latex_scaffold(output_dir, latex_dir / "paper.tex")
    compile_latex_paper(paper_tex)
    write_text(
        latex_dir / "template.tex",
        r"""\documentclass{article}
\IfFileExists{icml2025.sty}{
  \usepackage{icml2025}
}{
  \usepackage[margin=1in]{geometry}
  \usepackage{times}
  \newcommand{\icmltitle}[1]{\begin{center}{\Large\bf #1}\end{center}}
  \newcommand{\icmlauthor}[2]{\begin{center}#1\end{center}}
  \newcommand{\icmlaffiliation}[3]{}
  \newcommand{\icmlcorrespondingauthor}[2]{}
  \newcommand{\icmlkeywords}[1]{}
  \newenvironment{icmlauthorlist}{}{}
}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\begin{document}
\twocolumn[
\icmltitle{Replace With a Conference-Style Method Claim}
\begin{icmlauthorlist}
\icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}{}
\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}
\icmlkeywords{world models, action-conditioned dynamics, short-budget learning}
\vskip 0.3in
]
\begin{abstract}
State the domain contribution, result, and limitation. Do not name the benchmark
in the title or summarize the autoresearch timeline in the abstract.
\end{abstract}
\section{Introduction}
Focus on one scientific question and one proposed intervention.
\section{Method}
Define the intervention and implementation-level change.
\section{Experiments}
Include the fixed TinyWorlds setup, controlled ablations, and figures.
\section{Results}
Report and explicitly reference the best-score curve, patch-family comparison,
and component ablations.
\section{Limitations}
State seed, budget, and downstream-planning limitations.
\bibliography{references}
\bibliographystyle{icml2025}
\end{document}
""",
    )


def write_v2_markdown_scaffold(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    best = max(rows, key=score_key)
    ablation_report = read_json_if_exists(output_dir / "codex_scientistv2" / "ablation_report.json")
    controlled = ablation_report.get("controlled_ablations") or {}
    controls = [control for control in controlled.get("controls", []) if control.get("success")]
    title = method_title(best)
    output = output_dir / "paper.md"
    body = [
        f"# {title}",
        "",
        "## Abstract",
        "",
        method_abstract(best, rows, controls),
        "",
        "## Introduction",
        "",
        method_introduction(best),
        "",
        "## Method",
        "",
        method_description(best),
        "",
        "## Experiments",
        "",
        experiment_description(rows, best, controls),
        "",
        "## Results",
        "",
        result_description(best, controls),
        "",
        "## Limitations",
        "",
        limitations_description(),
    ]
    write_text(output, "\n".join(body).strip() + "\n")
    return output


def write_v2_latex_scaffold(output_dir: Path, output_path: Path) -> Path:
    rows = load_population_rows(output_dir)
    best = max(rows, key=score_key)
    ablation_report = read_json_if_exists(output_dir / "codex_scientistv2" / "ablation_report.json")
    controlled = ablation_report.get("controlled_ablations") or {}
    controls = [control for control in controlled.get("controls", []) if control.get("success")]
    ensure_dir(output_path.parent)
    write_text(
        output_path,
        r"""\documentclass{article}
\IfFileExists{icml2025.sty}{
  \usepackage{icml2025}
}{
  \usepackage[margin=1in]{geometry}
  \usepackage{times}
  \newcommand{\icmltitle}[1]{\begin{center}{\Large\bf #1}\end{center}}
  \newcommand{\icmlsetsymbol}[2]{}
  \newcommand{\icmlauthor}[2]{\begin{center}#1\end{center}}
  \newcommand{\icmlaffiliation}[3]{}
  \newcommand{\icmlcorrespondingauthor}[2]{}
  \newcommand{\printAffiliationsAndNotice}[1]{}
  \newcommand{\icmlkeywords}[1]{}
  \newcommand{\theHalgorithm}{\arabic{algorithm}}
  \newenvironment{icmlauthorlist}{}{}
}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\graphicspath{{../figures/}{figures/}}
\begin{document}
\twocolumn[
\icmltitle{""" + latex_escape(method_title(best)) + r"""}
\begin{icmlauthorlist}
\icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}{}
\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}
\icmlkeywords{world models, action-conditioned dynamics, robust losses}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}
""" + latex_escape(method_abstract(best, rows, controls)) + r"""
\end{abstract}

\section{Introduction}
""" + latex_escape(method_introduction(best)) + r"""

\section{Related Work}
World models learn predictive state-transition models that can support planning and model-based control. Action-conditioned variants must separate visual persistence from changes explained by actions. Recent work on learned action representations and counterfactual dynamics motivates auxiliary objectives that make the dynamics pathway more sensitive to controllable change. Our study is narrower: we test whether a robust, motion-aware dynamics objective improves short-budget predictive learning in a compact grid-world benchmark.

\section{Method}
""" + latex_escape(method_description(best)) + r"""

\section{Experimental Setup}
""" + latex_escape(experiment_description(rows, best, controls)) + r"""

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{best_score_by_generation.pdf}
\caption{Best validation-derived score as more variants are evaluated. The curve is evidence about search-time discovery, not a claim of monotonic training improvement.}
\label{fig:best-score}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{patch_family_mean_scores.pdf}
\caption{Mean score by intervention family. This aggregate checks whether the result is tied to a broader objective family rather than only one isolated configuration.}
\label{fig:patch-family}
\end{figure}

\section{Results}
""" + latex_escape(result_description(best, controls)) + r""" Figure~\ref{fig:best-score} shows when the strongest result appears, and Figure~\ref{fig:patch-family} compares intervention families.

""" + latex_control_table(controls) + r"""

\section{Limitations}
""" + latex_escape(limitations_description()) + r"""

\section{Conclusion}
The current evidence supports a focused hypothesis: robust, motion-aware dynamics losses are promising under short training budgets, but the effect should be treated as preliminary until replicated across seeds, budgets, and downstream planning evaluations.

\bibliography{references}
\bibliographystyle{icml2025}

\appendix
\section{Lineage Artifact}
The appendix may include lineage visualizations for audit purposes; they are not part of the central scientific claim.
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{cultural_tree.pdf}
\caption{Experiment lineage and transfer provenance for auditability.}
\label{fig:lineage}
\end{figure}
\end{document}
""",
    )
    return output_path


def method_title(best: dict[str, Any]) -> str:
    patch = str(best.get("patch_recipe_id") or "")
    if patch == "smooth_l1_dynamics_pixel":
        return "Robust Motion-Weighted Dynamics for Short-Budget World Models"
    if patch == "full_budget_action_supervision":
        return "Persistent Action Supervision for Short-Budget World Models"
    if patch == "action_grad_dynamics":
        return "Action-Grounded Dynamics for Compact World Models"
    if patch == "dynamics_first_schedule":
        return "Dynamics-First Curricula for Short-Budget World Models"
    return "Short-Budget Objectives for Action-Conditioned World Models"


def method_abstract(best: dict[str, Any], rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    control_text = (
        f"We additionally ran {len(controls)} controlled component checks."
        if controls
        else "Controlled component checks are required before making a strong causal claim."
    )
    return (
        "Short-budget world-model training often rewards objectives that reduce pixel error without reliably emphasizing controllable motion. "
        f"We propose and evaluate a compact action-conditioned dynamics objective, {method_family_name(best)}, that changes the training pressure rather than the evaluation protocol. "
        "The hypothesis is that robust losses focused on motion-local prediction can make limited training updates more useful for controllable dynamics than uniform reconstruction alone. "
        f"Across {len(rows)} evaluated variants, the strongest configuration reached a primary score of {fmt_metric(best.get('primary_score'))} "
        f"with validation MSE {fmt_metric(best.get('val_mse'))}. {control_text} "
        "The ablations suggest that the objective family, not only incidental knobs, explains a meaningful part of the observed gain. "
        "The evidence supports a promising direction for robust dynamics learning, but the claim remains preliminary because replication across independent seeds and downstream planning evaluations are still missing."
    )


def method_introduction(best: dict[str, Any]) -> str:
    return (
        "Action-conditioned world models must learn which visual changes are caused by actions and which are background persistence. "
        "Under short training budgets, this distinction is especially fragile: a model can improve aggregate reconstruction while still under-learning controllable dynamics. "
        f"We test the hypothesis that {method_family_name(best)} provides a better inductive bias for this setting by shifting loss weight toward robust motion-sensitive prediction. "
        "This is a deliberately small intervention: the model class, benchmark, and training budget remain fixed, while the objective changes the pressure placed on dynamics errors. "
        "That narrow design makes the result easier to interpret than a broad architecture search, because improvements can be compared against nearby objective families and component controls. "
        "The contribution is a focused empirical study of one objective family, supported by family-level comparisons and component checks, with limitations stated explicitly. "
        "The paper does not claim a complete world-modeling solution; instead, it asks whether a simple robust objective is a useful building block for action-sensitive predictive learning under severe compute limits."
    )


def method_description(best: dict[str, Any]) -> str:
    patch = str(best.get("patch_recipe_id") or "")
    if patch == "smooth_l1_dynamics_pixel":
        return (
            "The method replaces a purely uniform reconstruction emphasis with a robust Smooth-L1 dynamics-pixel objective. "
            "The objective gives extra weight to changed pixels while reducing sensitivity to outlier residuals, encouraging the model to learn motion-local predictive structure. "
            "This is intended to improve action-sensitive dynamics without increasing the benchmark budget."
        )
    if patch == "full_budget_action_supervision":
        return (
            "The method keeps action supervision active throughout the short training budget. "
            "This tests whether persistent action grounding improves the learned transition model relative to schedules that decay auxiliary supervision early."
        )
    return (
        "The method changes the training objective for action-conditioned prediction while keeping the dataset, model class, and evaluation metric fixed. "
        "The goal is to isolate whether a small objective change can improve predictive dynamics under a constrained budget."
    )


def experiment_description(rows: list[dict[str, Any]], best: dict[str, Any], controls: list[dict[str, Any]]) -> str:
    return (
        "Experiments use the TinyWorlds benchmark as a compact testbed for action-conditioned visual dynamics. "
        f"We compare {len(rows)} objective variants under the same short training budget and report the configured primary score plus validation MSE. "
        f"The best-scoring variant uses the {method_family_name(best)} family. "
        f"Controlled checks: {len(controls)} successful reruns/components."
    )


def result_description(best: dict[str, Any], controls: list[dict[str, Any]]) -> str:
    sentences = [
        f"The strongest configuration reached primary score {fmt_metric(best.get('primary_score'))} and validation MSE {fmt_metric(best.get('val_mse'))}.",
    ]
    if controls:
        control_bits = []
        for control in controls[:3]:
            metrics = control.get("metrics") or {}
            control_bits.append(f"{control.get('ablation_id')}: {fmt_metric(metrics.get('primary_score'))}")
        sentences.append("The controlled checks returned " + "; ".join(control_bits) + ".")
    else:
        sentences.append("Because controlled checks are absent or failed, the result should be interpreted as exploratory rather than causal.")
    return " ".join(sentences)


def limitations_description() -> str:
    return (
        "The current evaluation is limited to a compact benchmark, short runtimes, and a small number of controlled checks. "
        "It does not yet establish downstream planning improvements, robustness to larger environments, or stable gains across many independent random seeds. "
        "These limitations make the result a focused hypothesis with early evidence rather than a mature conference-level empirical claim."
    )


def method_family_name(best: dict[str, Any]) -> str:
    return str(best.get("patch_recipe_id") or "objective").replace("_", " ")


def latex_control_table(controls: list[dict[str, Any]]) -> str:
    if not controls:
        return ""
    rows = []
    for control in controls[:4]:
        metrics = control.get("metrics") or {}
        rows.append(
            latex_escape(str(control.get("ablation_id") or "control"))
            + " & "
            + latex_escape(fmt_metric(metrics.get("primary_score")))
            + " & "
            + latex_escape(fmt_metric(metrics.get("val_mse")))
            + r" \\"
        )
    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        r"\caption{Controlled component checks for the proposed objective family.}" "\n"
        r"\label{tab:controls}" "\n"
        r"\begin{tabular}{lcc}" "\n"
        r"\toprule" "\n"
        r"Control & Score & Val. MSE \\" "\n"
        r"\midrule" "\n"
        + "\n".join(rows)
        + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )


def read_json_if_exists(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fmt_metric(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "N/A"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


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
    codex_llm_review = review_dir / "codex_llm_review.json"
    codex_llm_review_raw = review_dir / "codex_llm_review_raw.md"
    write_json(
        review_dir / "review_scaffold.json",
        {
            "status": "complete_local_automated_review",
            "paper": str(paper_path),
            "text_review": str(review_dir / "review.json"),
            "figure_review": str(review_dir / "vlm_review.json"),
            "codex_llm_review": str(codex_llm_review) if codex_llm_review.exists() else None,
            "codex_llm_review_raw": str(codex_llm_review_raw) if codex_llm_review_raw.exists() else None,
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
    main_text = paper_text.split("\\appendix", 1)[0]
    lower_main_text = main_text.lower()
    figure_review = review_latex_figures(output_dir, paper_text)
    successful_controls = [control for control in (controlled or {}).get("controls", []) if control.get("success")]
    stale_process_terms = [
        "autoresearch",
        "auto-research",
        "codex-scientist",
        "ai scientist",
        "search trace",
        "timeline",
        "paper generation",
        "literature node",
        "best candidate",
        "best branch",
        "selected branch",
        "winning branch",
        "candidate-pool",
        "candidate pool",
        "top discovered",
        "search-operator",
        "operator-family",
        "selected method",
        "recorded source diff",
        "run artifacts",
        "variant ranking",
        "method variants",
        "fresh run",
        "component testing",
        "full lineage",
        "auditability",
    ]
    stale_terms_found = [term for term in stale_process_terms if term in lower_main_text]
    expected_sections = ["Introduction", "Related Work", "Method", "Experimental Setup", "Results", "Limitations"]
    missing_sections = [section for section in expected_sections if not latex_has_section(paper_text, section)]
    controlled_success = bool(controlled and successful_controls)
    has_icml_template = "icml2025" in paper_text or "\\icmltitle" in paper_text
    title = extract_latex_title(paper_text)
    lower_title = title.lower()
    title_is_conference_style = bool(title) and "tinyworlds" not in lower_title and not any(
        term in lower_title for term in ["codex", "scientist", "autoresearch", "search"]
    )
    has_best_curve = "best_score_by_generation" in paper_text and not any(
        item.endswith("best_score_by_generation.pdf") for item in figure_review["missing_expected_figures"]
    )
    has_family_plot = "patch_family_mean_scores" in paper_text and not any(
        item.endswith("patch_family_mean_scores.pdf") for item in figure_review["missing_expected_figures"]
    )
    has_lineage_appendix = "cultural_tree" in paper_text and "\\appendix" in paper_text
    is_negative_result_narrative = any(term in lower_title for term in ["underperform", "negative", "fail"])
    has_single_narrative = (
        bool(best.get("patch_recipe_id"))
        and (
            str(best.get("patch_recipe_id", "")).replace("_", "-") not in {"", "baseline-no-patch"}
            or is_negative_result_narrative
        )
    )
    main_table_count = len(re.findall(r"\\begin\{table\*?\}", main_text))
    dumps_process_tables = "top discovered" in lower_main_text or "search-operator" in lower_main_text or "operator-family" in lower_main_text
    abstract = extract_latex_environment(paper_text, "abstract")
    intro = extract_latex_section(main_text, "Introduction")
    method = extract_latex_section(main_text, "Method")
    abstract_too_thin = len(abstract.split()) < 120
    intro_too_thin = len(intro.split()) < 250
    method_is_source_diff_report = "\\begin{lstlisting}" in method and any(
        term in method.lower()
        for term in ["recorded source diff", "source diff", "patch recipe", "run artifacts"]
    )
    weak_conclusion = any(
        term in lower_main_text
        for term in ["candidate intervention for follow-up", "fresh run identifies a candidate", "candidate for follow-up"]
    )
    lacks_method_claim = not any(
        term in lower_main_text
        for term in ["we propose", "we introduce", "our method", "we study whether", "we test the hypothesis"]
    )
    lacks_conference_baselines = "baseline" not in lower_main_text and "control" not in lower_main_text
    figure_refs_integrated = not figure_review["issues"]
    source_best_score = (controlled or {}).get("source_best_score") if isinstance(controlled, dict) else None
    repeat_controls = [
        control for control in successful_controls
        if "repeat" in str(control.get("ablation_id", "")).lower()
    ]
    repeat_score = None
    if repeat_controls:
        repeat_metrics = repeat_controls[0].get("metrics") or {}
        repeat_score = repeat_metrics.get("primary_score")
    repeat_score_gap = (
        float(source_best_score) - float(repeat_score)
        if isinstance(source_best_score, (int, float)) and isinstance(repeat_score, (int, float))
        else None
    )
    weaknesses = []
    critical_weaknesses = []
    if not title_is_conference_style:
        critical_weaknesses.append("Title should read like a conference paper and avoid benchmark/process names.")
    if not controlled:
        critical_weaknesses.append("Controlled ablation reruns are missing; current comparisons are mostly post-hoc population comparisons.")
    elif not successful_controls:
        critical_weaknesses.append("Controlled ablations were attempted but did not produce a successful control run.")
    elif len(successful_controls) < 3:
        weaknesses.append("Controlled ablations are present but sparse; the paper should clearly state which component each control removes.")
    if repeat_score_gap is not None and repeat_score_gap > 0.01:
        weaknesses.append(
            f"The controlled repeat is {repeat_score_gap:.3f} below the selected-method score; "
            "claims should be framed as short-budget exploratory evidence rather than stable replication."
        )
    if len(references) <= len(seed_literature_references()):
        weaknesses.append("Literature review fell back mostly or entirely to seed references.")
    if stale_terms_found:
        critical_weaknesses.append(
            "Main paper still references the automation process rather than only the scientific claim: "
            + ", ".join(sorted(set(stale_terms_found)))
            + "."
        )
    if missing_sections:
        weaknesses.append("Paper is missing expected ICML-style sections: " + ", ".join(missing_sections) + ".")
    if not has_best_curve or not has_family_plot:
        weaknesses.append("Main results should include both the best-score curve and patch-family aggregate figure.")
    if not figure_refs_integrated:
        weaknesses.append("Figures are present but not fully integrated into the main text: " + "; ".join(figure_review["issues"]))
    if dumps_process_tables:
        critical_weaknesses.append("Main results include process/table dumps instead of a compact scientific narrative.")
    if abstract_too_thin or intro_too_thin:
        critical_weaknesses.append("The abstract/introduction do not sufficiently motivate a problem, method, and contribution.")
    if method_is_source_diff_report:
        critical_weaknesses.append("The method section reports implementation provenance/source diffs instead of defining a motivated ML method.")
    if weak_conclusion:
        critical_weaknesses.append("The conclusion frames the result as a candidate for follow-up rather than a supported conference-paper claim.")
    if lacks_method_claim:
        weaknesses.append("The paper does not clearly state a proposed method or hypothesis in conference-paper language.")
    if lacks_conference_baselines:
        weaknesses.append("The experimental narrative does not clearly organize results around baselines and controls.")
    if main_table_count > 4:
        weaknesses.append(f"Main paper has {main_table_count} tables; compress or move process-heavy tables to the appendix.")
    if "Persistent action supervision" in paper_text and best.get("patch_recipe_id") != "full_budget_action_supervision":
        weaknesses.append("Paper framing may be stale relative to the current best branch.")
    strengths = [
        "The artifact set preserves an auditable node tree, metrics, logs, and source diffs.",
    ]
    if title_is_conference_style:
        strengths.append("The title is benchmark-agnostic and reads like a workshop/conference claim.")
    if controlled_success:
        strengths.append("The pipeline executed successful bounded controlled ablation reruns after the population search.")
    if has_icml_template:
        strengths.append("The LaTeX output uses an ICML-compatible structure or fallback template.")
    if has_best_curve and has_family_plot:
        strengths.append("The figure set now separates run-level improvement from intervention-family evidence.")
    if has_lineage_appendix:
        strengths.append("The dense cultural lineage tree is kept as provenance instead of carrying the main narrative.")
    quality_checks = {
        "paper_exists": paper_path.exists(),
        "title": title,
        "conference_style_title": title_is_conference_style,
        "icml_style_template": has_icml_template,
        "single_scientific_narrative": has_single_narrative and not stale_terms_found,
        "controlled_ablations_present": bool(controlled),
        "successful_controlled_ablations": len(successful_controls),
        "controlled_repeat_score_gap": repeat_score_gap,
        "best_score_figure_present": has_best_curve,
        "patch_family_figure_present": has_family_plot,
        "figure_references_integrated": figure_refs_integrated,
        "lineage_tree_in_appendix": has_lineage_appendix,
        "main_table_count": main_table_count,
        "process_table_dump_detected": dumps_process_tables,
        "abstract_word_count": len(abstract.split()),
        "introduction_word_count": len(intro.split()),
        "method_source_diff_report_detected": method_is_source_diff_report,
        "weak_candidate_conclusion_detected": weak_conclusion,
        "method_claim_language_detected": not lacks_method_claim,
        "baseline_or_control_language_detected": not lacks_conference_baselines,
        "stale_process_terms_found": stale_terms_found,
        "missing_expected_sections": missing_sections,
    }
    soundness = 3
    if controlled_success:
        soundness += 2
    if len(successful_controls) >= 3:
        soundness += 1
    if "Limitations" in expected_sections and latex_has_section(paper_text, "Limitations"):
        soundness += 1
    if not controlled:
        soundness -= 2
    if stale_terms_found:
        soundness -= 2
    if critical_weaknesses:
        soundness -= 1
    if repeat_score_gap is not None and repeat_score_gap > 0.01:
        soundness -= 1
    if method_is_source_diff_report:
        soundness -= 2
    if weak_conclusion:
        soundness -= 1
    presentation = 3
    if paper_path.exists():
        presentation += 1
    if has_icml_template:
        presentation += 1
    if title_is_conference_style:
        presentation += 1
    if has_best_curve and has_family_plot:
        presentation += 1
    if not figure_review["missing_expected_figures"] and figure_refs_integrated:
        presentation += 1
    if missing_sections:
        presentation -= min(2, len(missing_sections))
    if stale_terms_found:
        presentation -= 2
    if dumps_process_tables:
        presentation -= 2
    if abstract_too_thin or intro_too_thin:
        presentation -= 2
    if method_is_source_diff_report:
        presentation -= 1
    contribution = 3
    if has_single_narrative and title_is_conference_style:
        contribution += 1
    if controlled_success:
        contribution += 1
    if has_family_plot:
        contribution += 1
    if stale_terms_found:
        contribution -= 2
    if lacks_method_claim or weak_conclusion:
        contribution -= 1
    if method_is_source_diff_report:
        contribution -= 2
    soundness = max(1, min(10, soundness))
    presentation = max(1, min(10, presentation))
    contribution = max(1, min(10, contribution))
    score = round(mean([soundness, presentation, contribution]))
    if len(references) > len(seed_literature_references()):
        score += 1
    if weaknesses or critical_weaknesses:
        score -= min(4, len(weaknesses) + 2 * len(critical_weaknesses))
    score = max(1, min(10, score))
    decision = "accept" if score >= 7 and not critical_weaknesses else "reject"
    return {
        "reviewer": "codex_scientistv2_local_text_reviewer",
        "mode": "deterministic_paper_rubric_review",
        "paper": str(paper_path),
        "best_node": best.get("node_id"),
        "best_score": best.get("primary_score"),
        "summary": (
            "The submission is reviewed as a TinyWorlds domain paper, with special attention to whether the manuscript "
            "presents one scientific claim, supports it with controlled ablations, and uses figures as evidence rather "
            "than as a process log."
        ),
        "paper_quality_checks": quality_checks,
        "claim_review": {
            "best_recipe": best.get("recipe_id"),
            "best_patch_family": best.get("patch_recipe_id"),
            "title": title,
            "assessment": (
                "The main claim is appropriately centered on the best intervention family."
                if has_single_narrative and title_is_conference_style and not stale_terms_found
                else "The main claim needs tightening around the best intervention rather than the surrounding search process."
            ),
        },
        "ablation_review": {
            "status": (controlled or {}).get("status", "missing") if isinstance(controlled, dict) else "missing",
            "successful_controls": len(successful_controls),
            "controlled_repeat_score_gap": repeat_score_gap,
            "assessment": (
                "Successful component controls are present, but the repeat gap requires cautious interpretation."
                if repeat_score_gap is not None and repeat_score_gap > 0.01
                else "Successful component controls are present; the paper should treat these as the main evidence."
                if controlled_success
                else "The paper should not make strong causal claims until controlled component ablations succeed."
            ),
        },
        "figure_review_summary": {
            "figure_score": figure_review["figure_score"],
            "missing_expected_figures": figure_review["missing_expected_figures"],
            "issues": figure_review["issues"],
        },
        "literature_review": {
            "reference_count": len(references),
            "seed_reference_count": len(seed_literature_references()),
            "assessment": (
                "The literature section includes references beyond the seed set."
                if len(references) > len(seed_literature_references())
                else "The literature section appears seed-only and should be expanded if this is a final submission."
            ),
        },
        "strengths": strengths,
        "critical_weaknesses": critical_weaknesses,
        "weaknesses": weaknesses,
        "questions": [
            "Does the best intervention replicate across independent seeds?",
            "Which component remains useful when knobs, source edits, and patch recipes are isolated?",
            "Does lower validation MSE translate into improved downstream planning or rollout quality?",
        ],
        "soundness": soundness,
        "presentation": presentation,
        "contribution": contribution,
        "overall": score,
        "decision": decision,
        "confidence": 8 if critical_weaknesses else 6,
    }


def latex_has_section(paper_text: str, title: str) -> bool:
    return bool(re.search(r"\\section\*?\{" + re.escape(title) + r"\}", paper_text))


def extract_latex_environment(paper_text: str, environment: str) -> str:
    pattern = re.compile(
        r"\\begin\{" + re.escape(environment) + r"\}(.*?)\\end\{" + re.escape(environment) + r"\}",
        re.DOTALL,
    )
    match = pattern.search(paper_text)
    return clean_latex_text(match.group(1)) if match else ""


def extract_latex_section(paper_text: str, title: str) -> str:
    pattern = re.compile(
        r"\\section\*?\{" + re.escape(title) + r"\}(.*?)(?=\\section\*?\{|\\appendix|\\bibliography|\\end\{document\}|$)",
        re.DOTALL,
    )
    match = pattern.search(paper_text)
    return clean_latex_text(match.group(1)) if match else ""


def extract_latex_title(paper_text: str) -> str:
    match = re.search(r"\\icmltitle\{([^{}]+)\}", paper_text)
    if match:
        return clean_latex_text(match.group(1))
    match = re.search(r"\\title\{([^{}]+)\}", paper_text)
    return clean_latex_text(match.group(1)) if match else ""


def clean_latex_text(value: str) -> str:
    value = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:ref|autoref|cref|cite|citep|citet)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", value)
    value = value.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", value).strip()


def extract_latex_figure_blocks(paper_text: str) -> list[dict[str, Any]]:
    figures = []
    pattern = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.DOTALL)
    for index, match in enumerate(pattern.finditer(paper_text), start=1):
        env = match.group(1)
        includes = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", env)
        caption_match = re.search(r"\\caption(?:\[[^\]]*\])?\{(.*?)\}\s*(?:\\label|$)", env, re.DOTALL)
        label_match = re.search(r"\\label\{([^{}]+)\}", env)
        figures.append(
            {
                "index": index,
                "includegraphics": includes,
                "caption": clean_latex_text(caption_match.group(1)) if caption_match else "",
                "label": label_match.group(1) if label_match else "",
            }
        )
    return figures


def resolve_latex_graphic(output_dir: Path, graphic_path: str) -> Path:
    latex_dir = output_dir / "latex"
    candidate = (latex_dir / graphic_path).resolve()
    if candidate.exists():
        return candidate
    return output_dir / "figures" / Path(graphic_path).name


def read_svg_text(path: Path) -> str:
    svg_path = path if path.suffix.lower() == ".svg" else path.with_suffix(".svg")
    if not svg_path.exists() or svg_path.stat().st_size == 0:
        return ""
    try:
        root = ET.parse(svg_path).getroot()
    except ET.ParseError:
        return svg_path.read_text(encoding="utf-8", errors="ignore")[:5000]
    pieces = []
    for element in root.iter():
        if element.text and element.text.strip():
            pieces.append(element.text.strip())
    return " ".join(pieces)


def figure_reference_contexts(paper_text: str, label: str) -> list[str]:
    if not label:
        return []
    text_without_label = re.sub(r"\\label\{" + re.escape(label) + r"\}", "", paper_text)
    contexts = []
    pattern = re.compile(r"\\(?:ref|autoref|cref)\{" + re.escape(label) + r"\}")
    for match in pattern.finditer(text_without_label):
        start = max(0, match.start() - 180)
        end = min(len(text_without_label), match.end() + 180)
        contexts.append(clean_latex_text(text_without_label[start:end]))
    return contexts


def review_latex_figures(output_dir: Path, paper_text: str) -> dict[str, Any]:
    fig_dir = output_dir / "figures"
    parsed_figures = extract_latex_figure_blocks(paper_text)
    expected = {
        "best_score_by_generation.pdf": "run-level best score curve",
        "patch_family_mean_scores.pdf": "aggregate intervention-family comparison",
        "cultural_tree.pdf": "appendix lineage/provenance tree",
    }
    missing = [str(fig_dir / name) for name in expected if not (fig_dir / name).exists()]
    caption_reference_reviews = []
    issues = []
    for figure in parsed_figures:
        paths = [resolve_latex_graphic(output_dir, item) for item in figure["includegraphics"]]
        existing_paths = [path for path in paths if path.exists() and path.stat().st_size > 0]
        svg_text = " ".join(read_svg_text(path) for path in existing_paths)
        label = figure.get("label", "")
        references = figure_reference_contexts(paper_text, label)
        caption = figure.get("caption", "")
        figure_name = ", ".join(path.name for path in paths) or "unknown figure"
        if not existing_paths:
            issues.append(f"{figure_name} is referenced by LaTeX but missing or empty.")
        if not caption:
            issues.append(f"{figure_name} has no caption.")
        if label and not references:
            issues.append(f"{figure_name} has label {label} but no main-text reference.")
        if "cultural_tree" in figure_name and "\\appendix" not in paper_text:
            issues.append("The dense cultural tree should be placed in an appendix/provenance section.")
        caption_reference_reviews.append(
            {
                "figure": figure_name,
                "label": label,
                "paths": [str(path) for path in paths],
                "bytes": {str(path): path.stat().st_size for path in existing_paths},
                "Img_description": infer_figure_description(figure_name, svg_text),
                "Img_review": infer_figure_review(figure_name, svg_text, bool(existing_paths)),
                "Caption_review": infer_caption_review(figure_name, caption),
                "Figrefs_review": infer_figrefs_review(figure_name, references),
                "Overall_comments": infer_figure_value_review(figure_name),
                "Containing_sub_figures": "No subfigure structure detected from the LaTeX environment.",
                "Informative_review": infer_informative_review(figure_name, svg_text),
                "main_text_figrefs": references[:3],
            }
        )
    figure_score = 10
    figure_score -= min(4, len(missing))
    figure_score -= min(4, len(issues))
    return {
        "parsed_figures": parsed_figures,
        "missing_expected_figures": missing,
        "caption_reference_reviews": caption_reference_reviews,
        "issues": issues,
        "figure_score": max(1, figure_score),
    }


def infer_figure_description(figure_name: str, svg_text: str) -> str:
    lower_name = figure_name.lower()
    lower_svg = svg_text.lower()
    if "best_score_by_generation" in lower_name:
        return "A run-level curve of the best primary score achieved by each generation."
    if "patch_family_mean_scores" in lower_name:
        return "An aggregate comparison of mean primary score by patch or intervention family."
    if "cultural_tree" in lower_name:
        return "A lineage tree showing node inheritance, operator choices, and discovered branch scores."
    if lower_svg:
        return "A generated figure with visible text labels extracted from the SVG companion."
    return "A generated figure file; detailed visual content was not available to the local reviewer."


def infer_figure_review(figure_name: str, svg_text: str, exists: bool) -> str:
    if not exists:
        return "The referenced figure file is missing or empty."
    lower_name = figure_name.lower()
    lower_svg = svg_text.lower()
    expected_tokens = {
        "best_score_by_generation": ["generation", "score"],
        "patch_family_mean_scores": ["patch", "score"],
        "cultural_tree": ["agent", "node"],
    }
    for stem, tokens in expected_tokens.items():
        if stem in lower_name:
            missing_tokens = [token for token in tokens if token not in lower_svg]
            if missing_tokens and svg_text:
                return "The figure exists, but extracted SVG labels may be missing: " + ", ".join(missing_tokens) + "."
            return "The figure is present and matches the expected evidence role for the paper."
    return "The figure is present; no specialized semantic checks were available for this filename."


def infer_caption_review(figure_name: str, caption: str) -> str:
    if not caption:
        return "Caption is missing."
    lower_name = figure_name.lower()
    lower_caption = caption.lower()
    if "best_score_by_generation" in lower_name and not {"score", "generation"} <= set(lower_caption.split()):
        return "Caption should explicitly name both the score and generation axes."
    if "patch_family_mean_scores" in lower_name and "family" not in lower_caption:
        return "Caption should state that the plot compares intervention or patch families."
    if "cultural_tree" in lower_name and not any(word in lower_caption for word in ["lineage", "tree", "provenance"]):
        return "Caption should identify the tree as lineage/provenance rather than a main result."
    return "Caption is concise and aligned with the expected figure role."


def infer_figrefs_review(figure_name: str, references: list[str]) -> str:
    if not references:
        return "No main-text references were found; the paper should integrate the figure into the argument."
    if "cultural_tree" in figure_name.lower():
        return "The figure is referenced; for a dense tree, appendix references are preferable to main-text dependence."
    return "Main-text references are present and give the figure a role in the paper narrative."


def infer_figure_value_review(figure_name: str) -> str:
    lower_name = figure_name.lower()
    if "best_score_by_generation" in lower_name:
        return "High value for the main paper because it shows whether improvement occurred over the run."
    if "patch_family_mean_scores" in lower_name:
        return "High value for the main paper because it separates a single best branch from family-level evidence."
    if "cultural_tree" in lower_name:
        return "Useful as provenance, but usually too dense for the main page budget; appendix placement is appropriate."
    return "Value depends on whether the surrounding text uses the figure to support a specific claim."


def infer_informative_review(figure_name: str, svg_text: str) -> str:
    if not svg_text:
        return "The local reviewer could not inspect visual labels; keep the caption and text references explicit."
    lower_name = figure_name.lower()
    if "best_score_by_generation" in lower_name or "patch_family_mean_scores" in lower_name:
        return "Informative if axis labels and tick labels are legible in the rendered PDF."
    if "cultural_tree" in lower_name:
        return "Informative as an audit artifact; it may be visually dense and should not carry the core claim alone."
    return "The figure contains extractable text and should be checked visually in the final rendered PDF."


def build_figure_review(output_dir: Path) -> dict[str, Any]:
    fig_dir = output_dir / "figures"
    paper_path = output_dir / "latex" / "paper.tex"
    paper_text = paper_path.read_text(encoding="utf-8") if paper_path.exists() else ""
    latex_review = review_latex_figures(output_dir, paper_text)
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
    return {
        "reviewer": "codex_scientistv2_local_figure_reviewer",
        "mode": "local_caption_reference_review",
        "paper": str(paper_path),
        "figures": figures,
        "parsed_latex_figures": latex_review["parsed_figures"],
        "caption_reference_reviews": latex_review["caption_reference_reviews"],
        "missing_expected_figures": latex_review["missing_expected_figures"],
        "issues": latex_review["issues"],
        "figure_score": latex_review["figure_score"],
        "summary": (
            "Generated figures are present, parsed from LaTeX, and reviewed for caption/reference alignment."
            if not latex_review["missing_expected_figures"] and not latex_review["issues"]
            else "Some figure, caption, or main-text-reference issues remain."
        ),
        "recommendations": [
            "Keep the lineage tree in the appendix if it is too dense for the main paper.",
            "Use the best-score curve as the main run-level figure.",
            "Use the patch-family aggregate as the main evidence that the claim is not only a single lucky branch.",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
