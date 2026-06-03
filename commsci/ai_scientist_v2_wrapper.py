from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .artifacts import write_json


def ensure_ai_scientist_v2(config: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    configured = config["paths"].get("ai_scientist_v2_dir")
    if configured and Path(configured).expanduser().exists():
        return {"status": "available", "path": str(Path(configured).expanduser().resolve())}
    if dry_run:
        return {
            "status": "dry_run_unavailable_ok",
            "path": configured,
            "message": "AI-Scientist-v2 was not required for dry-run orchestration.",
        }
    repo_url = config["base_system"]["repo_url"]
    target = Path("external") / "AI-Scientist-v2"
    if not target.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", repo_url, str(target)], check=True)
        except Exception as exc:
            raise RuntimeError(
                "AI-Scientist-v2 is required for real execution. Pass --ai_scientist_v2_dir "
                f"or ensure network access for cloning {repo_url}. Clone error: {exc}"
            ) from exc
    config["paths"]["ai_scientist_v2_dir"] = str(target.resolve())
    return {"status": "cloned", "path": str(target.resolve())}


def propose_initial_branch(agent_index: int, config: dict[str, Any]) -> tuple[str, str]:
    task = config["experiment"]["task_spec"]
    hypothesis = (
        f"Agent {agent_index} hypothesis: a small targeted TinyWorlds world-model change can improve "
        f"the primary metric under the fixed budget for: {task}"
    )
    plan = (
        "Run one bounded TinyWorlds training/evaluation pass, edit only allowed files, "
        "and compare primary metric plus failure status against the baseline."
    )
    return hypothesis, plan


def propose_next_experiment(agent_index: int) -> str:
    return (
        f"Agent {agent_index} proposed next experiment: keep compute fixed and test one controlled "
        "change suggested by experiment 1 metrics."
    )


def run_reviewer(note_path: Path, review_path: Path, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        review = {
            "reviewer_score": 6.5,
            "unsupported_claim_count": 1,
            "ablation_quality": 0.7,
            "raw": "Dry-run reviewer: plausible note with one unsupported claim.",
        }
        write_json(review_path, review)
        return review
    review = {
        "reviewer_score": None,
        "unsupported_claim_count": None,
        "ablation_quality": None,
        "failure": "Direct AI-Scientist-v2 reviewer integration is not implemented in this thin v0 adapter.",
        "note_path": str(note_path),
    }
    write_json(review_path, review)
    return review
