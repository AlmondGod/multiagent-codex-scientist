from __future__ import annotations

import os
import subprocess
import sys
import traceback
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


def run_reviewer(note_path: Path, review_path: Path, dry_run: bool, config: dict[str, Any]) -> dict[str, Any]:
    if dry_run:
        review = {
            "reviewer_score": 6.5,
            "unsupported_claim_count": 1,
            "ablation_quality": 0.7,
            "raw": "Dry-run reviewer: plausible note with one unsupported claim.",
        }
        write_json(review_path, review)
        return review
    if not config["base_system"].get("reviewer_enabled", True):
        review = {
            "reviewer_score": None,
            "unsupported_claim_count": heuristic_unsupported_claim_count(note_path),
            "ablation_quality": None,
            "failure": "Reviewer disabled by config.",
            "note_path": str(note_path),
        }
        write_json(review_path, review)
        return review
    try:
        review = run_ai_scientist_reviewer(note_path, config)
    except Exception as exc:
        review = {
            "reviewer_score": None,
            "unsupported_claim_count": heuristic_unsupported_claim_count(note_path),
            "ablation_quality": None,
            "failure": str(exc),
            "traceback": traceback.format_exc(),
            "note_path": str(note_path),
            "reviewer_backend": config["base_system"].get("reviewer_backend"),
            "reviewer_model": config["base_system"].get("reviewer_model"),
        }
    write_json(review_path, review)
    return review


def run_ai_scientist_reviewer(note_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    ai_scientist_dir = config["paths"].get("ai_scientist_v2_dir")
    if not ai_scientist_dir:
        raise RuntimeError("AI-Scientist-v2 reviewer requires --ai_scientist_v2_dir.")
    ai_path = Path(ai_scientist_dir).expanduser().resolve()
    if not ai_path.exists():
        raise RuntimeError(f"AI-Scientist-v2 directory does not exist: {ai_path}")
    if str(ai_path) not in sys.path:
        sys.path.insert(0, str(ai_path))

    from ai_scientist.llm import create_client
    from ai_scientist.perform_llm_review import perform_review

    reviewer_cfg = config["base_system"]
    model = reviewer_cfg.get("reviewer_model") or config["model"]["default_model"]
    backend = reviewer_cfg.get("reviewer_backend", "ai_scientist")
    model_url = reviewer_cfg.get("reviewer_model_url")
    if str(model).startswith("ollama/"):
        os.environ.setdefault("OPENAI_API_KEY", "ollama-local")
        os.environ.setdefault("OLLAMA_API_KEY", "ollama-local")

    if backend == "openai_compatible":
        import openai

        client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", os.environ.get("OLLAMA_API_KEY", "dummy")),
            base_url=(model_url or config["model"].get("model_url") or "http://localhost:1234/v1").rstrip("/"),
        )
    else:
        client, model = create_client(model)

    note_text = note_path.read_text(encoding="utf-8")
    raw_review = perform_review(
        note_text,
        model=model,
        client=client,
        num_reflections=int(reviewer_cfg.get("reviewer_num_reflections", 1)),
        num_fs_examples=int(reviewer_cfg.get("reviewer_num_fs_examples", 0)),
        num_reviews_ensemble=1,
        temperature=float(reviewer_cfg.get("reviewer_temperature", 0.2)),
    )
    unsupported_claim_count = heuristic_unsupported_claim_count(note_path)
    return {
        "reviewer_score": raw_review.get("Overall") if isinstance(raw_review, dict) else None,
        "unsupported_claim_count": unsupported_claim_count,
        "ablation_quality": derive_ablation_quality(raw_review),
        "raw_review": raw_review,
        "note_path": str(note_path),
        "reviewer_backend": backend,
        "reviewer_model": model,
        "reviewer_model_url": model_url or config["model"].get("model_url"),
    }


def derive_ablation_quality(raw_review: Any) -> float | None:
    if not isinstance(raw_review, dict):
        return None
    soundness = raw_review.get("Soundness")
    quality = raw_review.get("Quality")
    values = [value for value in (soundness, quality) if isinstance(value, (int, float))]
    if not values:
        return None
    return round(sum(values) / (4 * len(values)), 3)


def heuristic_unsupported_claim_count(note_path: Path) -> int:
    text = note_path.read_text(encoding="utf-8").lower()
    markers = ["unsupported", "not supported", "no evidence", "unclear evidence"]
    return sum(text.count(marker) for marker in markers)
