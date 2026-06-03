from __future__ import annotations

import json
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def run_experiment(
    workspace_dir: Path,
    artifact_dir: Path,
    agent_index: int,
    step: int,
    config: dict[str, Any],
    dry_run: bool,
    seed: int,
) -> tuple[dict[str, Any], str]:
    if dry_run:
        return fake_experiment(agent_index, step, seed)
    timeout = int(config["compute"]["max_runtime_minutes_per_experiment"] * 60)
    train_cmd = config["experiment"].get("train_command")
    eval_cmd = config["experiment"].get("eval_command")
    if not train_cmd and not eval_cmd:
        raise RuntimeError("Real execution requires --train_command and/or --eval_command.")
    logs: list[str] = []
    success = True
    start = time.time()
    for name, command in (("train", train_cmd), ("eval", eval_cmd)):
        if not command:
            continue
        result = subprocess.run(
            command,
            cwd=workspace_dir,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logs.append(f"$ {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
        if result.returncode != 0:
            success = False
            logs.append(f"{name} returned {result.returncode}\n")
    metrics = parse_metrics("\n".join(logs), artifact_dir, config)
    metrics.setdefault("runtime_seconds", round(time.time() - start, 3))
    metrics["experiment_success"] = success
    return metrics, "\n".join(logs)


def fake_experiment(agent_index: int, step: int, seed: int) -> tuple[dict[str, Any], str]:
    rng = random.Random(seed * 1000 + agent_index * 31 + step * 17)
    base = 1.0 - agent_index * 0.03
    improvement = 0.04 * step + rng.uniform(-0.015, 0.025)
    loss = max(0.05, base - improvement)
    metrics = {
        "primary_score": round(1.0 / (1.0 + loss), 4),
        "reconstruction_loss": round(loss, 4),
        "prediction_loss": round(loss + rng.uniform(0.01, 0.05), 4),
        "codebook_entropy": round(0.55 + rng.uniform(0.0, 0.2), 4),
        "runtime_seconds": round(15 + rng.uniform(0, 5), 3),
        "experiment_success": True,
    }
    logs = (
        f"DRY RUN agent={agent_index} step={step}\n"
        f"primary_score={metrics['primary_score']}\n"
        f"reconstruction_loss={metrics['reconstruction_loss']}\n"
    )
    return metrics, logs


def parse_metrics(logs: str, artifact_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, rel_path in (config["experiment"].get("metric_json_paths") or {}).items():
        path = artifact_dir / rel_path
        if path.exists():
            try:
                metrics[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                metrics[f"{name}_parse_error"] = str(exc)
    for name, pattern in (config["experiment"].get("metric_regexes") or {}).items():
        match = re.search(pattern, logs)
        if match:
            try:
                metrics[name] = float(match.group(1))
            except ValueError:
                metrics[name] = match.group(1)
    return metrics
