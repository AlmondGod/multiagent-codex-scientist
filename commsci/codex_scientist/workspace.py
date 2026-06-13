from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from commsci.ai_scientist_runner import build_canonical_runfile, resolve_ai_scientist_data_dir
from commsci.artifacts import ensure_dir, write_text


def prepare_node_workspace(config: dict[str, Any], node_root: Path) -> Path:
    """Create an isolated TinyWorlds workspace for one Codex-Scientist node.

    The source tree is copied without data/results/checkpoints, and the data
    directory is symlinked so each node can run independently without duplicating
    large files.
    """
    source = resolve_ai_scientist_data_dir(config)
    workspace = ensure_dir(node_root / "workspace")
    input_dir = workspace / "input"
    if not input_dir.exists():
        shutil.copytree(
            source,
            input_dir,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                ".venv",
                "data",
                "runs",
                "results",
                "working",
                "checkpoints",
                "wandb",
                "*.pt",
                "*.pth",
                "*.ckpt",
                "*.h5",
            ),
        )
        data_src = source / "data"
        if data_src.exists():
            (input_dir / "data").symlink_to(data_src, target_is_directory=True)
    time_budget = int(config["experiment"].get("codex_scientist_time_budget_seconds",
                                                config["experiment"].get("ai_scientist_time_budget_seconds", 100)))
    write_text(input_dir / "runfile.py", build_canonical_runfile(time_budget))
    ensure_dir(input_dir / "working")
    return input_dir
