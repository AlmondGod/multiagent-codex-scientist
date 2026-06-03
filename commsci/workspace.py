from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .artifacts import ensure_dir


def prepare_agent_workspace(
    source_dir: str | None,
    workspace_dir: Path,
    dry_run: bool,
    branch_name: str,
) -> str:
    ensure_dir(workspace_dir.parent)
    if workspace_dir.exists():
        return "existing"
    if dry_run:
        ensure_dir(workspace_dir)
        (workspace_dir / "DRY_RUN_WORKSPACE.txt").write_text(
            "Dry-run workspace placeholder. No TinyWorlds files were copied.\n",
            encoding="utf-8",
        )
        return "dry_run_placeholder"
    if not source_dir:
        raise RuntimeError("Real execution requires --tinyworlds_dir.")
    source = Path(source_dir).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"TinyWorlds directory does not exist: {source}")
    if (source / ".git").exists():
        try:
            subprocess.run(
                ["git", "-C", str(source), "worktree", "add", "-B", branch_name, str(workspace_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            return "git_worktree"
        except subprocess.CalledProcessError:
            pass
    shutil.copytree(source, workspace_dir, ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv"))
    return "copytree"


def collect_git_diff(workspace_dir: Path) -> str:
    if not (workspace_dir / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(workspace_dir), "diff", "--patch"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except Exception as exc:
        return f"Could not collect git diff: {exc}\n"
