from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .artifacts import ensure_dir


def prepare_agent_workspace(
    source_dir: str | None,
    workspace_dir: Path,
    dry_run: bool,
    branch_name: str,
) -> str:
    ensure_dir(workspace_dir.parent)
    if workspace_dir.exists():
        if not dry_run and source_dir:
            source = Path(source_dir).expanduser().resolve()
            link_shared_dirs(source, workspace_dir, ["data"])
            apply_tinyworlds_compatibility_patches(workspace_dir)
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
        # Copy instead of git-worktree for TinyWorlds smoke runs so large data/results
        # can be excluded and shared by symlink.
        pass
    shutil.copytree(
        source,
        workspace_dir,
        ignore=shutil.ignore_patterns(
            ".git",
            "__pycache__",
            ".venv",
            "runs",
            "results",
            "checkpoints",
            "wandb",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "data",
        ),
    )
    link_shared_dirs(source, workspace_dir, ["data"])
    apply_tinyworlds_compatibility_patches(workspace_dir)
    return "copytree_shared_data"


def link_shared_dirs(source: Path, workspace_dir: Path, names: Iterable[str]) -> None:
    for name in names:
        src = source / name
        dst = workspace_dir / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src, target_is_directory=src.is_dir())


def apply_tinyworlds_compatibility_patches(workspace_dir: Path) -> None:
    """Patch copied TinyWorlds workspaces for the A100 smoke environment.

    The source TinyWorlds repo is left untouched. These patches are only applied
    to isolated agent workspaces and cover single-GPU non-FSDP execution on
    torch versions that do not expose newer FSDP2 symbols.
    """
    replacements = {
        workspace_dir / "utils" / "utils.py": {
            "from torch.distributed.fsdp import FSDPModule": (
                "try:\n"
                "    from torch.distributed.fsdp import FSDPModule\n"
                "except ImportError:\n"
                "    class FSDPModule:\n"
                "        pass"
            ),
        },
        workspace_dir / "scripts" / "train_video_tokenizer.py": {
            "from torch.distributed.fsdp import FSDPModule": (
                "try:\n"
                "    from torch.distributed.fsdp import FSDPModule\n"
                "except ImportError:\n"
                "    class FSDPModule:\n"
                "        pass"
            ),
        },
        workspace_dir / "scripts" / "train_latent_actions.py": {
            "from torch.distributed.fsdp import FSDPModule": (
                "try:\n"
                "    from torch.distributed.fsdp import FSDPModule\n"
                "except ImportError:\n"
                "    class FSDPModule:\n"
                "        pass"
            ),
        },
        workspace_dir / "scripts" / "train_dynamics.py": {
            "from torch.distributed.fsdp import FSDPModule": (
                "try:\n"
                "    from torch.distributed.fsdp import FSDPModule\n"
                "except ImportError:\n"
                "    class FSDPModule:\n"
                "        pass"
            ),
        },
        workspace_dir / "utils" / "distributed.py": {
            "from torch.distributed.fsdp import fully_shard": (
                "try:\n"
                "    from torch.distributed.fsdp import fully_shard\n"
                "except ImportError:\n"
                "    def fully_shard(model, *args, **kwargs):\n"
                "        return model"
            ),
        },
        workspace_dir / "utils" / "config.py": {
            "from typing import Optional": "from typing import Optional, Any",
            "from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy": (
                "try:\n"
                "    from torch.distributed.fsdp import MixedPrecisionPolicy, CPUOffloadPolicy\n"
                "except ImportError:\n"
                "    class MixedPrecisionPolicy:\n"
                "        def __init__(self, **kwargs):\n"
                "            self.__dict__.update(kwargs)\n"
                "    class CPUOffloadPolicy:\n"
                "        pass"
            ),
            "offload_policy: CPUOffloadPolicy | None = None": "offload_policy: Any = None",
        },
        workspace_dir / "utils" / "optimizer_utils.py": {
            "fused=True": "fused=False",
        },
    }
    for path, file_replacements in replacements.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in file_replacements.items():
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8")


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
