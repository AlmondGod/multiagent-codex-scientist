from __future__ import annotations

from pathlib import Path
from typing import Any

from .communication import check_artifact_completeness
from .runner import parse_metrics


def check_metrics_fixture(path: Path) -> dict[str, Any]:
    return parse_metrics(path, "", 0)


__all__ = ["check_artifact_completeness", "check_metrics_fixture"]
