#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from commsci.codex_scientist.actions import apply_patch_recipe, normalize_action
from commsci.codex_scientist.checks import check_artifact_completeness, check_metrics_fixture
from commsci.codex_scientist.communication import load_critique_override, load_decision_override
from commsci.codex_scientist.schemas import CodexNode


def main() -> int:
    node = CodexNode(
        node_id="node_1",
        parent_id=None,
        agent_id="agent_0",
        branch_id="self_critique_agent_0",
        condition="self_critique",
        depth=1,
        hypothesis="smoke",
        action={"kind": "knob_recipe"},
        metrics={"experiment_success": True, "primary_score": 0.5},
        command=["python", "runfile.py"],
        workspace_path="/tmp/workspace",
        artifact_paths=["/tmp/node"],
    )
    assert node.to_dict()["node_id"] == "node_1"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"val_mse": 0.25, "runtime_sec": 1.0}), encoding="utf-8")
        metrics = check_metrics_fixture(metrics_path)
        assert metrics["experiment_success"] is True
        assert metrics["primary_score"] == 0.8
        completeness = check_artifact_completeness(tmp_path)
        assert completeness["complete"] is False
        assert "branch_summary.json" in completeness["missing_agent_artifacts"]
        critique_dir = tmp_path / "critiques"
        critique_dir.mkdir()
        (critique_dir / "agent_1_to_agent_0.json").write_text(
            json.dumps(
                {
                    "author_id": "agent_1",
                    "target_id": "agent_0",
                    "critique": "Live Codex critique",
                    "recommended_next_action": "Change one validated knob.",
                }
            ),
            encoding="utf-8",
        )
        decision_dir = tmp_path / "decisions"
        decision_dir.mkdir()
        (decision_dir / "agent_0_decision.md").write_text("Run the paired step-2 action override.", encoding="utf-8")
        config = {
            "experiment": {
                "codex_scientist_critique_overrides_dir": str(critique_dir),
                "codex_scientist_decision_overrides_dir": str(decision_dir),
            }
        }
        critique = load_critique_override(config, "agent_0", "agent_1", "peer_critique")
        assert critique and critique["critique"] == "Live Codex critique"
        decision = load_decision_override(config, "agent_0")
        assert decision and "paired step-2 action" in decision["revised_experiment_plan"]
        (tmp_path / "models.py").write_text(
            "import torch.nn.functional as F\n\n"
            "def loss(pred, target):\n"
            "        return F.mse_loss(pred[:, 0], target)\n",
            encoding="utf-8",
        )
        action = normalize_action(
            {
                "recipe_id": "smooth_l1_probe",
                "patch_recipe_id": "smooth_l1_dynamics_pixel",
                "rationale": "Probe a robust dynamics reconstruction loss.",
            },
            config,
            "fallback_recipe",
        )
        patch_result = apply_patch_recipe(tmp_path, action)
        assert patch_result["patch_applied"] is True
        assert patch_result["changed_files"] == ["models.py"]
        assert "smooth_l1_loss" in (tmp_path / "models.py").read_text(encoding="utf-8")
        assert "smooth_l1_loss" in patch_result["code_diff"]
    print("codex_scientist checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
