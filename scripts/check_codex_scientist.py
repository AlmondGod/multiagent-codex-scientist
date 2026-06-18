#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from commsci.codex_scientist.actions import apply_patch_recipe, normalize_action
from commsci.codex_scientist.checks import check_artifact_completeness, check_metrics_fixture
from commsci.codex_scientist.communication import load_critique_override, load_decision_override
from commsci.codex_scientist.schemas import CodexNode
from commsci.codex_scientistv2.pipeline import write_v2_outputs


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
                "inheritance_mode": "copy",
                "source_agent_id": "agent_1",
                "source_node_id": "peer_critique_agent_1_node_1",
                "copied_recipe_id": "agent_1_step_1",
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
        assert action["inheritance"]["mode"] == "copy"
        assert action["inheritance"]["source_agent_ids"] == ["agent_1"]
        assert action["inheritance"]["source_node_ids"] == ["peer_critique_agent_1_node_1"]
        (tmp_path / "train.py").write_text("ALPHA = 1\n", encoding="utf-8")
        edit_action = normalize_action(
            {
                "recipe_id": "direct_edit_probe",
                "inheritance_mode": "invent",
                "file_edits": [
                    {
                        "path": "train.py",
                        "find": "ALPHA = 1\n",
                        "replace": "ALPHA = 2\n",
                    }
                ],
            },
            {"experiment": {"allowed_files": ["train.py", "models.py"]}},
            "fallback_recipe",
        )
        edit_result = apply_patch_recipe(tmp_path, edit_action)
        assert edit_result["file_edits_applied"] == 1
        assert "ALPHA = 2" in (tmp_path / "train.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        node_id = "cultural_evolution_agent_0_node_1"
        node_dir = (
            tmp_path
            / "cultural_evolution"
            / "agent_0"
            / "artifacts"
            / "codex_scientist"
            / "nodes"
            / node_id
        )
        node_dir.mkdir(parents=True)
        (node_dir / "action.json").write_text(
            json.dumps(
                {
                    "recipe_id": "g0_agent0_fixture",
                    "patch_recipe": {"id": "baseline_no_patch"},
                    "inheritance": {"mode": "invent", "source_node_ids": []},
                    "knobs": {"depth": 1},
                }
            ),
            encoding="utf-8",
        )
        (node_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_success": True,
                    "primary_score": 0.75,
                    "val_mse": 0.25,
                    "runtime_sec": 1.5,
                    "params_M": 0.1,
                    "dataset": "fixture",
                }
            ),
            encoding="utf-8",
        )
        (node_dir / "patch_result.json").write_text(
            json.dumps({"patch_applied": False, "code_diff": ""}),
            encoding="utf-8",
        )
        (node_dir / "logs.txt").write_text("fixture log\n", encoding="utf-8")
        (tmp_path / "population_summary_generation_00.json").write_text(
            json.dumps(
                [
                    {
                        "generation": 0,
                        "agent_id": "agent_0",
                        "node_id": node_id,
                        "parent_id": None,
                        "primary_score": 0.75,
                        "val_mse": 0.25,
                        "experiment_success": True,
                        "recipe_id": "g0_agent0_fixture",
                        "patch_recipe_id": "baseline_no_patch",
                        "knobs": {"depth": 1},
                        "inheritance_mode": "invent",
                        "source_node_ids": [],
                        "rationale": "fixture",
                    }
                ]
            ),
            encoding="utf-8",
        )
        previous_offline = os.environ.get("CODEX_SCIENTISTV2_OFFLINE_LITERATURE")
        os.environ["CODEX_SCIENTISTV2_OFFLINE_LITERATURE"] = "1"
        try:
            v2_dir = write_v2_outputs(tmp_path)
        finally:
            if previous_offline is None:
                os.environ.pop("CODEX_SCIENTISTV2_OFFLINE_LITERATURE", None)
            else:
                os.environ["CODEX_SCIENTISTV2_OFFLINE_LITERATURE"] = previous_offline
        assert (v2_dir / "run_manifest.json").exists()
        assert (v2_dir / "rich_nodes.jsonl").exists()
        assert (v2_dir / "stage_reports" / "initial_implementation.json").exists()
        assert (v2_dir / "ablation_report.json").exists()
        assert (v2_dir / "literature" / "references.bib").exists()
        assert (v2_dir / "literature" / "literature_review.json").exists()
        assert (v2_dir / "codex_tasks" / "figure_making.md").exists()
        assert (v2_dir / "codex_tasks" / "paper_reflection.md").exists()
        assert (tmp_path / "figures" / "best_score_by_generation.svg").exists()
        assert (tmp_path / "figures" / "cultural_tree.svg").exists()
        assert (tmp_path / "latex" / "template.tex").exists()
        assert (tmp_path / "latex" / "paper.tex").exists()
        assert (tmp_path / "latex" / "compile.log").exists()
        assert (tmp_path / "review" / "review.json").exists()
        assert (tmp_path / "review" / "vlm_review.json").exists()
        assert (tmp_path / "review" / "review_scaffold.json").exists()
        assert (tmp_path / "paper.md").exists()
    print("codex_scientist checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
