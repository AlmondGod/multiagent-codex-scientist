# agent_2 TinyWorlds Research Note

## Hypothesis
Agent 2 hypothesis: a small targeted TinyWorlds world-model change can improve the primary metric under the fixed budget for: Improve TinyWorlds world-model quality under a fixed small compute budget.

## Method
Run one bounded TinyWorlds training/evaluation pass, edit only allowed files, and compare primary metric plus failure status against the baseline.

## Experiment 1 result
{
  "codebook_entropy": 0.5826,
  "experiment_success": true,
  "prediction_loss": 0.9332,
  "primary_score": 0.5238,
  "reconstruction_loss": 0.9092,
  "runtime_seconds": 16.229
}

## Critique received
1. Strongest concern: the current interpretation may over-credit one noisy metric.
2. Missing control, baseline, or ablation: add a one-variable ablation against the first experiment.
3. Metric or evaluation risk: report primary score plus runtime and failure status.
4. Implementation/debug risk: verify the changed path is inside allowed_files.
5. Suggested next experiment: keep compute fixed and run a smaller controlled variant.
6. Falsification: no improvement or a failed run under the same budget.
7. Run/change recommendation: change the proposed next experiment to the controlled ablation.

Condition context: peer_critique_with_roles. Prompt chars: 3961.


## Decision change
{
  "change_type": "added_ablation",
  "decision_changed": true,
  "evidence": "Primary score changed from 0.5238 to 0.5376.",
  "input_received": "{\n  \"decision_changed\": true,\n  \"change_type\": \"added_ablation\",\n  \"reason\": \"Dry-run critique recommends a smaller controlled follow-up.\",\n  \"revised_experiment_plan\": \"Run a bounded ablation that changes one world-model setting and compares the primary metric.\"\n}",
  "later_helped": true,
  "reason": "Dry-run critique recommends a smaller controlled follow-up.",
  "revised_experiment_plan": "Run a bounded ablation that changes one world-model setting and compares the primary metric."
}

## Experiment 2 result
{
  "codebook_entropy": 0.6292,
  "experiment_success": true,
  "prediction_loss": 0.9065,
  "primary_score": 0.5376,
  "reconstruction_loss": 0.8602,
  "runtime_seconds": 16.03
}

## Claim-to-evidence table
| Claim | Evidence |
| --- | --- |
| The second experiment followed critique input. | decision_change.json records change_type=added_ablation. |
| TinyWorlds metric changed after revision. | metrics_experiment_1.json and metrics_experiment_2.json report primary_score when available. |

## Limitations
This v0 note is intentionally short and bounded to two experiment meta-steps. Dry-run outputs are orchestration tests, not scientific evidence.

## Next step
Run the same condition on real TinyWorlds with matched training, runtime, token, and experiment budgets.
