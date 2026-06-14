# Codex-Scientist Code-Recipe Communication Smoke

## Motivation

The broader hypothesis is that automated science should improve when research
agents exchange useful information, analogous to cultural evolution: independent
branches explore different ideas, communication lets the population copy,
mutate, or reject those ideas, and later search should improve relative to
isolated self-critique.

Earlier TinyWorlds runs mostly varied scalar knobs. This smoke widened the
action surface to bounded architecture/training-loop changes while preserving
auditability: agents could choose from curated source patch recipes plus
validated `TW_*` knobs, and each patch was applied only inside the node's
isolated TinyWorlds workspace.

## Setup

- Substrate: TinyWorlds world-model training on local Mac.
- Runner: `experiment.runner: codex_scientist`.
- Conditions: `self_critique`, `peer_critique`, `peer_critique_with_roles`.
- Agents: 3 per condition.
- Search depth: exactly 2 node experiments per agent.
- Budget: 120 seconds per node experiment.
- Parallelism: at most 3 node experiments at once.
- Output runs:
  - `runs/code_recipe_seed0_self_critique`
  - `runs/code_recipe_seed0_peer_critique`
  - `runs/code_recipe_seed0_peer_critique_with_roles`

Live Codex subagents authored action overrides for agents 0 and 1; agent 2 was
authored locally to cover additional recipes while keeping the run bounded. The
communication/reviewer path used the repo's local Codex-Scientist fallback, not
live subagent-authored critique overrides.

## Action Surface

Agents could select one curated patch recipe:

- `action_grad_dynamics`: let action-tokenizer gradients flow into dynamics.
- `smooth_l1_dynamics_pixel`: replace dynamics pixel MSE with smooth-L1.
- `dynamics_first_schedule`: enter dynamics training earlier in short runs.
- `sharpen_change_weights`: focus change weights harder on changing patches.
- `full_budget_action_supervision`: keep action supervision active throughout training.

Each node saved `action.json`, `patch_result.json`, `code_diff.patch`,
`metrics.json`, and logs. The original TinyWorlds checkout was not modified.

## Results

| Condition | Mean final score | Mean improvement |
| --- | ---: | ---: |
| `peer_critique` | 0.968698 | 0.017532 |
| `peer_critique_with_roles` | 0.968385 | 0.016900 |
| `self_critique` | 0.967954 | 0.018253 |

Per-agent final scores:

| Condition | Agent 0 | Agent 1 | Agent 2 |
| --- | ---: | ---: | ---: |
| `self_critique` | 0.958825 | 0.981545 | 0.963491 |
| `peer_critique` | 0.960494 | 0.981679 | 0.963922 |
| `peer_critique_with_roles` | 0.959024 | 0.982020 | 0.964112 |

Best individual recipe result:

- `smooth_l1_dynamics_pixel` with `dynamics_pixel_loss_weight=6.0` and
  `motion_prior_weight=1.5` reached `primary_score=0.982020`
  (`val_mse=0.018309`).

Examples of code changes:

```diff
-        return F.mse_loss(pred[:, 0], target)
+        return F.smooth_l1_loss(pred[:, 0], target)
```

```diff
-        actions = self.action_conditioning(frames, env_actions=env_actions, detach=True)
+        actions = self.action_conditioning(frames, env_actions=env_actions, detach=False)
...
-            actions.detach(),
+            actions,
```

## Analysis

The run demonstrates that the Codex-Scientist scaffold can execute real
multi-agent tree-search nodes with bounded source-level training changes, not
just scalar knob tweaks. Artifact completeness passed for all agents and all
nodes, and every patch recipe produced a real TinyWorlds metric.

The communication conditions were slightly ahead on final score in this seed,
with `peer_critique` on top and `peer_critique_with_roles` close behind.
However, the margin over self-critique was very small, so this should be treated
as integration evidence rather than a statistical result.

The clearest scientific signal was recipe-level: `smooth_l1_dynamics_pixel`
looked strong, `full_budget_action_supervision` improved the weaker fast-schedule
branch, and `sharpen_change_weights` underperformed its first-step parent. This
is useful for the cultural-evolution story because the population produced
heterogeneous ideas, and later runs can test whether peer communication
increases reuse or recombination of the better-performing ideas.

## Next Step

Run more seeds with live Codex-authored critiques and step-2 decisions. The key
test is no longer whether the infrastructure works; it is whether peer
communication causes agents to preferentially copy, mutate, or recombine
high-performing recipes like smooth-L1 dynamics while avoiding weaker recipes
like sharpened change weighting.
