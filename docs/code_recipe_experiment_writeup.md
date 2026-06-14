# Codex-Scientist Cultural-Lineage Experiment

## Motivation

The broader hypothesis is that automated science should improve when research
agents exchange useful information, analogous to cultural evolution: independent
branches explore different ideas, communication lets the population copy,
mutate, recombine, or reject those ideas, and later search should improve
relative to isolated self-critique.

Earlier TinyWorlds runs mostly varied scalar knobs. This experiment widened the
action surface to bounded architecture/training-loop changes and added explicit
lineage metadata. Step-2 actions declare whether they are `copy`, `mutate`,
`recombine`, `reject`, or `invent`, plus the source agents/nodes/recipes. This
lets us measure cultural transfer directly rather than only comparing final
scores.

## Setup

- Substrate: TinyWorlds world-model training on local Mac.
- Runner: `experiment.runner: codex_scientist`.
- Conditions: `self_critique`, `peer_critique`, `peer_critique_with_roles`.
- Agents: 3 per condition.
- Search depth: exactly 2 node experiments per agent.
- Budget: 120 seconds per node experiment.
- Parallelism: at most 3 node experiments at once.
- Seeds: 0, 1, 2, 4, 5.
- Output runs:
  - `runs/cultural_lineage_seed{seed}_self_critique`
  - `runs/cultural_lineage_seed{seed}_peer_critique`
  - `runs/cultural_lineage_seed{seed}_peer_critique_with_roles`

All nodes ran real TinyWorlds experiments in isolated workspaces. The original
TinyWorlds checkout was not modified. Artifact completeness passed for all runs.

The current experiment uses preauthored lineage actions to test the mechanism:
self-critique mutates or rejects only each agent's own prior branch, while peer
conditions allow cross-agent copy/recombination. This is therefore evidence about
the cultural-lineage mechanism and its measured effect under this action design,
not yet evidence that autonomous agents independently discover the same protocol.

## Action Surface

Agents could select one curated patch recipe:

- `action_grad_dynamics`: let action-tokenizer gradients flow into dynamics.
- `smooth_l1_dynamics_pixel`: replace dynamics pixel MSE with smooth-L1.
- `dynamics_first_schedule`: enter dynamics training earlier in short runs.
- `sharpen_change_weights`: focus change weights harder on changing patches.
- `full_budget_action_supervision`: keep action supervision active throughout training.

Each node saved `action.json`, `patch_result.json`, `code_diff.patch`,
`metrics.json`, lineage metadata, and logs.

The key peer transfer pattern was:

- Agent 0 copied agent 1's robust smooth-L1 dynamics recipe.
- Agent 1 mutated its own smooth-L1 recipe.
- Agent 2 recombined agent 0's action-gradient idea with agent 1's smooth-L1
  reconstruction idea, while rejecting its own fast-schedule branch.

## Results

Aggregate over seeds 0, 1, 2, 4, and 5:

| Condition | Mean final score | Mean improvement | Cross-agent transfer |
| --- | ---: | ---: | ---: |
| `self_critique` | 0.974884 | 0.019771 | 0.000000 |
| `peer_critique` | 0.981672 | 0.025250 | 0.666667 |
| `peer_critique_with_roles` | 0.981181 | 0.024143 | 0.666667 |

Peer improvement over self by seed:

| Seed | Peer - self | Peer+roles - self |
| ---: | ---: | ---: |
| 0 | +0.005907 | +0.005567 |
| 1 | +0.006094 | +0.004109 |
| 2 | +0.006488 | +0.006333 |
| 4 | +0.009720 | +0.009895 |
| 5 | +0.005730 | +0.005580 |

Cultural operator rates:

| Condition | Copy | Mutate | Recombine | Reject |
| --- | ---: | ---: | ---: | ---: |
| `self_critique` | 0.000000 | 0.666667 | 0.000000 | 0.333333 |
| `peer_critique` | 0.333333 | 0.333333 | 0.333333 | 0.000000 |
| `peer_critique_with_roles` | 0.333333 | 0.333333 | 0.333333 | 0.000000 |

The combined summary is stored at:

```text
runs/cultural_lineage_multiseed_summary_seed0_1_2_4_5.json
```

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

The main result is consistent across all five seeds: both peer conditions beat
self-critique on final score, and the peer advantage is paired with measured
cross-agent transfer. The plain peer condition had the strongest mean final
score (`0.981672`), followed by peer+roles (`0.981181`), then self (`0.974884`).
Peer also had the highest mean improvement from step 1 to step 2.

This is the first result that looks like the cultural-evolution mechanism we
wanted to test. In self-critique, agents can only mutate or reject their own
branches, and transfer is zero. In peer conditions, two thirds of agents perform
cross-agent transfer through copy or recombination, and those conditions improve
more. The strongest qualitative pattern is that the population learns to move
toward the robust `smooth_l1_dynamics_pixel` idea and away from the weaker
fast-schedule branch.

The roles condition did not clearly improve over plain peer critique. It still
beat self in every seed, but its mean final score was slightly below plain peer.
For this setup, role prompting may add structure without adding useful selection
pressure.

The most important caveat is that lineage actions were preauthored. This means
the experiment currently supports a narrower claim: when cross-agent cultural
transfer is available and used to copy/recombine a better branch, it improves
TinyWorlds outcomes relative to matched self-only revisions. The next stronger
claim requires live Codex agents or a noninteractive Codex backend to author the
copy/mutate/recombine decisions from the observed population summaries.

## Live-Authored Transfer Smoke

Seed 6 tested that stronger claim in supervised form. Step 1 used the same fixed
diverse inventions, then the run paused. Live Codex subagents inspected the
step-1 artifacts and authored critique, decision, and step-2 action override
files.

Results:

| Condition | Mean final score | Mean improvement | Cross-agent transfer |
| --- | ---: | ---: | ---: |
| `self_critique` | 0.969123 | 0.013166 | 0.000000 |
| `peer_critique` | 0.981815 | 0.025201 | 1.000000 |

Live peer choices:

- Agent 0 copied agent 1's top-ranked `smooth_l1_dynamics_pixel` branch.
- Agent 1 recombined its own smooth-L1 branch with agent 0's action-conditioning branch.
- Agent 2 copied agent 1's top-ranked smooth-L1 branch.

All three live peer decisions improved their parent branch. The live self-control
only improved one of three branches. This is still a single supervised seed, but
it is qualitatively stronger than the preauthored runs because the peer transfer
operators were selected after observing the population summary.

## Next Step

The next experiment should keep the same lineage metrics but make the transfer
decision live-authored:

1. Step 1: run diverse inventions.
2. Expose `population_summary_step_1.json` to live Codex critics.
3. Require each step-2 action to choose `copy`, `mutate`, `recombine`, `reject`,
   or `invent` with source ids.
4. Compare against the same self-only control over more seeds.

If live peer agents independently copy or recombine high-performing recipes and
continue beating self, that would be substantially stronger evidence for
cultural evolution in multi-agent automated research.
