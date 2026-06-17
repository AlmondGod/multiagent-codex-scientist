# Peer Critique Search

Peer Critique Search is an experimental harness for asking a narrow question:

> Under the same model, token, and experiment budgets, does critique from a
> different tree-search worker improve TinyWorlds research decisions more than
> a worker critiquing itself?

The project is inspired by AI-Scientist-v2 style agentic tree search, but it is
not a drop-in replacement for AI-Scientist-v2. This repo focuses on controlled
communication experiments between isolated research workers. Each worker gets
its own TinyWorlds workspace, proposes and runs a bounded experiment, receives a
critique signal, then chooses a follow-up experiment. The harness records enough
artifacts to compare self-critique, peer critique, and role-conditioned peer
critique under matched budgets.

The current codebase also includes two Codex-Scientist runners for supervised
multi-agent research runs and paper-oriented TinyWorlds discovery runs. These
paths keep the same basic idea: preserve isolated worker state, make
communication explicit, and write durable artifacts that can be inspected after
the run.

## Table of Contents

- [What This Repo Does](#what-this-repo-does)
- [Safety and Scope](#safety-and-scope)
- [Requirements](#requirements)
- [Quick Start: Local Dry Run](#quick-start-local-dry-run)
- [Run Modes](#run-modes)
- [Running Real TinyWorlds Experiments](#running-real-tinyworlds-experiments)
- [Codex-Scientist Live Runs](#codex-scientist-live-runs)
- [Codex-Scientist-v2 Paper Runs](#codex-scientist-v2-paper-runs)
- [Aggregation](#aggregation)
- [Artifacts](#artifacts)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Related Docs](#related-docs)

## What This Repo Does

The harness compares three communication conditions:

- `self_critique`: each worker critiques its own branch summary.
- `peer_critique`: another worker critiques the target branch summary using the
  same prompt and budget.
- `peer_critique_with_roles`: another worker critiques the branch with a
  deterministic role prior: critic, ablator, optimizer, or explorer.

Communication happens exactly once: after the first experiment summaries are
written and before the second experiment plans are generated. There is no
pre-experiment chat, shared editing, group discussion, or post-hoc strategy
extraction.

The default reduced-compute setting uses three agents, two experiment
meta-steps per agent, and six total TinyWorlds executions per condition. The
code keeps budgets matched across conditions: number of agents, experiment
count, training-step budget, runtime limit, critique template, prompt/completion
token limits, temperature, and model backend.

## Safety and Scope

Some run modes execute LLM-written or LLM-selected code in copied TinyWorlds
workspaces. Use a sandboxed environment for real runs, especially when enabling
live Codex workers, external model calls, web-backed literature retrieval, or
AI-Scientist-v2 integrations.

The repo CLI owns experiment execution, workspace isolation, metric parsing,
artifact saving, and aggregation. It does not currently start Codex desktop
subagents by itself from a noninteractive shell. Supervised Codex-Scientist
runs are meant to be launched from an active Codex session where a user can
spawn workers, collect JSON artifacts, and let the CLI validate and execute the
bounded actions.

## Requirements

For local dry runs:

- Python 3.10 or newer
- `pyyaml`

For real TinyWorlds runs:

- a TinyWorlds or `tinyworlds-autoresearch` checkout
- the dependencies required by that TinyWorlds checkout
- a model endpoint if using non-mock critique/model calls
- Linux with NVIDIA CUDA/PyTorch for GPU-backed TinyWorlds or AI-Scientist-v2
  experiments, depending on the external project setup

For AI-Scientist-v2-backed runs:

- an AI-Scientist-v2 checkout, or network access so this repo can clone
  `https://github.com/SakanaAI/AI-Scientist-v2` into `external/AI-Scientist-v2`
- model credentials and dependencies expected by AI-Scientist-v2

Minimal local setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pyyaml
```

If you run the external TinyWorlds or AI-Scientist-v2 paths, install their
requirements in the same environment or in the environment used by their train
and eval commands.

## Quick Start: Local Dry Run

Dry-run mode is the fastest way to understand the artifact contract. It does
not require CUDA, TinyWorlds, AI-Scientist-v2, a local model server, or paid API
keys. It writes deterministic fake metrics and critiques so the orchestration
can be inspected locally.

Run one condition:

```bash
python3 run_ablation.py \
  --config configs/v0_mac.yaml \
  --condition self_critique \
  --dry_run \
  --output_dir runs/test_self \
  --seed 0
```

Run all three communication conditions into one run directory:

```bash
for condition in self_critique peer_critique peer_critique_with_roles; do
  python3 run_ablation.py \
    --config configs/v0_mac.yaml \
    --condition "$condition" \
    --dry_run \
    --output_dir runs/v0_dry \
    --seed 0
done
```

Aggregate the run:

```bash
python3 aggregate_results.py --run_dir runs/v0_dry
```

## Run Modes

This repository has three main execution paths.

### `tinyworlds_command`

The basic runner used by `run_ablation.py`. In dry-run mode it fabricates
metrics. In real mode it copies a TinyWorlds workspace per agent, runs the
configured train/eval commands, parses metrics, writes critiques, and aggregates
the results.

### `ai_scientist_v2`

An integration mode that delegates branch expansion to AI-Scientist-v2's
best-first tree-search loop with a reduced config. The communication wrapper
adds the checkpoint between two branch expansions.

Use this mode only after you have a working AI-Scientist-v2 environment and
model backend.

### `codex_scientist`

A supervised Codex-oriented runner. AI-Scientist-v2 remains a conceptual
reference, but this path does not depend on AI-Scientist-v2's LLM/code loop.
Each tree node gets an isolated TinyWorlds workspace, a canonical `runfile.py`,
node-local memory, selected action metadata, execution logs, metrics, and a
`node.json`.

For multi-agent conditions, the coordinator can launch node experiments in
parallel while keeping artifacts centralized.

## Running Real TinyWorlds Experiments

Real execution requires a TinyWorlds checkout and configured commands. The
example below uses an OpenAI-compatible local model endpoint for critiques.

```bash
python3 run_ablation.py \
  --config configs/v0_mac.yaml \
  --condition peer_critique \
  --tinyworlds_dir /path/to/tinyworlds \
  --ai_scientist_v2_dir /path/to/AI-Scientist-v2 \
  --model_url http://localhost:1234/v1 \
  --model_name qwen3-coder-30b-a3b-instruct \
  --train_command "python train.py --steps 1000" \
  --eval_command "python eval.py" \
  --output_dir runs/real_peer \
  --max_training_steps 1000 \
  --max_runtime_minutes_per_experiment 20 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

If `--ai_scientist_v2_dir` is omitted in real mode, the wrapper attempts to
clone AI-Scientist-v2 into `external/AI-Scientist-v2`. If cloning fails, the run
exits with a setup message.

### A100 Smoke Runs

`configs/a100_smoke.yaml` is for integration only, not final results. It runs a
very small TinyWorlds update, parses `Step N Loss: ...` from logs as
`reconstruction_loss`, and derives:

```text
primary_score = 1 / (1 + reconstruction_loss)
```

Run the self-critique smoke first:

```bash
python3 run_ablation.py \
  --condition self_critique \
  --num_agents 1 \
  --tinyworlds_dir /workspace/tinyworlds \
  --ai_scientist_v2_dir /workspace/AI-Scientist-v2 \
  --output_dir runs/a100_real_self_smoke \
  --config configs/a100_smoke.yaml \
  --max_training_steps 100 \
  --max_runtime_minutes_per_experiment 10 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

Then run the smallest peer smoke:

```bash
python3 run_ablation.py \
  --condition peer_critique \
  --num_agents 2 \
  --tinyworlds_dir /workspace/tinyworlds \
  --ai_scientist_v2_dir /workspace/AI-Scientist-v2 \
  --output_dir runs/a100_real_peer_smoke \
  --config configs/a100_smoke.yaml \
  --max_training_steps 100 \
  --max_runtime_minutes_per_experiment 10 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

Stop after this peer smoke unless you are intentionally running the full
ablation.

### AI-Scientist-v2 Branch Smoke

`configs/a100_ai_scientist_smoke.yaml` opts into:

```yaml
experiment:
  runner: ai_scientist_v2
```

In this mode, each communication meta-step delegates branch expansion to
AI-Scientist-v2's best-first tree-search loop with a reduced config: one
worker, one draft, one step, and no report generation. The communication
wrapper only adds the checkpoint between two AI-Scientist branch expansions.

This mode expects the TinyWorlds autoresearch harness, not the full TinyWorlds
repo:

```text
/workspace/tinyworlds-autoresearch/train.py
/workspace/tinyworlds-autoresearch/models.py
/workspace/tinyworlds-autoresearch/setup.py
```

Run only after a model backend is available for AI-Scientist-v2 code and
feedback calls. The default config uses Ollama-style model names because
AI-Scientist-v2's tree-search backend supports OpenAI and Ollama paths
directly.

Start with self-critique:

```bash
python3 run_ablation.py \
  --condition self_critique \
  --num_agents 1 \
  --tinyworlds_dir /workspace/tinyworlds-autoresearch \
  --ai_scientist_v2_dir /workspace/AI-Scientist-v2 \
  --output_dir runs/a100_ai_self_smoke \
  --config configs/a100_ai_scientist_smoke.yaml \
  --max_runtime_minutes_per_experiment 10 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

Then run the two-agent peer smoke:

```bash
python3 run_ablation.py \
  --condition peer_critique \
  --num_agents 2 \
  --tinyworlds_dir /workspace/tinyworlds-autoresearch \
  --ai_scientist_v2_dir /workspace/AI-Scientist-v2 \
  --output_dir runs/a100_ai_peer_smoke \
  --config configs/a100_ai_scientist_smoke.yaml \
  --max_runtime_minutes_per_experiment 10 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

## Codex-Scientist Live Runs

`configs/a100_codex_scientist_smoke.yaml` opts into:

```yaml
experiment:
  runner: codex_scientist
```

The canonical instructions for supervised Codex-Scientist agents are in
`docs/codex_scientist_autoresearch.md`. That document defines the research
goal, evidence standards, cultural operators, negative-result policy,
TinyWorlds directions, paper objective, and artifact contract.

Minimal smoke:

```bash
python3 run_ablation.py \
  --runner codex_scientist \
  --config configs/a100_codex_scientist_smoke.yaml \
  --condition peer_critique \
  --num_agents 2 \
  --tinyworlds_dir /Users/almondgod/Repositories/tinyworlds-autoresearch \
  --output_dir runs/codex_scientist_peer_smoke \
  --max_runtime_minutes_per_experiment 2 \
  --max_tokens_per_critique 1000 \
  --write_full_paper false \
  --seed 0
```

Live override files let supervised Codex workers choose bounded actions,
critiques, and second-step decisions.

Action override files live in
`experiment.codex_scientist_action_overrides_dir`:

```text
agent_0_step_1.json
agent_1_step_1.json
agent_0_step_2.json
agent_1_step_2.json
```

Example action override:

```json
{
  "recipe_id": "agent_0_step_2_action_conditioning",
  "knobs": {"use_env_actions": 1, "action_supervision_weight": 0.5},
  "rationale": "Live Codex worker selected a controlled action-conditioning follow-up."
}
```

Action overrides may include one bounded source patch recipe. The coordinator
applies these patches only inside the per-node isolated TinyWorlds workspace,
writes `patch_result.json` and `code_diff.patch`, and never modifies the
original TinyWorlds checkout.

Supported patch recipes:

- `baseline_no_patch`
- `dynamics_first_schedule`
- `action_grad_dynamics`
- `smooth_l1_dynamics_pixel`
- `sharpen_change_weights`
- `full_budget_action_supervision`

For cultural-evolution runs, every step-2 action should declare its inheritance
operator:

- `copy`
- `mutate`
- `recombine`
- `reject`
- `invent`

Peer conditions write `population_summary_step_1.json` and include a compact
population scoreboard in critique prompts when
`experiment.codex_scientist_population_context: true`.

## Codex-Scientist-v2 Paper Runs

`codex_scientistv2` is the expanded paper-oriented runner. It keeps the cultural
operators and TinyWorlds execution path, then adds AI-Scientist-v2 style stages:
richer node storage, multiagent literature-review nodes, focused ablations, plot
aggregation, live literature retrieval with BibTeX refresh, Codex task prompts,
LaTeX manuscript generation, compile attempts, and automated text/figure
reviews.

For longer paper-oriented runs, `docs/codex-aiscientistv2.md` shifts the brief
toward literature-inspired world-model ideas and complete manuscript
generation. In that mode, `paper.md` is the scientist's domain paper about the
discovered World Models intervention, and `search_report.md` is the separate
meta-report about the Codex-Scientist search process.

One-generation Mac smoke:

```bash
python3 scripts/run_codex_scientistv2.py \
  --config configs/codex_scientistv2_mac.yaml
```

Postprocess an existing cultural run without rerunning experiments:

```bash
python3 scripts/run_codex_scientistv2.py \
  --config configs/codex_scientistv2_mac.yaml \
  --skip_experiments \
  --output_dir runs/codex_aiscientistv2_paper_seed1 \
  --doctrine_doc docs/codex-aiscientistv2.md
```

By default, a non-`--skip_experiments` `codex_scientistv2` run executes:

- one initial literature-review node per agent, run in parallel through
  Semantic Scholar with arXiv fallback
- literature-node summaries exposed as the first experimental generation's
  context
- configured cultural population search
- bounded controlled ablations around the best branch
- final best-idea literature retrieval through Semantic Scholar with arXiv
  fallback, merged with the initial literature-node references
- figure and tree generation
- `latex/paper.tex`, `latex/references.bib`, and `latex/compile.log`
- local automated paper and figure reviews

## Aggregation

Aggregation writes three top-level outputs:

- `results.csv`
- `results.json`
- `summary.md`

Run:

```bash
python3 aggregate_results.py --run_dir runs/v0_dry
```

The aggregation compares useful decision changes, fraction of decisions
changed, fraction later helped, communication value, TinyWorlds metric
improvement, final score, unsupported claim count, duplicate experiment rate,
ablation quality, failure avoidance, success rate, reviewer score, runtime, and
token usage.

## Artifacts

Each standard ablation agent writes:

- `config.yaml`
- `prompts/`
- `completions/`
- `hypothesis.md`
- `experiment_plan_1.md`
- `branch_summary.json`
- `metrics_experiment_1.json`
- `logs_experiment_1.txt`
- `critique.md`
- `critique.json`
- `decision_change.json`
- `experiment_plan_2.md`
- `metrics_experiment_2.json`
- `logs_experiment_2.txt`
- `git_diff.patch`
- `research_note.md`
- `review.json`
- `metadata.json`
- `artifact_completeness.json` for Codex-Scientist runs

Workspaces are isolated under:

```text
runs/{run_id}/{condition}/agent_{i}/workspace/
runs/{run_id}/{condition}/agent_{i}/artifacts/
runs/{run_id}/global/
```

Every Codex-Scientist node writes:

- `codex_scientist/nodes/{node_id}/node.json`
- `action.json`
- `worker_task.md`
- `memory.md`
- `metrics.json`
- `logs.txt`
- `patch_result.json`
- `code_diff.patch` when a source patch recipe changed files
- `branch_expansion.json`

## Frequently Asked Questions

### Is this a full AI-Scientist-v2 implementation?

No. The core ablation harness uses AI-Scientist-v2 as a reference point for
tree-search research automation, but the main question is about communication
between workers. The `ai_scientist_v2` runner can delegate branch expansion to
AI-Scientist-v2 when that external environment is available.

### Can I run this on a Mac?

Yes for dry runs and some Codex-Scientist-v2 postprocessing paths. Real
TinyWorlds training may require Linux, CUDA, and the dependencies of your
TinyWorlds checkout.

### Why do dry runs produce metrics without TinyWorlds?

Dry runs are for validating orchestration, artifact structure, aggregation, and
prompt plumbing. They are not scientific results.

### What should I run first on a GPU machine?

Start with the one-agent self-critique smoke. If it passes, run the two-agent
peer-critique smoke. Only then scale to full ablations.

### Where do live Codex worker decisions enter the run?

Through override directories configured in YAML. The CLI accepts bounded action
JSON, critique text or JSON, and decision JSON. If no live override exists, the
coordinator falls back to local Codex-Scientist critique/decision logic for
unattended smoke runs.

### Does the CLI modify my original TinyWorlds checkout?

The Codex-Scientist patch recipes are applied only inside isolated per-node
workspaces. The original TinyWorlds checkout is copied or referenced as a
source and is not patched by those recipes.

## Related Docs

- `docs/codex_scientist_autoresearch.md`: canonical supervised
  Codex-Scientist agent instructions.
- `docs/codex-aiscientistv2.md`: paper-oriented Codex-Scientist-v2 doctrine.
- `docs/live_seed7_peer_tree.md`: example live peer-tree notes.
- `docs/code_recipe_experiment_writeup.md`: notes on source patch recipes and
  experiment writeups.
