# Codex-Scientist Autoresearch Doctrine

## Title

Cultural Evolution in Multi-Agent Automated Research Populations

## Keywords

automated science, cultural evolution, multi-agent search, tree search,
communication, memory, negative results, TinyWorlds, Codex-Scientist

## TL;DR

Codex-Scientist studies whether automated research improves when multiple
research agents explore in parallel, share bounded evidence, and deliberately
copy, mutate, recombine, reject, or invent ideas. Each agent must run real
experiments, preserve auditable lineage, and treat negative results as useful
selection pressure rather than failed work.

## Research Goal

The central hypothesis is that automated research populations can benefit from
cultural evolution. A single isolated agent can only improve by local search.
A population can additionally preserve discoveries, abandon weak branches,
combine partial successes, and spread useful procedures across agents.

The experiment should therefore test whether communication and lineage-aware
transfer improve research outcomes compared with matched self-only controls.
The goal is not merely to maximize one TinyWorlds score. The goal is to produce
evidence about whether multi-agent autoresearch populations can accumulate and
transmit useful research knowledge.

## What Counts As An Idea

An idea is a concrete, executable research intervention with a stated
hypothesis. It must include enough detail for another agent to rerun, copy, or
modify it.

Valid ideas may include:

- A TinyWorlds knob configuration.
- A curated patch recipe.
- Exact file edits against allowlisted source files.
- A training-loop change.
- An architecture change.
- A loss, auxiliary objective, regularizer, schedule, or evaluation change.
- A controlled rejection of a prior idea.

An idea is not valid if it is only a vague suggestion, a synthetic substitute
experiment, an unbounded rewrite, or a claim unsupported by artifacts.

## What Counts As Evidence

Evidence must be produced by the canonical experiment harness under the active
budget. For TinyWorlds runs, this means the node executes `train.py` through the
prepared runfile, writes `working/metrics.json`, and preserves logs.

Useful evidence includes:

- Primary metric score.
- Validation MSE or other configured task metrics.
- Runtime and success/failure status.
- Exact action metadata.
- Code diff or knob diff.
- Logs and traceback when a run fails.
- Comparison to the parent node and visible source nodes.
- Whether a copied, mutated, or recombined idea helped after transfer.

Negative evidence is still evidence. A failed patch, worse score, missing metric,
or brittle source edit should be recorded and used to guide rejection, debugging,
or tighter controls.

## Allowed Cultural Operators

Every non-initial node should declare exactly one cultural operator.

### Invent

Create a new idea without relying on a prior source node. Use this when the
population needs diversity or when no visible prior branch is worth preserving.

Required metadata:

- `inheritance_mode: "invent"`
- rationale for why the idea is worth trying

### Copy

Reproduce a prior idea with minimal intentional change. Use this when a source
node is strong enough that verifying or spreading it is more valuable than
adding novelty.

Required metadata:

- `inheritance_mode: "copy"`
- `source_agent_ids`
- `source_node_ids`
- `copied_recipe_id` when available

### Mutate

Change one prior idea in a bounded way. Use this when a source node is promising
but has a clear weakness, missing control, or obvious next variant.

Required metadata:

- `inheritance_mode: "mutate"`
- `source_agent_ids`
- `source_node_ids`
- rationale describing what changed and why

### Recombine

Combine components from two or more prior ideas. Use this when different source
nodes contain complementary mechanisms, such as one strong loss and one useful
training schedule.

Required metadata:

- `inheritance_mode: "recombine"`
- at least two source ids when possible
- `recombined_recipe_ids` when available
- rationale describing what each source contributed

### Reject

Explicitly move away from a weak, brittle, or overcomplicated prior idea. Reject
does not mean doing nothing. It means choosing a safer or more informative
alternative because the prior evidence was poor.

Required metadata:

- `inheritance_mode: "reject"`
- `source_agent_ids`
- `source_node_ids`
- `rejected_recipe_id` when available
- rationale describing the failure mode or risk being avoided

## Agent Behavior Rules

Agents should behave like bounded experimental scientists, not unconstrained
code generators.

- Run real experiments through the configured harness.
- Keep the compute budget fixed unless explicitly instructed otherwise.
- Prefer simple, testable hypotheses over sprawling changes.
- Make high-variance proposals only when they remain executable and auditable.
- Preserve exact lineage: source agents, source nodes, operator, rationale.
- Do not claim improvement without metrics.
- Do not hide failures.
- Do not alter the original TinyWorlds repository; edit only isolated node
  workspaces.
- Do not copy large datasets into every node workspace when shared data or
  symlinks are available.
- Keep paper-writing separate from evidence generation.
- Treat the communication checkpoint as the only cross-agent information path
  unless shared memory is explicitly enabled.

## Negative-Result Policy

Negative results are first-class outputs. A failed or worse branch can improve
the population by showing what to reject, simplify, or control.

When a result is negative, the agent should record:

- what failed or worsened
- whether the failure was conceptual, implementation-level, or budget-related
- whether the idea should be retried with a smaller change
- whether the population should reject it
- what artifact supports that conclusion

Do not convert a negative result into a success narrative. A useful rejection is
better than an unsupported positive claim.

## TinyWorlds-Specific Research Directions

TinyWorlds is currently the main substrate for Codex-Scientist smoke tests and
cultural-lineage experiments. Productive directions include:

- Dynamics-model loss design.
- Robust pixel or latent reconstruction objectives.
- Action-conditioned world modeling.
- Action-tokenizer supervision and action-cycle consistency.
- Training schedules for short compute budgets.
- Patch/change weighting for moving regions.
- Counterfactual or imagined rollout consistency.
- Simpler recovery branches that reject overcomplicated mechanisms.
- Controls that distinguish real metric gains from noise or budget artifacts.

Avoid directions that require long training, large new datasets, extensive video
generation, or manual inspection unless the run is explicitly configured for
that purpose.

## Paper-Generation Objective

A paper-oriented run should produce an auditable scientific case, not just a
leaderboard. In `codex_scientistv2`, the final paper should be a real LaTeX
workshop manuscript at `latex/paper.tex`, with `latex/references.bib` and a
compiled `latex/paper.pdf` when a LaTeX toolchain is available. Markdown may be
kept as a companion draft, but it should not be the primary paper artifact.

The final writeup should explain:

- the cultural-evolution hypothesis
- the experimental substrate
- the agent population and tree structure
- the communication or memory condition
- the operator distribution
- the best ideas and their lineage
- negative branches and rejected ideas
- quantitative results
- limitations and controls still needed

The paper should distinguish between evidence from one exploratory run and
evidence from replicated controls. It should not overclaim cultural evolution
unless transfer decisions were live-authored or otherwise independent of the
analysis.

For LaTeX paper runs, do not wrap Markdown inside `verbatim`. Convert the
manuscript into normal LaTeX sections, tables, figures, citations, captions,
and bibliography commands.

## Artifact Contract

Each Codex-Scientist node should preserve:

- `node.json`
- `action.json`
- `worker_task.md`
- `memory.md`
- `metrics.json`
- `logs.txt`
- `patch_result.json`
- `code_diff.patch` when code changed
- `branch_expansion.json`

Communication and two-step ablation runs should also preserve:

- `branch_summary.json`
- `critique.md`
- `critique.json`
- `decision_change.json`
- `metrics_experiment_1.json`
- `metrics_experiment_2.json`
- `logs_experiment_1.txt`
- `logs_experiment_2.txt`
- `research_note.md`
- `review.json`

Population runs should preserve:

- per-generation population summaries
- latest population summary
- lineage graph
- run config
- run spec
- final paper or research report when requested
- `latex/paper.tex` and `latex/references.bib` for `codex_scientistv2`
- `latex/paper.pdf` and `latex/compile.log` when LaTeX compilation is available

Artifacts should be complete enough that a later researcher can reconstruct
what each agent knew, what it changed, what command ran, what happened, and how
the result affected later branches.
