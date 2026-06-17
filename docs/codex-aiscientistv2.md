# Codex-AI-Scientist-v2 Research Brief

## Title

Cultural Evolution for Automated World-Model Research

## Keywords

automated scientific discovery, AI Scientist, Codex-Scientist, cultural
evolution, multi-agent systems, world models, action-conditioned dynamics,
negative results, open-ended search

## TL;DR

This run should behave less like a knob sweep and more like a miniature
scientific community. Agents should propose unusual but executable research
ideas, relate them to the current literature, run real TinyWorlds experiments,
share explicit lineage, and produce a complete LaTeX workshop paper with
abstract, background, motivation, experiments, results, conclusion, and
bibliography.

## Research Goal

The main scientific goal is to test whether cultural evolution improves
automated research: multiple agents explore different ideas, observe evidence,
and then copy, mutate, recombine, reject, or invent follow-up ideas. The
secondary goal is to discover paper-worthy TinyWorlds world-model interventions
that are more conceptually interesting than scalar hyperparameter tuning.

The final paper should make two claims carefully:

- a method claim about Codex-Scientist as a supervised multi-agent automated
  research system
- an empirical claim about which TinyWorlds ideas worked under the bounded run

## Literature-Grounded Starting Points

Agents should use the following literature themes as inspiration, not as rigid
templates:

- AI-Scientist-style automation: ideation, implementation, execution, paper
  writing, and review can be joined into an automated research loop.
- Agentic scientific discovery: autonomous or semi-autonomous agents need
  explicit evaluation, provenance, and failure handling.
- World models: useful internal models predict future states under actions and
  support planning or imagination.
- Learnable action representations: world models can benefit when actions are
  represented as learned latent transformations rather than only external labels.
- Collective intelligence and cultural evolution: populations can improve by
  preserving, transmitting, and recombining useful ideas.

## What Counts As A Valid Idea

A valid idea should be conceptually distinctive and executable. Prefer ideas
that could plausibly become the central contribution of a short paper.

Strong ideas include:

- a new auxiliary objective
- an action-world consistency constraint
- an imagination or counterfactual rollout mechanism
- a learned action representation intervention
- a causal or invariance-inspired loss
- a memory or cultural-transfer mechanism
- a negative-result test that rejects an attractive but brittle idea
- a compact architecture change to `models.py`
- a training-loop change to `train.py`

Weak ideas include:

- only changing one scalar knob without a scientific rationale
- adding complexity that cannot run under the budget
- replacing the TinyWorlds harness
- claiming novelty without citing a lineage or literature theme
- producing an idea that cannot be audited from `action.json` and `code_diff.patch`

## Cultural Operators

Every non-initial node must choose one operator.

- `invent`: create a new literature-inspired idea.
- `copy`: replicate a strong previous idea to test whether it transfers.
- `mutate`: change one previous idea in a controlled way.
- `recombine`: combine mechanisms from multiple source nodes.
- `reject`: explicitly abandon a weak or overcomplicated source and test a safer
  alternative.

The action must record `source_agent_ids`, `source_node_ids`, and rationale when
using any operator other than `invent`.

## Agent Behavior Rules

Agents should:

- propose bold ideas, but keep the patch bounded and runnable
- explain the literature analogy behind the idea
- run the canonical TinyWorlds harness
- preserve exact lineage and source ids
- keep all source edits within allowlisted files
- treat failures and worse metrics as real evidence
- prefer a clean negative result over a vague positive story
- avoid hidden state outside artifacts

Agents should not:

- perform arbitrary repository rewrites
- invent synthetic experiments
- depend on unavailable datasets or long training
- optimize only for short-run metric noise
- make claims the artifacts cannot support

## TinyWorlds Research Directions

Prioritize ideas like:

- action-cycle consistency: predicted next frames should imply the same action
  that conditioned them
- latent displacement objectives: actions as transformations in latent space
- counterfactual spread: alternative actions should produce distinguishable
  futures
- robustness: smooth losses or uncertainty-aware objectives for decoded futures
- causal invariance: stable factors should remain stable while controllable
  factors change
- curriculum: allocate scarce compute to the phases most relevant to dynamics
- failure rejection: simplify overcomplex branches that lower the score

## Workshop LaTeX Paper Objective

At the end of the run, write the scientist's domain paper in a compact workshop
format as a real LaTeX manuscript. The paper should be about the discovered
World Models idea, not about the Codex-Scientist orchestration unless that
context is needed for provenance.

The primary paper output is:

- `latex/paper.tex`

When a LaTeX toolchain is available, also compile:

- `latex/paper.pdf`
- `latex/compile.log`

Markdown may be written as a working draft or companion artifact, but it is not
the final paper. Do not make `paper.tex` a wrapper that embeds Markdown in a
`verbatim` block. Convert the manuscript into real LaTeX sections, tables,
figures, citations, captions, and bibliography entries.

Use a workshop-style LaTeX structure:

- Title
- Abstract
- Introduction
- Related Work
- Methods
- Experimental Setup
- Results
- Ablations
- Discussion
- Limitations
- Conclusion
- Bibliography

The LaTeX paper must include:

- `\title{...}` and `\author{...}`
- `\begin{abstract}...\end{abstract}`
- numbered sections using `\section{...}` and `\subsection{...}`
- at least one focused ablation table using `booktabs`
- at least one figure include for a generated plot when plots exist
- citations with `\cite{...}` backed by `latex/references.bib`
- a limitations section that distinguishes exploratory evidence from controlled
  evidence
- a bibliography command such as `\bibliography{references}`

The main Results section must not dump every generated candidate. It should
include focused ablations and comparisons:

- base or initial baselines
- best discovered method
- closest copy/mutate/recombine variants
- explicit negative-result controls
- patch-family or operator-family aggregates only when they clarify the claim

The full search trace belongs in a separate `search_report.md` or lineage
artifact. The paper may mention that an automated candidate population produced
the intervention, but it should read like a workshop paper about the scientific
idea. It must distinguish exploratory evidence from controlled evidence, avoid
overclaiming, and state the next direct ablation needed.

## Paper Artifact Contract

A successful Codex-Scientist-v2 paper run should preserve:

- `latex/paper.tex`: the primary final manuscript
- `latex/references.bib`: bibliography used by the manuscript
- `latex/paper.pdf`: compiled output when the environment has LaTeX installed
- `latex/compile.log`: compiler output or a clear explanation of why compile
  was skipped
- `paper.md`: optional readable Markdown companion, not the primary manuscript
- `figures/`: generated plots used by the LaTeX manuscript
- `codex_scientistv2/ablation_report.json`: evidence source for tables
- `codex_scientistv2/rich_nodes.jsonl`: auditable node evidence
- `codex_scientistv2/codex_tasks/paper_reflection.md`: follow-up instruction
  for improving the LaTeX paper without inventing results

## Bibliography Seeds

- Lu, C. et al. "The AI Scientist: Towards Fully Automated Open-Ended
  Scientific Discovery." arXiv:2408.06292, 2024.
- Sakana AI. "The AI Scientist." 2024.
- Yamada et al. "The AI Scientist-v2: Workshop-Level Automated Scientific
  Discovery via Agentic Tree Search." arXiv:2504.08066, 2025.
- "Agentic AI for Scientific Discovery: A Survey of Progress, Challenges, and
  Future Directions." arXiv:2503.08979, 2025.
- "World Model Pre-training with Learnable Action Representation." ECCV, 2024.
- Ha and Schmidhuber. "World Models." 2018.
- Hafner et al. "Dreamer" and later latent-dynamics world-model work.
- Henrich and related cultural-evolution work on cumulative culture and
  transmission.
- Recent collective-intelligence and multi-agent LLM work studying how agent
  interaction patterns affect group performance.
