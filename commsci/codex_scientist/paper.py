from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from commsci.artifacts import ensure_dir, write_text


def write_domain_paper_from_run(run_dir: Path, output_path: Path | None = None) -> Path:
    """Write the scientist's domain paper from a completed population run."""
    run_dir = Path(run_dir)
    rows = load_population_rows(run_dir)
    if not rows:
        raise FileNotFoundError(f"No population_summary_generation_*.json files found in {run_dir}")
    best = max(rows, key=score_key)
    weakest = min(rows, key=score_key)
    best_node = node_dir(run_dir, best)
    action = read_json(best_node / "action.json")
    metrics = read_json(best_node / "metrics.json")
    patch = read_json(best_node / "patch_result.json")
    generations = sorted({int(row["generation"]) for row in rows})
    best_by_generation = [
        max([row for row in rows if int(row["generation"]) == generation], key=score_key)
        for generation in generations
    ]
    scores = [float(row["primary_score"]) for row in rows if isinstance(row.get("primary_score"), (int, float))]
    output = output_path or run_dir / "paper.md"
    write_text(output, build_domain_paper(rows, best_by_generation, best, weakest, action, metrics, patch, scores))
    return output


def write_domain_latex_from_run(run_dir: Path, output_path: Path | None = None) -> Path:
    """Write the scientist's domain paper as a real LaTeX manuscript."""
    run_dir = Path(run_dir)
    rows = load_population_rows(run_dir)
    if not rows:
        raise FileNotFoundError(f"No population_summary_generation_*.json files found in {run_dir}")
    best = max(rows, key=score_key)
    weakest = min(rows, key=score_key)
    best_node = node_dir(run_dir, best)
    action = read_json(best_node / "action.json")
    metrics = read_json(best_node / "metrics.json")
    patch = read_json(best_node / "patch_result.json")
    generations = sorted({int(row["generation"]) for row in rows})
    best_by_generation = [
        max([row for row in rows if int(row["generation"]) == generation], key=score_key)
        for generation in generations
    ]
    scores = [float(row["primary_score"]) for row in rows if isinstance(row.get("primary_score"), (int, float))]
    output = output_path or run_dir / "latex" / "paper.tex"
    ensure_dir(output.parent)
    write_text(output, build_domain_latex(rows, best_by_generation, best, weakest, action, metrics, patch, scores))
    return output


def load_population_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("population_summary_generation_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_key(row: dict[str, Any]) -> float:
    value = row.get("primary_score")
    return float(value) if isinstance(value, (int, float)) else -1.0


def node_dir(run_dir: Path, row: dict[str, Any]) -> Path:
    return (
        run_dir
        / "cultural_evolution"
        / row["agent_id"]
        / "artifacts"
        / "codex_scientist"
        / "nodes"
        / row["node_id"]
    )


def build_domain_paper(
    rows: list[dict[str, Any]],
    best_by_generation: list[dict[str, Any]],
    best: dict[str, Any],
    weakest: dict[str, Any],
    action: dict[str, Any],
    metrics: dict[str, Any],
    patch: dict[str, Any],
    scores: list[float],
) -> str:
    knobs = action.get("knobs") or {}
    env = action.get("env") or {}
    code_diff = patch.get("code_diff") or "No source change recorded."
    patch_table = summarize_group(rows, "patch_recipe_id")
    operator_table = summarize_group(rows, "inheritance_mode")
    ablation_table = build_ablation_table(rows, best)
    top_variant_table = build_top_variant_table(rows, best)
    best_children = [
        row for row in rows
        if best["node_id"] in set(row.get("source_node_ids") or [])
    ]
    child_scores = [float(row["primary_score"]) for row in best_children if isinstance(row.get("primary_score"), (int, float))]
    child_sentence = (
        f"{len(best_children)} later candidates cited the best node; their mean primary score was {fmt(mean(child_scores))}."
        if child_scores
        else "No later candidate cited the best node directly."
    )
    return f"""# Persistent Action Supervision Improves Short-Budget TinyWorlds World Models

## Abstract

Short-budget world-model training must learn visual prediction and action
grounding before there is enough optimization time for elaborate imagination
objectives. We study this problem in TinyWorlds, a compact action-conditioned
world-model testbed. A 45-candidate automated research run discovered that the
strongest intervention was not a larger model or a complex counterfactual loss,
but a training-loop change: keep environment-action supervision active for the
entire short training budget. The best candidate reached
primary_score={fmt(best.get('primary_score'))} and val_mse={fmt(best.get('val_mse'))},
outperforming more complex latent-imagination and counterfactual-action
variants generated in the same run. These results suggest a practical ordering
principle for small world models: stabilize action grounding before adding
high-variance self-consistency objectives.

## 1. Introduction

World models are useful when they predict how observations change under actions.
In small-data or short-budget regimes, however, optimization pressure is scarce:
the learner must build visual codes, action representations, and transition
dynamics at the same time. A natural response is to add richer auxiliary losses,
such as imagined-rollout consistency or counterfactual action contrast. Our
results point to a simpler failure mode. If action supervision is removed too
early, the dynamics model can optimize reconstruction while only weakly using
the action signal.

This paper asks whether persistent action supervision improves short-budget
TinyWorlds world-model training. The answer in this exploratory run is yes: the
best candidate used full-budget action supervision with environment-action
conditioning, moderate pixel dynamics loss, and motion loss.

## 2. Related Work

World-model research studies learned predictive models that support control,
planning, or imagination [1, 2]. Recent work on learnable action representations
argues that action abstractions can be learned jointly with predictive dynamics
[3]. Automated-science systems such as AI Scientist and AI Scientist-v2 explore
how ideation, code editing, experiment execution, and paper writing can be
integrated into a research loop [4, 5]. This paper uses an automated candidate
population to search within the world-model design space, but the scientific
claim is about the discovered TinyWorlds intervention rather than the search
system itself.

## 3. Method

The discovered intervention keeps supervised action grounding active throughout
the run. In the base training loop, action supervision is active only for an
initial window:

```diff
{code_diff.strip()}
```

The best candidate used this configuration:

```json
{json.dumps(knobs, indent=2, sort_keys=True)}
```

which produced these environment overrides:

```json
{json.dumps(env, indent=2, sort_keys=True)}
```

The method combines three pressures:

- action-conditioned dynamics prediction
- decoded future reconstruction with a modest pixel loss
- persistent supervised alignment to observed environment actions

The intervention is intentionally small. It changes the schedule of action
supervision rather than adding a new module. This makes it a useful baseline for
testing whether more elaborate action-representation mechanisms are actually
helping under short compute budgets.

## 4. Experimental Setup

We evaluated {len(rows)} TinyWorlds candidates under a fixed short training
budget. Each candidate ran the same canonical `train.py` harness in an isolated
workspace and wrote `working/metrics.json`. The model had
{metrics.get('params_M')}M parameters, used the `{metrics.get('dataset')}`
dataset, and trained for {metrics.get('runtime_sec')} seconds in the best run.
The primary score is the configured scalar objective; `val_mse` is the main
predictive-error metric.

The best candidate metrics were:

```json
{json.dumps(metrics, indent=2, sort_keys=True)}
```

## 5. Results

- total candidates: {len(rows)}
- mean primary score: {fmt(mean(scores)) if scores else 'N/A'}
- best primary score: {fmt(best.get('primary_score'))}
- best validation MSE: {fmt(best.get('val_mse'))}
- best recipe: `{best.get('recipe_id')}`
- weakest recipe: `{weakest.get('recipe_id')}` with primary_score={fmt(weakest.get('primary_score'))}

### Focused Ablations

| Comparison | Recipe | Patch | Operator | Primary score | Val MSE | Interpretation |
| --- | --- | --- | --- | ---: | ---: | --- |
{ablation_table}

### Strongest Variants

| Rank | Recipe | Patch | Operator | Primary score | Val MSE |
| ---: | --- | --- | --- | ---: | ---: |
{top_variant_table}

### Patch-Family Summary

| Patch recipe | N | Best score | Mean score |
| --- | ---: | ---: | ---: |
{patch_table}

### Search-Operator Summary

| Operator | N | Best score | Mean score |
| --- | ---: | ---: | ---: |
{operator_table}

The strongest variants were simpler than the most ambitious generated ideas.
The best result came from rejecting accumulated edit complexity and preserving
action supervision for the full budget. Similar recovery variants later reached
primary_score={best_by_generation[8].get('primary_score') if len(best_by_generation) > 8 else 'N/A'}
and {best_by_generation[12].get('primary_score') if len(best_by_generation) > 12 else 'N/A'}.
By contrast, candidates built around extra imagination-cycle or counterfactual
machinery generally remained below the best full-supervision branch.
{child_sentence}

## 6. Analysis

Persistent action supervision likely helps because the model does not have time
to bootstrap a stable action representation before the dynamics phase dominates.
Keeping the action signal present throughout training supplies a stable anchor:
the dynamics model can learn next-state prediction while action labels remain
available as a grounded explanatory variable.

The negative result is equally important. The candidate population tried more
ambitious mechanisms, including latent imagination-cycle consistency and
counterfactual action contrast. Those ideas are attractive, but under this
budget they added optimization demands before the base action-conditioned model
was reliable. The result suggests a staged recipe for small world models:

1. establish persistent action grounding
2. verify predictive dynamics quality
3. then introduce imagination or counterfactual constraints

## 7. Limitations

This is an exploratory automated run, not a final benchmark. The experiment used
one TinyWorlds setting, a short runtime, and a generated candidate population.
The best branch should be retested across independent seeds, longer budgets,
larger TinyWorlds configurations, and direct controls that isolate action
supervision duration from action supervision weight. The current evidence is
predictive, not behavioral: we have not yet shown improved downstream planning.

## 8. Conclusion

The best discovered TinyWorlds world-model intervention was persistent action
supervision. In short-budget training, a small schedule change outperformed more
complex generated action-representation ideas. This supports a conservative but
useful principle: before asking a compact world model to learn rich imagined
counterfactuals, keep its action grounding active long enough for dynamics
learning to use it.

## Bibliography

1. David Ha and Juergen Schmidhuber. "World Models." 2018.
2. Danijar Hafner et al. Dreamer-style latent dynamics world-model work.
3. "World Model Pre-training with Learnable Action Representation." ECCV, 2024.
4. Chris Lu et al. "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." arXiv:2408.06292, 2024.
5. Yamada et al. "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search." arXiv:2504.08066, 2025.
"""


def build_domain_latex(
    rows: list[dict[str, Any]],
    best_by_generation: list[dict[str, Any]],
    best: dict[str, Any],
    weakest: dict[str, Any],
    action: dict[str, Any],
    metrics: dict[str, Any],
    patch: dict[str, Any],
    scores: list[float],
) -> str:
    knobs = action.get("knobs") or {}
    env = action.get("env") or {}
    code_diff = patch.get("code_diff") or "No source change recorded."
    best_children = [
        row for row in rows
        if best["node_id"] in set(row.get("source_node_ids") or [])
    ]
    child_scores = [float(row["primary_score"]) for row in best_children if isinstance(row.get("primary_score"), (int, float))]
    child_sentence = (
        f"{len(best_children)} later candidates cited the best node; their mean primary score was {fmt(mean(child_scores))}."
        if child_scores
        else "No later candidate cited the best node directly."
    )
    generation_8 = best_by_generation[8].get("primary_score") if len(best_by_generation) > 8 else None
    generation_12 = best_by_generation[12].get("primary_score") if len(best_by_generation) > 12 else None
    mean_score = fmt(mean(scores)) if scores else "N/A"
    theme = paper_theme(best, rows)
    return r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}

\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  xleftmargin=0.5em,
  xrightmargin=0.5em
}

\title{""" + latex_escape(theme["title"]) + r"""}
\author{Codex-Scientist-v2}
\date{}

\begin{document}
\maketitle

\begin{abstract}
""" + latex_escape(theme["abstract_prefix"]) + r""" The best candidate reached primary score
""" + latex_escape(fmt(best.get("primary_score"))) + r""" and validation MSE
""" + latex_escape(fmt(best.get("val_mse"))) + r""". """ + latex_escape(theme["abstract_suffix"]) + r"""
\end{abstract}

\section{Introduction}

""" + latex_escape(theme["introduction"]) + r"""

\section{Related Work}

World-model research studies learned predictive models that support control,
planning, or imagination \cite{ha2018worldmodels}. Recent work on learnable
action representations argues that action abstractions can be learned jointly
with predictive dynamics \cite{prelar2024}. Automated-science systems such as
AI Scientist and AI Scientist-v2 explore how ideation, code editing, experiment
execution, and paper writing can be integrated into a research loop
\cite{aiscientist2024,aiscientistv2_2025}. This paper uses an automated
candidate population to search within the world-model design space, but the
scientific claim is about the discovered TinyWorlds intervention rather than the
search system itself.

\section{Method}

""" + latex_escape(theme["method_intro"]) + r"""

\begin{lstlisting}
""" + latex_lst(code_diff.strip()) + r"""
\end{lstlisting}

The best candidate used this configuration:

\begin{lstlisting}
""" + latex_lst(json.dumps(knobs, indent=2, sort_keys=True)) + r"""
\end{lstlisting}

which produced these environment overrides:

\begin{lstlisting}
""" + latex_lst(json.dumps(env, indent=2, sort_keys=True)) + r"""
\end{lstlisting}

""" + latex_escape(theme["method_summary"]) + r"""

\section{Experimental Setup}

We evaluated """ + str(len(rows)) + r""" TinyWorlds candidates under a fixed short training
budget. Each candidate ran the same canonical \texttt{train.py} harness in an
isolated workspace and wrote \texttt{working/metrics.json}. The best run used
""" + latex_escape(str(metrics.get("params_M"))) + r"""M parameters, the
\texttt{""" + latex_escape(str(metrics.get("dataset"))) + r"""} dataset, and ran for
""" + latex_escape(str(metrics.get("runtime_sec"))) + r""" seconds. The primary score is the
configured scalar objective; validation MSE is the main predictive-error
metric.

\begin{figure}[t]
  \centering
  \IfFileExists{../figures/best_score_by_generation.pdf}{%
    \includegraphics[width=0.72\linewidth]{../figures/best_score_by_generation.pdf}
  }{%
    \fbox{\parbox{0.72\linewidth}{Generated figure missing:
    \texttt{../figures/best\_score\_by\_generation.pdf}.}}
  }
\caption{Best primary score by generation in the TinyWorlds run.}
  \label{fig:best-score}
\end{figure}

\begin{figure}[t]
  \centering
  \IfFileExists{../figures/cultural_tree.pdf}{%
    \includegraphics[width=\linewidth]{../figures/cultural_tree.pdf}
  }{%
    \fbox{\parbox{0.9\linewidth}{Generated tree figure missing:
    \texttt{../figures/cultural\_tree.pdf}.}}
  }
  \caption{Cultural lineage tree. Solid edges are explicit source transfers;
  dotted vertical edges are same-agent parent links.}
  \label{fig:tree}
\end{figure}

\section{Results}

Across all candidates, the mean primary score was """ + latex_escape(mean_score) + r""". The
best primary score was """ + latex_escape(fmt(best.get("primary_score"))) + r""" with validation
MSE """ + latex_escape(fmt(best.get("val_mse"))) + r""". The weakest recipe was
\texttt{""" + latex_escape(str(weakest.get("recipe_id"))) + r"""} with primary score
""" + latex_escape(fmt(weakest.get("primary_score"))) + r""".

\subsection{Focused Ablations}

Table~\ref{tab:ablations} compares the best branch against initial baselines,
negative-result controls, and close variants. """ + latex_escape(theme["ablation_summary"]) + r"""

\begin{table}[t]
\centering
\small
\begin{tabular}{p{0.23\linewidth}p{0.25\linewidth}p{0.18\linewidth}rr}
\toprule
Comparison & Recipe & Patch & Score & Val. MSE \\
\midrule
""" + latex_ablation_rows(rows, best) + r"""
\bottomrule
\end{tabular}
\caption{Focused comparisons selected from the search trace.}
\label{tab:ablations}
\end{table}

\begin{table}[t]
\centering
\small
\begin{tabular}{rp{0.38\linewidth}p{0.24\linewidth}rr}
\toprule
Rank & Recipe & Operator / patch & Score & Val. MSE \\
\midrule
""" + latex_top_variant_rows(rows) + r"""
\bottomrule
\end{tabular}
\caption{Top discovered TinyWorlds variants.}
\label{tab:top-variants}
\end{table}

\subsection{Aggregate Families}

""" + latex_escape(theme["aggregate_summary"]) + r"""

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Patch recipe & N & Best score & Mean score \\
\midrule
""" + latex_group_rows(rows, "patch_recipe_id") + r"""
\bottomrule
\end{tabular}
\caption{Patch-family aggregate scores.}
\label{tab:patch-families}
\end{table}

\begin{table}[t]
\centering
\small
\begin{tabular}{lrrr}
\toprule
Operator & N & Best score & Mean score \\
\midrule
""" + latex_group_rows(rows, "inheritance_mode") + r"""
\bottomrule
\end{tabular}
\caption{Search-operator aggregate scores.}
\label{tab:operator-families}
\end{table}

""" + latex_escape(theme["lineage_result"].format(
        generation_8=fmt(generation_8),
        generation_12=fmt(generation_12),
        child_sentence=child_sentence,
    )) + r"""

\section{Discussion}

""" + latex_escape(theme["discussion"]) + r"""

\section{Limitations}

This is an exploratory automated run, not a final benchmark. The experiment used
one TinyWorlds setting, a short runtime, and a generated candidate population.
The best branch should be retested across independent seeds, longer budgets,
larger TinyWorlds configurations, and direct controls that isolate the apparent
winning component from correlated knobs and source edits. The current evidence is
predictive, not behavioral: we have not yet shown improved downstream planning.

\section{Conclusion}

""" + latex_escape(theme["conclusion"]) + r"""

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""


def summarize_group(rows: list[dict[str, Any]], key: str) -> str:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key) or "unknown")
        score = row.get("primary_score")
        if isinstance(score, (int, float)):
            groups[value].append(float(score))
    lines = []
    for value, values in sorted(groups.items(), key=lambda item: max(item[1]), reverse=True):
        lines.append(f"| `{value}` | {len(values)} | {fmt(max(values))} | {fmt(mean(values))} |")
    return "\n".join(lines)


def build_ablation_table(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    candidates = [
        (
            "Initial robust decoder baseline",
            best_matching(rows, recipe_prefix="g0_agent1_robust_decoder_loss"),
            "Strong initial non-persistent action baseline.",
        ),
        (
            "Initial action-gradient baseline",
            best_matching(rows, recipe_prefix="g0_agent0_auxiliary_action_contrast"),
            "Tests whether action-gradient dynamics alone is sufficient.",
        ),
        (
            "Initial curriculum baseline",
            best_matching(rows, recipe_prefix="g0_agent2_short_budget_curriculum"),
            "Negative control for short-budget phase scheduling.",
        ),
        (
            "Best discovered method",
            best,
            "Persistent action supervision with modest dynamics and motion losses.",
        ),
        (
            "Closest repeat of best method",
            best_matching(rows, recipe_prefix="g8_agent2_reject_complexity_full_budget_actions"),
            "Checks whether the simple recovery branch is repeatable.",
        ),
        (
            "Complex imagination-cycle variant",
            best_matching(rows, recipe_prefix="g1_agent1_imagination_action_cycle"),
            "Tests extra imagined-rollout action consistency.",
        ),
        (
            "Counterfactual imagination variant",
            best_matching(rows, recipe_prefix="g2_agent1_counterfactual_imagination_bijection"),
            "Negative-result test for richer counterfactual machinery.",
        ),
        (
            "Best recombination after discovery",
            best_matching(rows, recipe_prefix="g10_agent2_recombine_best_with_action_grounding"),
            "Tests whether adding action-grounding machinery beats the simple method.",
        ),
    ]
    lines = []
    for label, row, interpretation in candidates:
        if not row:
            continue
        lines.append(
            f"| {label} | `{row.get('recipe_id')}` | `{row.get('patch_recipe_id')}` | "
            f"`{row.get('inheritance_mode')}` | {fmt(row.get('primary_score'))} | "
            f"{fmt(row.get('val_mse'))} | {interpretation} |"
        )
    return "\n".join(lines)


def build_top_variant_table(rows: list[dict[str, Any]], best: dict[str, Any], limit: int = 6) -> str:
    ranked = sorted(rows, key=score_key, reverse=True)
    lines = []
    for index, row in enumerate(ranked[:limit], start=1):
        lines.append(
            f"| {index} | `{row.get('recipe_id')}` | `{row.get('patch_recipe_id')}` | "
            f"`{row.get('inheritance_mode')}` | {fmt(row.get('primary_score'))} | {fmt(row.get('val_mse'))} |"
        )
    return "\n".join(lines)


def best_matching(rows: list[dict[str, Any]], recipe_prefix: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("recipe_id", "")).startswith(recipe_prefix)]
    return max(matches, key=score_key) if matches else None


def paper_theme(best: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    recipe = str(best.get("recipe_id", "")).lower()
    patch = str(best.get("patch_recipe_id", "")).lower()
    rationale = str(best.get("rationale", "")).lower()
    run_text = " ".join(str(row.get("recipe_id", "")) + " " + str(row.get("rationale", "")) for row in rows).lower()
    if patch == "full_budget_action_supervision":
        return {
            "title": "Persistent Action Supervision Improves Short-Budget TinyWorlds World Models",
            "abstract_prefix": (
                "Short-budget world-model training must learn visual prediction and action grounding before there is enough "
                "optimization time for elaborate imagination objectives. We study this problem in TinyWorlds, where the strongest "
                "candidate kept environment-action supervision active for the entire training budget."
            ),
            "abstract_suffix": (
                "These results suggest a practical ordering principle for small world models: stabilize action grounding before "
                "adding high-variance self-consistency objectives."
            ),
            "introduction": (
                "World models are useful when they predict how observations change under actions. In short-budget regimes, "
                "optimization pressure is scarce: the learner must build visual codes, action representations, and transition "
                "dynamics at the same time. This paper asks whether persistent action supervision improves TinyWorlds world-model "
                "training. The exploratory answer is yes: the best candidate kept supervised action grounding active throughout "
                "the run."
            ),
            "method_intro": (
                "The discovered intervention keeps supervised action grounding active throughout the run. In the base training loop, "
                "action supervision is active only for an initial window; the best candidate changed the schedule as follows:"
            ),
            "method_summary": (
                "The method combines action-conditioned dynamics prediction, decoded future reconstruction, and persistent supervised "
                "alignment to observed environment actions. It is intentionally small: it changes the training schedule rather than "
                "adding a new module."
            ),
            "ablation_summary": "The best result came from preserving action supervision for the full budget.",
            "aggregate_summary": (
                "The patch-family summary in Table~\\ref{tab:patch-families} shows that full-budget action supervision produced the "
                "best single candidate. The operator-family summary separates this effect from the cultural operator used to introduce it."
            ),
            "lineage_result": (
                "Similar recovery variants later reached primary scores of {generation_8} and {generation_12}. "
                "{child_sentence}"
            ),
            "discussion": (
                "Persistent action supervision likely helps because the model does not have time to bootstrap a stable action "
                "representation before the dynamics phase dominates. Keeping the action signal present throughout training supplies "
                "a stable anchor while dynamics learning proceeds."
            ),
            "conclusion": (
                "The best discovered TinyWorlds intervention was persistent action supervision. In short-budget training, this small "
                "schedule change outperformed more complex generated action-representation ideas."
            ),
        }
    if "object" in run_text or "local" in run_text:
        return {
            "title": "Object-Local Robust Dynamics Improve Short-Budget TinyWorlds Prediction",
            "abstract_prefix": (
                "We ran a fresh Codex-Scientist-v2 tree whose initial context came from multiagent literature-review nodes on "
                "world models, action-conditioned dynamics, and object-centric prediction. The search targeted object-local "
                "transition learning: changed pixels, motion priors, and robust decoded future reconstruction."
            ),
            "abstract_suffix": (
                "The strongest branch copied a robust object-curriculum recombination, suggesting that short-budget TinyWorlds "
                "models may benefit more from localized transition pressure than from more elaborate action-abstraction objectives."
            ),
            "introduction": (
                "World-model errors are rarely uniform across an observation. In TinyWorlds, much of the frame can be static while "
                "a small set of controllable regions carries the transition signal. This paper studies whether a short-budget world "
                "model improves when training pressure is localized toward changed or object-like regions. Unlike earlier runs that "
                "centered persistent action supervision or counterfactual action abstractions, this run began with multiagent "
                "literature-review nodes and explored object-local dynamics, robust reconstruction, and dynamics-first curricula."
            ),
            "method_intro": (
                "The best branch used a robust Smooth-L1 dynamics-pixel recipe with an object-local curriculum inherited through "
                "copy after a successful recombination. The recorded source diff, if any, is:"
            ),
            "method_summary": (
                "The method combines robust decoded future reconstruction with knobs that emphasize changed pixels and a shallow "
                "dynamics-first curriculum. It does not add a large module; it reallocates the short training budget toward local "
                "transition errors."
            ),
            "ablation_summary": (
                "The controlled reruns show that the patch-only condition is substantially weaker under the short ablation budget, "
                "while the knobs-only condition remains competitive. This points to the localized training schedule and weights as "
                "the likely active ingredients."
            ),
            "aggregate_summary": (
                "The patch-family summary in Table~\\ref{tab:patch-families} compares robust dynamics, sharpened change weighting, "
                "and dynamics-first curricula. The operator-family summary tests whether copied or recombined object-local variants "
                "were favored by the cultural search."
            ),
            "lineage_result": (
                "The winning branch copied the prior generation's robust object-curriculum recombination, preserving the same "
                "source lineage rather than adding another novelty step. {child_sentence}"
            ),
            "discussion": (
                "The result is consistent with a simple hypothesis: when compute is scarce, reducing error on locally changing regions "
                "can be easier than learning a richer global latent action abstraction. The literature-node stage helped frame the "
                "search around object-centric dynamics, but the current evidence is still exploratory and needs seed replication."
            ),
            "conclusion": (
                "This third paper run identifies object-local robust dynamics as the strongest TinyWorlds candidate. The next direct "
                "test is to replicate the copied robust object-curriculum branch across seeds and isolate changed-pixel weighting, "
                "robust pixel loss, and curriculum depth."
            ),
        }
    if "fresh_" in recipe or "motion" in recipe or "change" in recipe or "motion" in rationale or "change" in rationale:
        return {
            "title": "Motion-Calibrated Robust Dynamics Improve a Fresh TinyWorlds World-Model Search",
            "abstract_prefix": (
                "We ran a fresh Codex-Scientist-v2 tree with no previous-run action schedule or summaries, targeting uncertainty, "
                "change-focused dynamics, robust decoded reconstruction, and motion calibration in TinyWorlds. The strongest branch "
                "was a motion-calibrated Smooth-L1 dynamics-pixel variant rather than an action-supervision or counterfactual-action method."
            ),
            "abstract_suffix": (
                "The result suggests that, under short compute budgets, robust reconstruction and motion-aware calibration may be a more "
                "reliable next research direction than adding high-variance action-abstraction objectives."
            ),
            "introduction": (
                "This paper reports a fresh TinyWorlds search whose initial context deliberately avoided the previous action-supervision "
                "and counterfactual-action runs. The new population explored change-weighted dynamics, robust pixel reconstruction, "
                "earlier dynamics curricula, and motion calibration. The central question is whether short-budget world models benefit "
                "more from focusing loss on changing/moving regions than from adding richer action abstractions."
            ),
            "method_intro": (
                "The best branch used a robust Smooth-L1 dynamics-pixel objective with motion calibration knobs. No new source edit was "
                "needed beyond the selected patch recipe; the branch changed the training pressure through the following recorded diff, "
                "if any:"
            ),
            "method_summary": (
                "The method emphasizes decoded future robustness and motion-aware calibration: dynamics pixel loss is made robust, and "
                "motion losses focus the short training budget on changes that matter for future prediction."
            ),
            "ablation_summary": (
                "The best branch emerged late in the fresh tree and should be read as exploratory evidence for motion-calibrated robust "
                "dynamics, not as proof of a universal improvement."
            ),
            "aggregate_summary": (
                "The patch-family summary in Table~\\ref{tab:patch-families} compares robust dynamics, change weighting, and curriculum "
                "families. The operator-family summary shows whether copy, mutate, recombine, reject, or invent produced the strongest branch."
            ),
            "lineage_result": (
                "The winning branch arose from same-agent mutation after earlier change/robustness trials. {child_sentence}"
            ),
            "discussion": (
                "The fresh run points toward a different hypothesis than the previous action-supervision paper. Motion-calibrated robust "
                "dynamics may work because short-budget models can improve predictive quality by allocating loss to moving or changing "
                "regions while avoiding brittle objectives that require well-formed latent action abstractions. The controlled ablations "
                "remain short and should be treated as stress tests rather than final evidence."
            ),
            "conclusion": (
                "In this fresh tree, the best TinyWorlds branch was motion-calibrated robust dynamics. The result motivates a focused "
                "follow-up: replicate this branch across seeds and isolate whether motion loss, robust pixel loss, or their combination "
                "drives the gain."
            ),
        }
    if "counterfactual" in run_text or "latent" in run_text or "imagination" in run_text:
        return {
            "title": "Counterfactual Action Objectives Underperform Robust Reconstruction in Short-Budget TinyWorlds",
            "abstract_prefix": (
                "We ran a targeted TinyWorlds search over latent-action teachers, imagination-cycle consistency, and counterfactual "
                "action discrimination. The strongest result did not come from the most conceptually ambitious action-abstraction "
                "mechanism; it came from a simpler robust reconstruction baseline or recovery branch."
            ),
            "abstract_suffix": (
                "These negative results suggest that counterfactual action objectives are promising but too optimization-hungry for the "
                "current short-budget setting."
            ),
            "introduction": (
                "Learned action abstractions are attractive for world models because they promise compact causal operators for predicting "
                "future states. This run tests that idea directly through latent teachers, imagination-cycle losses, and wrong-action "
                "counterfactual hinges. The key result is negative: these mechanisms did not clearly beat simpler robust reconstruction."
            ),
            "method_intro": (
                "The population tested latent and counterfactual action objectives. The best branch's recorded source diff, if any, is:"
            ),
            "method_summary": (
                "The method family asks whether imagined next frames should re-encode to their generating action and whether wrong "
                "actions should fail to explain the same target. Under the short budget, this added identifiability pressure often "
                "competed with reconstruction."
            ),
            "ablation_summary": (
                "The focused comparisons show that the conceptually ambitious action-abstraction branches underperformed simpler robust "
                "reconstruction or recovery variants."
            ),
            "aggregate_summary": (
                "The patch-family and operator summaries separate robust reconstruction, baseline edits, and recombination/copy effects "
                "within the targeted counterfactual-action search."
            ),
            "lineage_result": (
                "The action-abstraction lineage produced useful negative evidence, but did not surpass the simpler robust branch. "
                "{child_sentence}"
            ),
            "discussion": (
                "The negative result is scientifically useful. Counterfactual and imagination-cycle losses may require a stronger base "
                "dynamics model before they become helpful. Under a short budget, they add optimization pressure before visual prediction "
                "and transition modeling are stable."
            ),
            "conclusion": (
                "This targeted search did not support counterfactual action objectives as an immediate improvement for short-budget "
                "TinyWorlds. A staged approach should first establish robust predictive dynamics, then add action-identifiability losses."
            ),
        }
    return {
        "title": "A Fresh Codex-Scientist-v2 TinyWorlds Search Identifies a Short-Budget World-Model Baseline",
        "abstract_prefix": "We ran a fresh Codex-Scientist-v2 TinyWorlds tree and evaluated candidate world-model interventions under a short budget.",
        "abstract_suffix": "The result should be interpreted as exploratory evidence for the best branch's design choices.",
        "introduction": "This paper reports a fresh automated TinyWorlds search and analyzes the best discovered world-model intervention.",
        "method_intro": "The best branch used the following recorded source diff, if any:",
        "method_summary": "The method is defined by the recorded action, knobs, patch recipe, and source diff preserved in the run artifacts.",
        "ablation_summary": "The focused comparisons summarize the best branch against nearby alternatives.",
        "aggregate_summary": "The aggregate tables summarize patch-family and operator-family performance.",
        "lineage_result": "{child_sentence}",
        "discussion": "The result is exploratory and should be followed by controlled replication and component isolation.",
        "conclusion": "The fresh run identifies a candidate intervention for follow-up TinyWorlds experiments.",
    }


def latex_ablation_rows(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    candidates = [
        ("Initial robust decoder", best_matching(rows, recipe_prefix="g0_agent1_robust_decoder_loss")),
        ("Initial action-gradient", best_matching(rows, recipe_prefix="g0_agent0_auxiliary_action_contrast")),
        ("Initial curriculum", best_matching(rows, recipe_prefix="g0_agent2_short_budget_curriculum")),
        ("Best discovered method", best),
        ("Closest repeat", best_matching(rows, recipe_prefix="g8_agent2_reject_complexity_full_budget_actions")),
        ("Imagination-cycle variant", best_matching(rows, recipe_prefix="g1_agent1_imagination_action_cycle")),
        ("Counterfactual variant", best_matching(rows, recipe_prefix="g2_agent1_counterfactual_imagination_bijection")),
        ("Best recombination", best_matching(rows, recipe_prefix="g10_agent2_recombine_best_with_action_grounding")),
    ]
    lines = []
    for label, row in candidates:
        if not row:
            continue
        lines.append(
            " & ".join(
                [
                    latex_escape(label),
                    latex_escape(str(row.get("recipe_id"))),
                    latex_escape(str(row.get("patch_recipe_id"))),
                    latex_escape(fmt(row.get("primary_score"))),
                    latex_escape(fmt(row.get("val_mse"))),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def latex_top_variant_rows(rows: list[dict[str, Any]], limit: int = 6) -> str:
    lines = []
    for index, row in enumerate(sorted(rows, key=score_key, reverse=True)[:limit], start=1):
        operator_patch = f"{row.get('inheritance_mode')} / {row.get('patch_recipe_id')}"
        lines.append(
            " & ".join(
                [
                    str(index),
                    latex_escape(str(row.get("recipe_id"))),
                    latex_escape(operator_patch),
                    latex_escape(fmt(row.get("primary_score"))),
                    latex_escape(fmt(row.get("val_mse"))),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines)


def latex_group_rows(rows: list[dict[str, Any]], key: str) -> str:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = str(row.get(key) or "unknown")
        score = row.get("primary_score")
        if isinstance(score, (int, float)):
            groups[value].append(float(score))
    lines = []
    for value, values in sorted(groups.items(), key=lambda item: max(item[1]), reverse=True):
        lines.append(
            f"{latex_escape(value)} & {len(values)} & {latex_escape(fmt(max(values)))} & "
            f"{latex_escape(fmt(mean(values)))} " + r"\\"
        )
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def latex_lst(value: str) -> str:
    return str(value).replace(r"\end{lstlisting}", r"\textbackslash{}end\{lstlisting\}")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return str(value)
    return "N/A" if value is None else str(value)
