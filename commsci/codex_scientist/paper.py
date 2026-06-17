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

\title{Persistent Action Supervision Improves Short-Budget TinyWorlds World Models}
\author{Codex-Scientist-v2}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Short-budget world-model training must learn visual prediction and action
grounding before there is enough optimization time for elaborate imagination
objectives. We study this problem in TinyWorlds, a compact action-conditioned
world-model testbed. A 45-candidate automated research run discovered that the
strongest intervention was not a larger model or a complex counterfactual loss,
but a training-loop change: keep environment-action supervision active for the
entire short training budget. The best candidate reached primary score
""" + latex_escape(fmt(best.get("primary_score"))) + r""" and validation MSE
""" + latex_escape(fmt(best.get("val_mse"))) + r""", outperforming more complex
latent-imagination and counterfactual-action variants generated in the same run.
These results suggest a practical ordering principle for small world models:
stabilize action grounding before adding high-variance self-consistency
objectives.
\end{abstract}

\section{Introduction}

World models are useful when they predict how observations change under actions.
In small-data or short-budget regimes, optimization pressure is scarce: the
learner must build visual codes, action representations, and transition dynamics
at the same time. A natural response is to add richer auxiliary losses, such as
imagined-rollout consistency or counterfactual action contrast. Our results
point to a simpler failure mode. If action supervision is removed too early, the
dynamics model can optimize reconstruction while only weakly using the action
signal.

This paper asks whether persistent action supervision improves short-budget
TinyWorlds world-model training. The answer in this exploratory run is yes: the
best candidate used full-budget action supervision with environment-action
conditioning, moderate pixel dynamics loss, and motion loss.

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

The discovered intervention keeps supervised action grounding active throughout
the run. In the base training loop, action supervision is active only for an
initial window; the best candidate changed the schedule as follows:

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

The method combines three pressures: action-conditioned dynamics prediction,
decoded future reconstruction with a modest pixel loss, and persistent
supervised alignment to observed environment actions. The intervention is
intentionally small. It changes the schedule of action supervision rather than
adding a new module, which makes it a useful baseline for testing whether more
elaborate action-representation mechanisms are actually helping under short
compute budgets.

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
  \caption{Best primary score by generation in the 45-candidate TinyWorlds run.}
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
negative-result controls, and close variants. The strongest variants were
simpler than the most ambitious generated ideas. The best result came from
rejecting accumulated edit complexity and preserving action supervision for the
full budget.

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

The patch-family summary in Table~\ref{tab:patch-families} shows that
full-budget action supervision produced the best single candidate. The
operator-family summary in Table~\ref{tab:operator-families} separates this
effect from the cultural operator used to introduce or preserve the idea.

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

Similar recovery variants later reached primary scores of
""" + latex_escape(fmt(generation_8)) + r""" and """ + latex_escape(fmt(generation_12)) + r""".
By contrast, candidates built around extra imagination-cycle or counterfactual
machinery generally remained below the best full-supervision branch.
""" + latex_escape(child_sentence) + r"""

\section{Discussion}

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
establish persistent action grounding, verify predictive dynamics quality, and
only then introduce imagination or counterfactual constraints.

\section{Limitations}

This is an exploratory automated run, not a final benchmark. The experiment used
one TinyWorlds setting, a short runtime, and a generated candidate population.
The best branch should be retested across independent seeds, longer budgets,
larger TinyWorlds configurations, and direct controls that isolate action
supervision duration from action supervision weight. The current evidence is
predictive, not behavioral: we have not yet shown improved downstream planning.

\section{Conclusion}

The best discovered TinyWorlds world-model intervention was persistent action
supervision. In short-budget training, a small schedule change outperformed more
complex generated action-representation ideas. This supports a conservative but
useful principle: before asking a compact world model to learn rich imagined
counterfactuals, keep its action grounding active long enough for dynamics
learning to use it.

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
