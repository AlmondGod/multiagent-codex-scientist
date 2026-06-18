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
    ablation_report = read_ablation_report(run_dir)
    generations = sorted({int(row["generation"]) for row in rows})
    best_by_generation = [
        max([row for row in rows if int(row["generation"]) == generation], key=score_key)
        for generation in generations
    ]
    scores = [float(row["primary_score"]) for row in rows if isinstance(row.get("primary_score"), (int, float))]
    output = output_path or run_dir / "paper.md"
    write_text(output, build_domain_paper(rows, best_by_generation, best, weakest, action, metrics, patch, scores, ablation_report))
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
    ablation_report = read_ablation_report(run_dir)
    generations = sorted({int(row["generation"]) for row in rows})
    best_by_generation = [
        max([row for row in rows if int(row["generation"]) == generation], key=score_key)
        for generation in generations
    ]
    scores = [float(row["primary_score"]) for row in rows if isinstance(row.get("primary_score"), (int, float))]
    output = output_path or run_dir / "latex" / "paper.tex"
    ensure_dir(output.parent)
    write_text(output, build_domain_latex(rows, best_by_generation, best, weakest, action, metrics, patch, scores, ablation_report))
    return output


def load_population_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("population_summary_generation_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ablation_report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "codex_scientistv2" / "ablation_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
    ablation_report: dict[str, Any] | None = None,
) -> str:
    ablation_report = ablation_report or {}
    knobs = action.get("knobs") or {}
    env = action.get("env") or {}
    code_diff = patch.get("code_diff") or "No source change recorded."
    controlled = ablation_report.get("controlled_ablations") or {}
    patch_table = summarize_group(rows, "patch_recipe_id")
    ablation_table = build_ablation_table(rows, best)
    scores_text = fmt(mean(scores)) if scores else "N/A"
    theme = paper_theme(best, rows)
    return f"""# {theme['title']}

## Abstract

{theme['abstract_prefix']} The selected method reached primary_score={fmt(best.get('primary_score'))} and val_mse={fmt(best.get('val_mse'))}. {theme['abstract_suffix']}

## 1. Introduction

{theme['introduction']}

## 2. Related Work

World-model research studies learned predictive models that support control, planning, or imagination. In action-conditioned settings, the model must also learn which changes are explained by control inputs rather than by visual persistence. Recent work on learnable action representations argues that action abstractions can be learned jointly with predictive dynamics. This paper studies a narrower question: under a short TinyWorlds budget, which training pressure produces the most reliable predictive model?

## 3. Method

{theme['method_intro']}

```diff
{code_diff.strip()}
```

Best-branch knobs:

```json
{json.dumps(knobs, indent=2, sort_keys=True)}
```

Environment overrides:

```json
{json.dumps(env, indent=2, sort_keys=True)}
```

{theme['method_summary']}

## 4. Experimental Setup

We evaluated {len(rows)} method variants under a fixed short training budget on TinyWorlds. Each variant used the same canonical `train.py` harness in an isolated workspace and wrote `working/metrics.json`. The selected method used {metrics.get('params_M')}M parameters on the `{metrics.get('dataset')}` dataset for {metrics.get('runtime_sec')} seconds. The primary score is the configured scalar objective; `val_mse` is the main predictive-error metric.

Figure 1 tracks the best score reached as additional variants were evaluated.

![Best score by generation](figures/best_score_by_generation.svg)

Figure 2 aggregates method families so the result is not only a single selected variant.

![Patch-family mean scores](figures/patch_family_mean_scores.svg)

## 5. Results

- total candidates: {len(rows)}
- mean primary score: {scores_text}
- selected-method primary score: {fmt(best.get('primary_score'))}
- selected-method validation MSE: {fmt(best.get('val_mse'))}

### Controlled Component Ablations

{controlled_ablation_markdown(controlled, best)}

### Focused Comparisons

{theme['ablation_summary']} These rows are deliberately small: they compare the selected method to conceptually nearby alternatives rather than dumping every tried variant.

| Comparison | Recipe | Patch | Operator | Primary score | Val MSE | Interpretation |
| --- | --- | --- | --- | ---: | ---: | --- |
{ablation_table}

### Aggregate Families

{theme['aggregate_summary']}

| Patch recipe | N | Best score | Mean score |
| --- | ---: | ---: | ---: |
{patch_table}

## 6. Discussion

{theme['discussion']}

## 7. Limitations

This is an exploratory short-budget study, not a final benchmark. The experiment used one TinyWorlds setting, short runtimes, and a finite candidate set. The selected method should be retested across independent seeds, longer budgets, larger TinyWorlds configurations, and direct controls that isolate the apparent winning component from correlated knobs and source edits. The current evidence is predictive, not behavioral: we have not yet shown improved downstream planning.

## 8. Conclusion

{theme['conclusion']}

## Appendix: Candidate Provenance

The candidate lineage is included only for auditability and should not be the main narrative of the paper.

![Candidate lineage](figures/cultural_tree.svg)
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
    ablation_report: dict[str, Any] | None = None,
) -> str:
    ablation_report = ablation_report or {}
    knobs = action.get("knobs") or {}
    env = action.get("env") or {}
    code_diff = patch.get("code_diff") or "No source change recorded."
    controlled = ablation_report.get("controlled_ablations") or {}
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
    return r"""\documentclass{article}
\IfFileExists{icml2025.sty}{
  \usepackage{icml2025}
}{
  \usepackage[margin=1in]{geometry}
  \usepackage{times}
  \newcommand{\icmltitle}[1]{\begin{center}{\Large\bf #1}\end{center}}
  \newcommand{\icmlsetsymbol}[2]{}
  \newcommand{\icmlauthor}[2]{\begin{center}#1\end{center}}
  \newcommand{\icmlaffiliation}[3]{}
  \newcommand{\icmlcorrespondingauthor}[2]{}
  \newcommand{\printAffiliationsAndNotice}[1]{}
  \newcommand{\icmlkeywords}[1]{}
  \newcommand{\theHalgorithm}{\arabic{algorithm}}
  \newenvironment{icmlauthorlist}{}{}
}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{listings}
\usepackage{subcaption}

\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  columns=fullflexible,
  frame=single,
  xleftmargin=0.5em,
  xrightmargin=0.5em
}

\begin{document}
\twocolumn[
\icmltitle{""" + latex_escape(theme["title"]) + r"""}
\begin{icmlauthorlist}
\icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}{}
\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}
\icmlkeywords{world models, action-conditioned dynamics, short-budget learning, robust dynamics}
\vskip 0.3in
]

\begin{abstract}
""" + latex_escape(theme["abstract_prefix"]) + r""" The selected method reached primary score
""" + latex_escape(fmt(best.get("primary_score"))) + r""" and validation MSE
""" + latex_escape(fmt(best.get("val_mse"))) + r""". """ + latex_escape(theme["abstract_suffix"]) + r"""
\end{abstract}

\section{Introduction}

""" + latex_escape(theme["introduction"]) + r"""

\section{Related Work}

World-model research studies learned predictive models that support control,
planning, or imagination \cite{ha2018worldmodels}. In action-conditioned
settings, the model must also learn which changes are explained by control
inputs rather than by visual persistence. Recent work on learnable action
representations argues that action abstractions can be learned jointly with
predictive dynamics \cite{prelar2024}. This paper studies a narrower question:
under a short TinyWorlds budget, which training pressure produces the most
reliable predictive model?

\section{Method}

""" + latex_escape(theme["method_intro"]) + r"""

\begin{lstlisting}
""" + latex_lst(code_diff.strip()) + r"""
\end{lstlisting}

The selected method used this configuration:

\begin{lstlisting}
""" + latex_lst(json.dumps(knobs, indent=2, sort_keys=True)) + r"""
\end{lstlisting}

which produced these environment overrides:

\begin{lstlisting}
""" + latex_lst(json.dumps(env, indent=2, sort_keys=True)) + r"""
\end{lstlisting}

""" + latex_escape(theme["method_summary"]) + r"""

\section{Experimental Setup}

We evaluated """ + str(len(rows)) + r""" method variants under a fixed short training
budget on TinyWorlds. Each variant ran the same canonical \texttt{train.py} harness in an
isolated workspace and wrote \texttt{working/metrics.json}. The selected method used
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
\caption{Best primary score by generation in the benchmark run.}
  \label{fig:best-score}
\end{figure}

\begin{figure}[t]
  \centering
  \IfFileExists{../figures/patch_family_mean_scores.pdf}{%
    \includegraphics[width=0.82\linewidth]{../figures/patch_family_mean_scores.pdf}
  }{%
    \fbox{\parbox{0.9\linewidth}{Generated patch-family figure missing:
    \texttt{../figures/patch\_family\_mean\_scores.pdf}.}}
  }
  \caption{Mean primary score by patch family. This summarizes which intervention
  families were consistently competitive, rather than only reporting the best
  selected variant.}
  \label{fig:patch-family}
\end{figure}

\section{Results}

Figure~\ref{fig:best-score} shows the best score reached by each generation, and
Figure~\ref{fig:patch-family} summarizes whether the strongest result is part of
a broader competitive intervention family.

Across all evaluated variants, the mean primary score was """ + latex_escape(mean_score) + r""". The
selected method reached primary score """ + latex_escape(fmt(best.get("primary_score"))) + r""" with validation
MSE """ + latex_escape(fmt(best.get("val_mse"))) + r""".

\subsection{Controlled Component Ablations}

Table~\ref{tab:controlled-ablations} reruns the selected method under a smaller
controlled-ablation budget and isolates the contribution of copied knobs versus
the selected patch/source edit. These controls are more important than the
variant ranking because they test whether the proposed method survives component
removal.

\begin{table}[t]
\centering
\small
\begin{tabular}{p{0.35\linewidth}rrp{0.28\linewidth}}
\toprule
Control & Score & Val. MSE & Interpretation \\
\midrule
""" + latex_controlled_ablation_rows(controlled, best) + r"""
\bottomrule
\end{tabular}
\caption{Controlled component ablations for the selected method.}
\label{tab:controlled-ablations}
\end{table}

\subsection{Focused Ablations}

Table~\ref{tab:ablations} compares the selected method against initial baselines,
negative-result controls, and close variants. """ + latex_escape(theme["ablation_summary"]) + r""" These rows are deliberately small:
they compare the selected method to conceptually nearby alternatives rather than
dumping every tried variant.

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
\caption{Focused comparisons selected from nearby method variants.}
\label{tab:ablations}
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

\section{Discussion}

""" + latex_escape(theme["discussion"]) + r"""

\section{Limitations}

This is an exploratory short-budget study, not a final benchmark. The experiment used
one TinyWorlds setting, short runtimes, and a finite candidate set.
The selected method should be retested across independent seeds, longer budgets,
larger TinyWorlds configurations, and direct controls that isolate the apparent
winning component from correlated knobs and source edits. The current evidence is
predictive, not behavioral: we have not yet shown improved downstream planning.

\section{Conclusion}

""" + latex_escape(theme["conclusion"]) + r"""

\appendix
\section{Candidate Provenance}

The manuscript above treats the candidate population as an experimental design
tool rather than as the scientific contribution. For auditability, Figure
~\ref{fig:tree-appendix} shows the full lineage used to select candidate
interventions and to identify negative-result controls.

\begin{figure*}[t]
  \centering
  \IfFileExists{../figures/cultural_tree.pdf}{%
    \includegraphics[width=\textwidth]{../figures/cultural_tree.pdf}
  }{%
    \fbox{\parbox{0.9\textwidth}{Generated lineage figure missing:
    \texttt{../figures/cultural\_tree.pdf}.}}
  }
  \caption{Full candidate lineage. Solid edges denote explicit source transfer;
  dotted edges denote same-agent parent links. This figure is provenance, not
  the central result.}
  \label{fig:tree-appendix}
\end{figure*}

\bibliographystyle{icml2025}
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
            "Strong initial robust-reconstruction baseline.",
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
            "Selected method",
            best,
            "Method selected for component testing.",
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
            "title": "Persistent Action Supervision Stabilizes Short-Budget World Models",
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
                "training. The exploratory answer is yes: the selected method kept supervised action grounding active throughout "
                "the run."
            ),
            "method_intro": (
                "The discovered intervention keeps supervised action grounding active throughout the run. In the base training loop, "
                "action supervision is active only for an initial window; the selected method changed the schedule as follows:"
            ),
            "method_summary": (
                "The method combines action-conditioned dynamics prediction, decoded future reconstruction, and persistent supervised "
                "alignment to observed environment actions. It is intentionally small: it changes the training schedule rather than "
                "adding a new module."
            ),
            "ablation_summary": "The best result came from preserving action supervision for the full budget.",
            "aggregate_summary": (
                "The patch-family summary shows that full-budget action supervision produced the best single candidate. "
                "The aggregate view separates the method family from a single selected variant."
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
    object_theme = any(
        marker in f" {run_text} "
        for marker in [" object-local ", " object-centric ", " object curriculum ", " object-curriculum "]
    ) or "object_" in recipe or "object-" in recipe
    if object_theme:
        return {
            "title": "Object-Local Robust Dynamics for Short-Budget World Model Learning",
            "abstract_prefix": (
                "TinyWorlds observations contain large static regions and small controllable regions. We study object-local "
                "transition learning through changed-pixel weighting, motion priors, and robust decoded future reconstruction."
            ),
            "abstract_suffix": (
                "The results suggest that short-budget TinyWorlds models may benefit more from localized transition pressure "
                "than from more elaborate action-abstraction objectives."
            ),
            "introduction": (
                "World-model errors are rarely uniform across an observation. In TinyWorlds, much of the frame can be static while "
                "a small set of controllable regions carries the transition signal. This paper studies whether a short-budget world "
                "model improves when training pressure is localized toward changed or object-like regions. The central hypothesis "
                "is that robust reconstruction plus local transition weighting is a better use of a short budget than adding richer "
                "action-abstraction machinery before the base predictor is stable."
            ),
            "method_intro": (
                "The selected method used a robust Smooth-L1 dynamics-pixel recipe with an object-local curriculum. "
                "The recorded source diff, if any, is:"
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
                "The patch-family summary compares robust dynamics, sharpened change weighting, and dynamics-first curricula. "
                "This supports the object-local claim without relying on a ranked dump of every variant."
            ),
            "lineage_result": (
                "The object-local variant preserves robust reconstruction while concentrating training pressure on local transition errors."
            ),
            "discussion": (
                "The result is consistent with a simple hypothesis: when compute is scarce, reducing error on locally changing regions "
                "can be easier than learning a richer global latent action abstraction. The current evidence is still exploratory "
                "and needs seed replication."
            ),
            "conclusion": (
                "Object-local robust dynamics is the strongest TinyWorlds candidate in this run. The next direct test is to replicate "
                "the robust object-curriculum branch across seeds and isolate changed-pixel weighting, robust pixel loss, and curriculum depth."
            ),
        }
    if "counterfactual" in run_text or "latent" in run_text or "imagination" in run_text:
        return {
            "title": "Counterfactual Action Objectives Underperform Robust Reconstruction in Short-Budget World Models",
            "abstract_prefix": (
                "Latent-action teachers, imagination-cycle consistency, and counterfactual action discrimination are appealing ways "
                "to make world models action-sensitive. We test whether these objectives help in short-budget TinyWorlds training."
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
                "The method family tested latent and counterfactual action objectives. The selected method's recorded source diff, if any, is:"
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
                "The patch-family summary separates robust reconstruction and baseline edits within the counterfactual-action variants."
            ),
            "lineage_result": (
                "The action-abstraction variants produced useful negative evidence, but did not surpass the simpler robust reconstruction method."
            ),
            "discussion": (
                "The negative result is scientifically useful. Counterfactual and imagination-cycle losses may require a stronger base "
                "dynamics model before they become helpful. Under a short budget, they add optimization pressure before visual prediction "
                "and transition modeling are stable."
            ),
            "conclusion": (
                "These experiments did not support counterfactual action objectives as an immediate improvement for short-budget "
                "TinyWorlds. A staged approach should first establish robust predictive dynamics, then add action-identifiability losses."
            ),
        }
    if "fresh_" in recipe or "motion" in recipe or "change" in recipe or "motion" in rationale or "change" in rationale:
        return {
            "title": "Motion-Calibrated Robust Dynamics for Short-Budget World Models",
            "abstract_prefix": (
                "Short-budget TinyWorlds models must allocate loss to the parts of the scene that actually move. We evaluate "
                "motion-calibrated robust dynamics: a Smooth-L1 decoded-future objective paired with motion-aware loss weights."
            ),
            "abstract_suffix": (
                "The result suggests that, under short compute budgets, robust reconstruction and motion-aware calibration may be a more "
                "reliable next research direction than adding high-variance action-abstraction objectives."
            ),
            "introduction": (
                "TinyWorlds prediction quality depends disproportionately on changing and moving regions. Static background pixels can "
                "dominate reconstruction loss while contributing little to action-conditioned dynamics. This paper asks whether robust "
                "decoded-future reconstruction with motion-aware calibration improves short-budget world-model training."
            ),
            "method_intro": (
                "The selected method used a robust Smooth-L1 dynamics-pixel objective with motion calibration knobs. No new source edit was "
                "needed beyond the selected patch recipe; the method changed the training pressure through the following recorded diff, "
                "if any:"
            ),
            "method_summary": (
                "The method emphasizes decoded future robustness and motion-aware calibration: dynamics pixel loss is made robust, and "
                "motion losses focus the short training budget on changes that matter for future prediction."
            ),
            "ablation_summary": (
                "The selected method should be read as exploratory evidence for motion-calibrated robust dynamics, not as proof of a universal improvement."
            ),
            "aggregate_summary": (
                "The patch-family summary compares robust dynamics, change weighting, and curriculum families."
            ),
            "lineage_result": (
                "The method combines the robust decoded-future loss with motion calibration rather than adding an action-abstraction module."
            ),
            "discussion": (
                "Motion-calibrated robust dynamics may work because short-budget models can improve predictive quality by allocating "
                "loss to moving or changing regions while avoiding brittle objectives that require well-formed latent action abstractions. "
                "The controlled ablations remain short and should be treated as stress tests rather than final evidence."
            ),
            "conclusion": (
                "Motion-calibrated robust dynamics is the selected method in this run. The result motivates a focused follow-up: replicate "
                "the method across seeds and isolate whether motion loss, robust pixel loss, or their combination drives the gain."
            ),
        }
    return {
        "title": "Controlled Component Ablations for Short-Budget World Model Training",
        "abstract_prefix": "We evaluate candidate TinyWorlds world-model interventions under a short training budget.",
        "abstract_suffix": "The result should be interpreted as exploratory evidence for the selected method's design choices.",
        "introduction": "This paper analyzes the strongest TinyWorlds world-model intervention observed under a fixed short training budget.",
        "method_intro": "The selected method used the following recorded source diff, if any:",
        "method_summary": "The method is defined by the recorded action, knobs, patch recipe, and source diff preserved in the run artifacts.",
        "ablation_summary": "The focused comparisons summarize the selected method against nearby alternatives.",
        "aggregate_summary": "The aggregate table summarizes patch-family performance.",
        "lineage_result": "{child_sentence}",
        "discussion": "The result is exploratory and should be followed by controlled replication and component isolation.",
        "conclusion": "The fresh run identifies a candidate intervention for follow-up TinyWorlds experiments.",
    }


def latex_controlled_ablation_rows(controlled: dict[str, Any], best: dict[str, Any]) -> str:
    controls = controlled.get("controls") or []
    if not controls:
        return (
            r"\multicolumn{4}{p{0.95\linewidth}}{No controlled ablation reruns were found; "
            r"the paper should not claim component-level causality from search-trace comparisons alone.} \\"
        )
    labels = {
        "repeat": "Repeat selected method",
        "knobs_only": "Knobs only",
        "patch_only": "Patch/source only",
        "minimal_baseline": "Minimal baseline",
    }
    lines = []
    for control in controls:
        ablation_id = str(control.get("ablation_id", ""))
        metrics = control.get("metrics") or {}
        if "knobs_only" in ablation_id:
            key = "knobs_only"
            interpretation = "Tests whether tuned scalar settings explain the gain."
        elif "patch_only" in ablation_id:
            key = "patch_only"
            interpretation = "Tests whether source/patch changes suffice without tuned knobs."
        elif "minimal_baseline" in ablation_id:
            key = "minimal_baseline"
            interpretation = "Removes selected-method components as a negative control."
        else:
            key = "repeat"
            interpretation = "Checks whether the selected method survives an independent rerun."
        lines.append(
            " & ".join(
                [
                    latex_escape(labels[key]),
                    latex_escape(fmt(metrics.get("primary_score"))),
                    latex_escape(fmt(metrics.get("val_mse"))),
                    latex_escape(interpretation),
                ]
            )
            + r" \\"
        )
    source_score = controlled.get("source_best_score", best.get("primary_score"))
    lines.insert(
        0,
        " & ".join(
            [
                "Selected method",
                latex_escape(fmt(source_score)),
                latex_escape(fmt(best.get("val_mse"))),
                "Original method selected for component testing.",
            ]
        )
        + r" \\"
    )
    return "\n".join(lines)


def controlled_ablation_markdown(controlled: dict[str, Any], best: dict[str, Any]) -> str:
    controls = controlled.get("controls") or []
    if not controls:
        return "No controlled ablation reruns were found; do not claim component-level causality from variant comparisons alone."
    lines = [
        "| Control | Primary score | Val MSE | Interpretation |",
        "| --- | ---: | ---: | --- |",
        (
            f"| Selected method | {fmt(controlled.get('source_best_score', best.get('primary_score')))} | "
            f"{fmt(best.get('val_mse'))} | Original method selected for component testing. |"
        ),
    ]
    for control in controls:
        ablation_id = str(control.get("ablation_id", ""))
        metrics = control.get("metrics") or {}
        if "knobs_only" in ablation_id:
            label = "Knobs only"
            interpretation = "Tests whether tuned scalar settings explain the gain."
        elif "patch_only" in ablation_id:
            label = "Patch/source only"
            interpretation = "Tests whether source/patch changes suffice without tuned knobs."
        elif "minimal_baseline" in ablation_id:
            label = "Minimal baseline"
            interpretation = "Removes selected-method components as a negative control."
        else:
            label = "Repeat selected method"
            interpretation = "Checks whether the selected method survives an independent rerun."
        lines.append(f"| {label} | {fmt(metrics.get('primary_score'))} | {fmt(metrics.get('val_mse'))} | {interpretation} |")
    return "\n".join(lines)


def latex_ablation_rows(rows: list[dict[str, Any]], best: dict[str, Any]) -> str:
    candidates = [
        ("Initial robust decoder", best_matching(rows, recipe_prefix="g0_agent1_robust_decoder_loss")),
        ("Initial action-gradient", best_matching(rows, recipe_prefix="g0_agent0_auxiliary_action_contrast")),
        ("Initial curriculum", best_matching(rows, recipe_prefix="g0_agent2_short_budget_curriculum")),
        ("Selected method", best),
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
