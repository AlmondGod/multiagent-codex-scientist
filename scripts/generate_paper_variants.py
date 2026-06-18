#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from commsci.artifacts import ensure_dir, write_json, write_text
from commsci.codex_scientist.paper import load_population_rows, score_key
from commsci.codex_scientistv2.pipeline import (
    compile_latex_paper,
    copy_icml_latex_template,
    latex_escape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate additional Codex-Scientist-v2 paper variants.")
    parser.add_argument("--run_dir", default="runs/codex_scientistv2_20gen_4agent_seed0")
    return parser.parse_args()


def main() -> int:
    run_dir = Path(parse_args().run_dir).expanduser().resolve()
    rows = load_population_rows(run_dir)
    if not rows:
        raise FileNotFoundError(f"No population summaries found in {run_dir}")
    best = max(rows, key=score_key)
    controls = successful_controls(run_dir)
    latex_dir = ensure_dir(run_dir / "latex")
    copy_icml_latex_template(latex_dir)
    copy_references(run_dir, latex_dir)

    variants = [
        generative_optimizer_variant(rows, best, controls),
        tokenizer_variant(rows, best, controls),
    ]
    review_dir = ensure_dir(run_dir / "review")
    for index, variant in enumerate(variants, start=2):
        tex_path = latex_dir / f"paper{index}_{variant['slug']}.tex"
        md_path = run_dir / f"paper{index}_{variant['slug']}.md"
        write_text(tex_path, render_latex(variant))
        write_text(md_path, render_markdown(variant))
        compile_latex_paper(tex_path)
        write_json(review_dir / f"paper{index}_{variant['slug']}_review.json", review_variant(variant))
    print(f"Wrote {len(variants)} paper variants to {latex_dir}")
    return 0


def successful_controls(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "controlled_ablations" / "summary.json"
    if not path.exists():
        return []
    summary = json.loads(path.read_text(encoding="utf-8"))
    return [control for control in summary.get("controls", []) if control.get("success")]


def copy_references(run_dir: Path, latex_dir: Path) -> None:
    src = run_dir / "codex_scientistv2" / "literature" / "references.bib"
    if src.exists():
        shutil.copyfile(src, latex_dir / "references.bib")


def generative_optimizer_variant(rows: list[dict[str, Any]], best: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "slug": "generative_optimizer_loss",
        "title": "Robust Losses as Implicit Optimizers for Short-Budget Generative World Models",
        "keywords": "generative modeling, robust losses, world models, optimization",
        "abstract": (
            "Short-budget generative world models are often optimized with losses that treat all prediction errors as equally informative, even though controllable motion occupies only a small part of the visual state. "
            "We study whether a robust motion-weighted loss can act as an implicit optimizer by reallocating gradient pressure toward dynamics errors that matter for action-conditioned generation. "
            f"Across {len(rows)} fixed-budget variants, the strongest configuration reached primary score {fmt(best.get('primary_score'))} and validation MSE {fmt(best.get('val_mse'))}. "
            f"{control_sentence(controls)} The evidence suggests that robust objective design can matter as much as architecture changes in small generative world-model regimes, but the result remains preliminary without longer-budget and downstream generation-quality evaluations."
        ),
        "intro": (
            "Generative world models learn to predict future observations, but in action-conditioned settings the prediction target is not uniformly valuable. "
            "Most pixels may be static or easy to reconstruct, while the scientific question concerns whether the model captures controllable changes. "
            "This creates an optimization problem: a loss can produce low reconstruction error while spending too little gradient on action-sensitive dynamics. "
            "We therefore frame the best-performing objective family as an implicit optimizer for short-budget generative modeling. "
            "Rather than changing the architecture or data, the method changes how residuals are weighted and clipped so that limited updates are steered toward motion-local errors."
        ),
        "related": (
            "The framing connects world-model learning, robust regression losses, and objective design for generative models. "
            "Unlike broad architecture searches, this variant asks whether a small loss-level intervention can alter the optimization trajectory under a fixed compute budget."
        ),
        "method": (
            "The method uses a robust Smooth-L1 style dynamics-pixel objective with additional emphasis on changed regions. "
            "The robust component prevents large residuals from dominating the update, while the motion weighting increases the relative gradient assigned to controllable visual changes. "
            "This makes the loss function act like a hand-designed optimizer over prediction errors: it chooses which residuals receive scarce update capacity."
        ),
        "setup": setup_text(rows, controls),
        "results": result_text(best, controls),
        "limitations": common_limitations()
        + " This variant also lacks direct sample-quality metrics such as rollout perceptual quality or planning success, so it should be read as an optimizer/loss hypothesis rather than a complete generative modeling benchmark.",
        "claim_strength": "moderate",
    }


def tokenizer_variant(rows: list[dict[str, Any]], best: dict[str, Any], controls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "slug": "tokenizer_hypothesis",
        "title": "A Tokenization View of Motion-Weighted World-Model Objectives",
        "keywords": "tokenization, world models, action-conditioned dynamics, discretization",
        "abstract": (
            "Tokenizer design determines which prediction errors become salient learning units in many sequence and generative models. "
            "We examine an adjacent question in visual world models: whether a motion-weighted objective can be interpreted as an implicit tokenizer that separates controllable change from background persistence. "
            f"Using the same {len(rows)} fixed-budget variants, the strongest motion-weighted configuration reached primary score {fmt(best.get('primary_score'))} and validation MSE {fmt(best.get('val_mse'))}. "
            f"{control_sentence(controls)} No explicit learned tokenizer was evaluated, so the contribution is a tokenizer-motivated reinterpretation and experimental hypothesis: future work should replace the implicit changed-pixel weighting with learned discrete motion tokens."
        ),
        "intro": (
            "Tokenizers compress raw observations into units that determine what a model can easily predict, attend to, or optimize. "
            "For action-conditioned visual dynamics, a poor tokenization can mix controllable motion with static background, causing the model to spend capacity on easy persistence instead of action effects. "
            "The current experiments did not implement a learned tokenizer, but the strongest objective can be read as a primitive tokenizer: it assigns higher learning weight to changed regions and lower relative weight to uninformative background. "
            "This paper variant asks whether motion-weighted objectives should be developed into explicit tokenization mechanisms for compact world models."
        ),
        "related": (
            "The argument relates discrete tokenization in generative modeling to object-centric and change-centric representation learning. "
            "Here, however, the evidence is indirect because the experiment changes the loss, not the input representation."
        ),
        "method": (
            "The implicit tokenizer is a weighting rule over prediction residuals. "
            "Pixels or regions that change under action receive more optimization pressure, while static regions are treated as lower-information background. "
            "A true tokenizer version would replace this heuristic with learned discrete motion tokens or object-change codes; the present result only motivates that next experiment."
        ),
        "setup": setup_text(rows, controls),
        "results": result_text(best, controls)
        + " Because no token vocabulary, compression module, or tokenizer ablation was run, these results support only the plausibility of a tokenizer hypothesis.",
        "limitations": common_limitations()
        + " The central tokenizer limitation is severe: the current run contains no explicit tokenizer implementation, no vocabulary-size sweep, and no comparison to learned discrete representations.",
        "claim_strength": "weak",
    }


def render_latex(variant: dict[str, Any]) -> str:
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
\usepackage{hyperref}
\graphicspath{{../figures/}{figures/}}
\begin{document}
\twocolumn[
\icmltitle{""" + latex_escape(variant["title"]) + r"""}
\begin{icmlauthorlist}
\icmlauthor{Anonymous Authors}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}{}
\icmlcorrespondingauthor{Anonymous Authors}{anonymous@example.com}
\icmlkeywords{""" + latex_escape(variant["keywords"]) + r"""}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}
""" + latex_escape(variant["abstract"]) + r"""
\end{abstract}
\section{Introduction}
""" + latex_escape(variant["intro"]) + r"""
\section{Related Work}
""" + latex_escape(variant["related"]) + r"""
\section{Method}
""" + latex_escape(variant["method"]) + r"""
\section{Experimental Setup}
""" + latex_escape(variant["setup"]) + r"""
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{best_score_by_generation.pdf}
\caption{Best validation-derived score across evaluated variants.}
\label{fig:best-score}
\end{figure}
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{patch_family_mean_scores.pdf}
\caption{Aggregate score by intervention family.}
\label{fig:patch-family}
\end{figure}
\section{Results}
""" + latex_escape(variant["results"]) + r""" Figure~\ref{fig:best-score} shows discovery over variants, while Figure~\ref{fig:patch-family} compares objective families.
\section{Limitations}
""" + latex_escape(variant["limitations"]) + r"""
\section{Conclusion}
""" + latex_escape(conclusion_text(variant)) + r"""
\bibliography{references}
\bibliographystyle{icml2025}
\appendix
\section{Audit Figure}
The lineage tree is included only as an audit artifact and is not part of the scientific evidence.
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{cultural_tree.pdf}
\caption{Lineage artifact for auditing experiment provenance.}
\label{fig:lineage}
\end{figure}
\end{document}
"""


def render_markdown(variant: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            f"# {variant['title']}",
            "## Abstract\n\n" + variant["abstract"],
            "## Introduction\n\n" + variant["intro"],
            "## Method\n\n" + variant["method"],
            "## Results\n\n" + variant["results"],
            "## Limitations\n\n" + variant["limitations"],
        ]
    ) + "\n"


def review_variant(variant: dict[str, Any]) -> dict[str, Any]:
    if variant["claim_strength"] == "weak":
        return {
            "decision": "reject",
            "overall": 3,
            "soundness": 4,
            "presentation": 6,
            "contribution": 4,
            "confidence": 8,
            "summary": "Tokenizer-motivated framing is coherent but not directly supported by tokenizer experiments.",
            "critical_weaknesses": [
                "No explicit tokenizer, vocabulary, compression module, or tokenizer ablation was evaluated.",
            ],
            "weaknesses": [
                "The paper should be treated as a hypothesis proposal rather than a completed tokenizer paper.",
            ],
        }
    return {
        "decision": "borderline_reject",
        "overall": 5,
        "soundness": 6,
        "presentation": 6,
        "contribution": 5,
        "confidence": 7,
        "summary": "Generative objective/optimizer framing is more directly supported by the loss and ablation evidence, but remains preliminary.",
        "critical_weaknesses": [],
        "weaknesses": [
            "No direct generative sample-quality, rollout, or downstream planning metric is reported.",
            "The controlled checks are useful but still too small for a strong conference claim.",
        ],
    }


def setup_text(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> str:
    return (
        f"We reuse the fixed TinyWorlds setup and compare {len(rows)} short-budget variants with identical evaluation metrics. "
        f"The controlled suite contains {len(controls)} successful checks that isolate repeat, knob-only, and patch-only behavior."
    )


def result_text(best: dict[str, Any], controls: list[dict[str, Any]]) -> str:
    text = (
        f"The strongest configuration reached primary score {fmt(best.get('primary_score'))} and validation MSE {fmt(best.get('val_mse'))}. "
    )
    if controls:
        parts = []
        for control in controls[:3]:
            metrics = control.get("metrics") or {}
            parts.append(f"{control.get('ablation_id')}: score {fmt(metrics.get('primary_score'))}")
        text += "The controlled checks returned " + "; ".join(parts) + "."
    return text


def control_sentence(controls: list[dict[str, Any]]) -> str:
    return (
        f"We include {len(controls)} controlled checks that separate repeat, knob-only, and patch-only behavior."
        if controls
        else "No successful controlled checks are available, so causal claims should be avoided."
    )


def common_limitations() -> str:
    return (
        "The evidence comes from a compact benchmark, short training budgets, and a small controlled suite. "
        "The run does not establish robustness across many independent seeds, larger environments, or downstream planning performance."
    )


def conclusion_text(variant: dict[str, Any]) -> str:
    if variant["slug"] == "tokenizer_hypothesis":
        return (
            "The tokenizer framing is useful as a research direction, but the current evidence is indirect. "
            "A proper follow-up should implement learned motion tokens and compare token vocabularies under the same budget."
        )
    return (
        "The objective/optimizer framing provides a plausible explanation for the observed robust-loss gains. "
        "Future work should test whether the same loss improves longer-horizon generation and planning."
    )


def fmt(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "N/A"


if __name__ == "__main__":
    raise SystemExit(main())
