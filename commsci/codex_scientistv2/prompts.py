from __future__ import annotations

import json
from typing import Any


MAX_FIGURES = 12


REVIEWER_SYSTEM_PROMPT_NEG = (
    "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue. "
    "Be critical and cautious in your decision. If a paper is bad or you are unsure, give it bad scores and reject it."
)


REVIEW_TEMPLATE_INSTRUCTIONS = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Be specific to the current paper. Do not make generic comments.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, do not use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Use only Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""


NEURIPS_REVIEW_FORM = (
    """
## Review Form
Review this as a real ML conference submission, not as an internal project report.

1. Summary: Briefly summarize the paper and its contributions.
2. Strengths and Weaknesses: Assess originality, technical quality, clarity, and significance.
3. Questions: List questions whose answers could change your opinion.
4. Limitations: Check whether limitations and potential negative impact are adequately addressed.
5. Soundness: Rate whether the technical claims, methodology, and evidence support the conclusions.
6. Presentation: Rate writing, organization, mathematical/method detail, and contextualization.
7. Contribution: Rate whether the paper asks an important question and contributes a significant idea or result.
8. Overall: Use 1-10 where 6+ should be reserved for papers that could plausibly be accepted at a strong ML venue.
9. Confidence: State your confidence.

Calibration:
- A real conference paper should have a crisp research problem, a motivated method, enough detail to reproduce the method, serious baselines, controlled experiments, and a narrative that reads like a scientific contribution.
- Papers such as "Attention Is All You Need" and strong meta-learning papers define a clear problem, motivate a method from prior limitations, explain the model, and then present experiments that directly support the claim. Use that standard.
- A paper that mainly reports an automated search timeline, selected candidate, best branch, source diff, run artifacts, generated figures, or provenance is not a conference paper and should usually be rejected.
- A paper that says the result is only a candidate for follow-up, lacks independent seeds, lacks strong baselines, or cannot show downstream significance should receive low significance and usually an Overall <= 4.
- Do not reward a paper merely for having an ICML template, tables, figures, or ablations. Judge whether they support a real contribution.
- If you are unsure whether the paper is acceptable, reject it.
"""
    + REVIEW_TEMPLATE_INSTRUCTIONS
)


IDEATION_SYSTEM_PROMPT = """You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

In this Codex-Scientist-v2 run, Codex is the model executing the research loop. You do not call an external Claude API. You must produce concrete, auditable action JSON that the local runner can execute.

You may use the following actions:

- **SearchSemanticScholar**: Propose a literature-search query that would inform the idea.
- **FinalizeIdea**: Finalize your idea by providing the executable idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format, approximately 250 words.
- "Experiments": A list of simple feasible experiments. Be specific in exactly how you would test the hypothesis, detail precise algorithmic changes, and include evaluation metrics.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.

Respond in the following format:

ACTION:
<The action to take, exactly one of "SearchSemanticScholar" or "FinalizeIdea">

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {"query": "your search query"}. If ACTION is "FinalizeIdea", provide the idea details as {"idea": { ... }} with the IDEA JSON specified above.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{
  "idea": {
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
  }
}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should use the provided literature context as the equivalent of the literature-search step when live tool use is unavailable. If the literature context is empty, explicitly state what search query should be run before finalizing."""


IDEA_GENERATION_PROMPT = """{workshop_description}

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""


IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action, if any:

{last_tool_results}
"""


CODEX_ACTION_FINALIZATION_PROMPT = """For this repository, finalize the idea as executable Codex-Scientist action JSON.

Write only valid JSON to the target action file. The action JSON may include:

```json
{
  "recipe_id": "short_unique_name",
  "inheritance_mode": "invent | copy | mutate | recombine | reject",
  "source_agent_ids": ["agent_1"],
  "source_node_ids": ["cultural_evolution_agent_1_node_2"],
  "patch_recipe_id": "baseline_no_patch | dynamics_first_schedule | action_grad_dynamics | smooth_l1_dynamics_pixel | sharpen_change_weights | full_budget_action_supervision",
  "knobs": {},
  "file_edits": [
    {
      "path": "models.py",
      "description": "what this edit does",
      "find": "exact existing text",
      "replace": "replacement text"
    }
  ],
  "rationale": "why this is a high-variance, literature-informed idea worth trying"
}
```

The JSON is not a prose proposal. It is the executable research action for the next Codex node. Keep the intervention simple, feasible, and falsifiable under the fixed TinyWorlds budget."""


WRITEUP_SYSTEM_MESSAGE_TEMPLATE = """You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field.
Ensure that the paper is scientifically accurate, objective, and truthful. Accurately report the experimental results, even if they are negative or inconclusive.
You are planning to submit to a top-tier ML conference, which has guidelines:
- The main paper is limited to {page_limit} pages, including all figures and tables, but excluding references, the impact statement, and optional appendices. In general, try to use the available space and include all relevant information.
- The main paper should be double-column format, while the appendices can be in single-column format. When in double column format, make sure that tables and figures are correctly placed.
- Do not change the overall style which is mandated by the conference. Keep to the current method of including the references.bib file.
- Do not remove the \\graphicspath directive or no figures will be found.

Here are some tips for each section of the paper:

- **Title**:
  - Title should be catchy and informative. It should give a good idea of what the paper is about.
  - Try to keep it under 2 lines.
  - In this project, do not title the paper after the benchmark or the autoresearch process; title it after the scientific method, claim, or negative result.

- **Abstract**:
  - TL;DR of the paper.
  - What are we trying to do and why is it relevant?
  - Make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph.

- **Introduction**:
  - Longer version of the Abstract, i.e., an overview of the entire paper.
  - Provide context to the study and explain its relevance.
  - If results are inconclusive or negative, present them frankly; if they are positive, you may highlight how the approach effectively addresses the research question or problem.
  - Summarize your contributions, highlighting pertinent findings, insights, or proposed methods.

- **Related Work**:
  - Academic siblings of our work, i.e., alternative attempts in literature at trying to address the same or similar problems.
  - Compare and contrast their approach with yours, noting key differences or similarities.
  - Ensure proper citations are provided.

- **Background**:
  - Present foundational concepts or prior work needed to understand your method.
  - This should include necessary definitions, the problem setting, or relevant theoretical constructs.

- **Method**:
  - Clearly detail what you propose to do and why. If your study aims to address certain hypotheses, describe them and how your method is constructed to test them.
  - If results are negative or inconclusive, you may suggest improvements or discuss possible causes.

- **Experimental Setup**:
  - Explain how you tested your method or hypothesis.
  - Describe necessary details such as data, environment, and baselines, but omit hardware details unless explicitly mentioned.

- **Experiments**:
  - Present the results truthfully according to the data you have. If outcomes are not as expected, discuss it transparently.
  - Include comparisons to baselines if available, and only include analyses supported by genuine data.
  - Try to include all relevant plots and tables. Consider combining multiple plots into one figure if they are related.

- **Conclusion**:
  - Summarize the entire paper, including key strengths or findings.
  - If results are strong, highlight how they might address the research problem.
  - If results are negative or inconclusive, highlight potential improvements or reasons and propose future directions.

- **Appendix**:
  - Place for supplementary material that did not fit in the main paper.

Ensure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:
- LaTeX syntax errors, unenclosed math, unmatched braces, etc.
- Duplicate figure labels or references.
- Unescaped special characters: & % $ # _ {{ }} ~ ^ \\
- Proper table/figure closure.
- Do not hallucinate new citations or any results not in the logs.

When returning final code, place it in fenced triple backticks with 'latex' syntax highlighting."""


WRITEUP_PROMPT = """Your goal is to write up the following idea:

```markdown
{idea_text}
```

We have the following experiment summaries (JSON):
```json
{summaries}
```

We also have a script used to produce the final plots or a description of how the plots were generated. Use this to see how the plots are generated and what names are used in the legend:
```python
{aggregator_code}
```
Please also consider which plots should naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have figure descriptions or local figure-review notes:
```
{plot_descriptions}
```

Your current progress on the LaTeX write-up is:
```latex
{latex_writeup}
```

Produce the final version of the LaTeX manuscript now, ensuring the paper is coherent, concise, and reports results accurately.
Return the entire file in full, with no unfilled placeholders!
This must be an acceptable complete LaTeX writeup.

Please provide the updated LaTeX code for 'template.tex', wrapped in triple backticks
with "latex" syntax highlighting, like so:

```latex
<UPDATED LATEX CODE>
```"""


WRITEUP_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.

Review the current LaTeX manuscript as a strict ML conference author before resubmitting it.
Fix scientific framing, unsupported claims, missing or weak ablations, figure references, captions, citations, limitations, table placement, and LaTeX errors.
Do not invent results, citations, or figures.
If results are negative or inconclusive, make that the paper's honest narrative.
If the main paper contains process-log language about the autoresearch system, remove it or move provenance to the appendix.

Return the full updated LaTeX code for 'template.tex' wrapped in a fenced ```latex block."""


def plot_aggregation_system_prompt(max_figures: int = MAX_FIGURES) -> str:
    return f"""You are an ambitious AI researcher who is preparing final plots for a scientific paper submission.
You have multiple experiment summaries, each possibly containing references to different plots or numerical insights.
There is also a top-level research idea or selected-method description that outlines the overarching research direction.
Your job is to produce ONE Python script that fully aggregates and visualizes the final results for a comprehensive research paper.

Key points:
1) Combine or replicate relevant existing plotting code, referencing how data was originally generated to ensure correctness.
2) Create a complete set of final scientific plots, stored in 'figures/' only, since only those are used in the final paper.
3) Use existing numerical data for analysis; do NOT hallucinate data. If single numeric results are needed, these may be copied from JSON summaries.
4) Only create plots where the data is best presented as a figure and not as a table. For example, do not use bar plots if the data is hard to visually compare.
5) The final aggregator script must be in triple backticks and stand alone so it can be dropped into a codebase and run.
6) If there are plots based on synthetic data, include them in the appendix.

Implement best practices:
- Do not produce extraneous or irrelevant plots.
- Maintain clarity, minimal but sufficient code.
- Demonstrate thoroughness for a final research paper submission.
- Do NOT reference non-existent files or images.
- Use available data files when present and key numbers from the JSON summaries.
- Demarcate each individual plot, and put them in separate try-catch blocks so that the failure of one plot does not affect the others.
- Make sure to only create plots that are unique and needed for the final paper and appendix. A good number could be around {max_figures} plots in total.
- Aim to aggregate multiple figures into one plot if suitable, i.e. if they are all related to the same topic. You can place up to 3 plots in one row.
- Provide well-labeled plots (axes, legends, titles) that highlight main findings. Use informative names everywhere, including in the legend for referencing them in the final paper. Make sure the legend is always visible.
- Make the plots look professional: no top and right spines where applicable, dpi of 300, adequate y-limits, readable labels, and clear captions.
- Do not use labels with underscores, e.g. "loss_vs_epoch" should be "loss vs epoch".
- For image examples, select a few categories/classes to showcase diversity instead of showing a single category/class. Some can be included in the main paper, while the rest can go in the appendix.

Your output should be the entire Python aggregator script in triple backticks."""


PLOT_AGGREGATION_PROMPT = """We have JSON summaries of scientific experiments: baseline variants, research variants, and ablations.
They may contain lists of figure descriptions, code to generate figures, paths to numerical data, and direct scalar results.
Our goal is to produce final, publishable figures.

--- RESEARCH IDEA ---
```
{idea_text}
```

IMPORTANT:
- If experiment data files are available, the aggregator script must load existing experiment data using full and exact file paths from the summaries.
- If no data files are available, use only key numbers copied from the JSON summaries. Do not hallucinate any extra measurements.
- It should call os.makedirs("figures", exist_ok=True) before saving any plots.
- Aim for a balance of empirical results, ablations, and diverse, informative visuals in 'figures/' that comprehensively showcase the finalized research outcomes.
- If you need paths from the summary, only copy those paths directly rather than inventing new paths.

Your generated Python script must:
1) Load or refer to relevant data from these summaries using exact paths where paths exist.
2) Synthesize or directly create final, scientifically meaningful plots for a final research paper, referencing the original plotting logic if needed to see how the data was generated.
3) Carefully combine or replicate relevant existing plotting code to produce final aggregated plots in 'figures/' only, since only those are used in the final paper.
4) Do not hallucinate data. Data must either be loaded from files or copied from JSON summaries.
5) The aggregator script must be fully self-contained, and place the final plots in 'figures/'.
6) This aggregator script should produce a comprehensive and final set of scientific plots for the final paper, reflecting all major findings from the experiment data.
7) Make sure that every plot is unique and not duplicated from the original plots. Delete or skip duplicate plots if necessary.
8) Each figure can have up to 3 subplots using fig, ax = plt.subplots(1, 3).
9) Use a font size larger than the default for plot labels and titles to ensure they are readable in the final PDF paper.

Available existing figure files:
```
{plot_list}
```

Existing plotting implementation or local plotting description:
```python
{aggregator_code}
```

Below are the summaries in JSON:

{combined_summaries_str}

Respond with a Python script in triple backticks."""


PLOT_REFLECTION_PROMPT_TEMPLATE = """We have run your aggregator script and it produced {figure_count} figure(s). The script's output is:
```
{aggregator_out}
```

Please criticize the current script for any flaws including but not limited to:
- Are these enough plots for a final paper submission? Do not create more than {max_figures} plots.
- Have you made sure to both use key numbers and generate more detailed plots from available data files?
- Does each figure title and legend have informative and descriptive names? These plots are the final versions, so ensure there are no comments or other notes.
- Can you aggregate multiple plots into one figure if suitable?
- Do the labels have underscores? If so, replace them with spaces.
- Is every plot unique and not duplicated from the original plots?
- Are the figures appropriate for the main paper versus appendix?

If you believe you are done, simply say: "I am done". Otherwise, please provide an updated aggregator script in triple backticks."""


def codex_live_ideation_prompt(
    *,
    workshop_description: str,
    previous_ideas: str,
    literature_context: str,
    last_tool_results: str = "No new results.",
) -> str:
    return (
        "## AI-Scientist-v2-Style Ideation Prompt Adapted for Codex\n\n"
        "### System Prompt\n"
        + IDEATION_SYSTEM_PROMPT
        + "\n\n### Initial Idea Generation Prompt\n"
        + IDEA_GENERATION_PROMPT.format(
            workshop_description=workshop_description,
            prev_ideas_string=previous_ideas or "No previous proposals.",
        )
        + "\n\n### Reflection Prompt\n"
        + IDEA_REFLECTION_PROMPT.format(
            current_round=1,
            num_reflections=3,
            last_tool_results=last_tool_results or "No new results.",
        )
        + "\n\n### Literature Context Available To Codex\n```markdown\n"
        + (literature_context or "No literature context provided.")
        + "\n```\n\n"
        + CODEX_ACTION_FINALIZATION_PROMPT
    )


def codex_writeup_prompt_bundle(
    *,
    idea_text: str,
    summaries: dict[str, Any],
    aggregator_code: str,
    plot_list: list[str],
    plot_descriptions: str,
    latex_writeup: str,
    page_limit: int = 8,
) -> str:
    return (
        "## AI-Scientist-v2-Style Writeup Prompt Adapted for Codex\n\n"
        "### System Prompt\n"
        + WRITEUP_SYSTEM_MESSAGE_TEMPLATE.format(page_limit=page_limit)
        + "\n\n### Writeup Prompt\n"
        + WRITEUP_PROMPT.format(
            idea_text=idea_text,
            summaries=json.dumps(summaries, indent=2, sort_keys=True),
            aggregator_code=aggregator_code,
            plot_list="\n".join(plot_list),
            plot_descriptions=plot_descriptions,
            latex_writeup=latex_writeup,
        )
        + "\n\n### Reflection Prompt\n"
        + WRITEUP_REFLECTION_PROMPT.format(current_round=1, num_reflections=3)
    )


def codex_plotting_prompt_bundle(
    *,
    idea_text: str,
    summaries: dict[str, Any],
    aggregator_code: str,
    plot_list: list[str],
    figure_count: int = 0,
    aggregator_out: str = "Aggregator has not been run yet.",
    max_figures: int = MAX_FIGURES,
) -> str:
    return (
        "## AI-Scientist-v2-Style Plot Aggregation Prompt Adapted for Codex\n\n"
        "### System Prompt\n"
        + plot_aggregation_system_prompt(max_figures=max_figures)
        + "\n\n### Aggregator Prompt\n"
        + PLOT_AGGREGATION_PROMPT.format(
            idea_text=idea_text,
            combined_summaries_str=json.dumps(summaries, indent=2, sort_keys=True),
            aggregator_code=aggregator_code,
            plot_list="\n".join(plot_list),
        )
        + "\n\n### Reflection Prompt\n"
        + PLOT_REFLECTION_PROMPT_TEMPLATE.format(
            figure_count=figure_count,
            aggregator_out=aggregator_out,
            max_figures=max_figures,
        )
    )


def codex_strict_review_prompt_bundle(
    *,
    paper_text: str,
    evidence: dict[str, Any] | None = None,
) -> str:
    return (
        "## Strict NeurIPS-Style Automated Review Prompt Adapted for Codex\n\n"
        "### System Prompt\n"
        + REVIEWER_SYSTEM_PROMPT_NEG
        + "\n\n### Review Form\n"
        + NEURIPS_REVIEW_FORM
        + "\n\n### Extra Calibration For Codex-Scientist-v2 Manuscripts\n"
        "- Review the submitted paper as a standalone ML conference paper. The reviewer should not need to know that an automated search system produced it.\n"
        "- Strongly penalize any main-paper language about an autoresearch timeline, candidate search, selected branch, source diff, run artifacts, provenance, or process logs.\n"
        "- A manuscript can mention the benchmark in the experimental setup, but the title, abstract, and introduction should center a method, hypothesis, or clear negative result.\n"
        "- Figures, ICML formatting, and ablations only help if they support a coherent scientific claim. Do not reward artifact completeness by itself.\n"
        "- If the evidence shows only a single run, weak replication, post-hoc population comparisons, or no downstream significance, assign low significance and reject.\n"
        "- If the paper reads like a lab report or run summary rather than a conference paper, the Decision must be Reject.\n\n"
        "### Available Run Evidence\n"
        "Use this evidence only to check truthfulness and unsupported claims. Do not excuse weak writing or process-log framing.\n"
        "```json\n"
        + json.dumps(evidence or {}, indent=2, sort_keys=True)[:30000]
        + "\n```\n\n"
        "### Paper To Review\n"
        "```latex\n"
        + paper_text[:60000]
        + "\n```\n"
    )
