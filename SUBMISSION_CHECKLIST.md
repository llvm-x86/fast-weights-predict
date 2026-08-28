# arXiv submission rubric — compliance checklist

The rubric the moderator is applying is arXiv's official moderation policy, which has two
parts: **content moderation** ([arXiv Content Moderation](https://info.arxiv.org/help/moderation/index.html))
and **format requirements** ([arXiv Format Requirements](https://info.arxiv.org/help/policies/format_requirements.md)).
arXiv moderators decline submissions that fail either. This file maps every item to its status
for this paper.

Legend: ✅ resolved · ⚠️ needs an input from you · ➖ not applicable

## A. Content moderation (scholarly standards & interest)

| Item | Requirement | Status |
|---|---|---|
| A1 | **Title and authorship — no anonymous submissions** | ✅ Byline set to Gabriel Hill (gabriel@familyfungroup.com) in `paper.tex` and `paper.md`. |
| A2 | Carefully prepared sections, figures, tables, references | ✅ Title, Abstract, §1–§7, Tables 1–3, 12 references, and a disclosures section are present. |
| A3 | Professional, sufficiently neutral tone | ✅ No promotional or extraneous content. |
| A4 | Originality, novelty, significance; no misrepresentation | ✅ Honest scope (toy domain stated explicitly). All numbers reproduce from `bench.py` (10 seeds); abstract and body percentages trace to Table 1; the effective prey speed is documented as the 160 px/s `PREY_VMAX` cap. |
| A5 | Original work; no plagiarism; legal right to license | ✅ Original benchmark; the BDH paper is cited, not reproduced. |
| A6 | Fits a served category | ✅ Fits `cs.LG` / `cs.NE` / `cs.AI`. Note: new arXiv accounts in `cs.*` may need an **endorsement** before the first submission — request one if prompted. |
| A7 | No duplicated/versioned look-alike submissions | ➖ Single submission. |
| A8 | ≤ 3 papers/day | ➖ One paper. |
| A9 | No offensive imagery | ➖ No images. |

## B. Format requirements

| Item | Requirement | Status |
|---|---|---|
| B1 | LaTeX (or PDF) source — **Markdown is not accepted** | ✅ `paper.tex` (11pt article, `amsmath`/`booktabs`/`hyperref`) compiles cleanly to `paper.pdf` (11 pages, letter, 1" margins; zero errors, zero undefined references). |
| B2 | Single-spaced, 10–14pt type, ≥ 1" margins | ✅ `\documentclass[11pt]{article}` + `geometry[margin=1in]`. |
| B3 | Machine readable | ✅ Standard LaTeX, no scanned text. |
| B4 | No line numbers, watermarks, highlighted text, margin notes, referee remarks, slides, or obstructive copyright statements | ✅ None present. |
| B5 | Complete references | ✅ 12 references (incl. SAC, Haarnoja et al.\ 2018); Frémaux & Gerstner cited uniformly as 2015; B. Widrow and BDH authors (Kosowski et al.) verified. |
| B6 | Code/data links must resolve to a **public** repository | ✅ Published at <https://github.com/llvm-x86/fast-weights-predict>; URL referenced in `paper.tex` and `paper.md`. |

## C. Generative-AI policy (arXiv requires reporting, not prohibition)

| Item | Requirement | Status |
|---|---|---|
| C1 | Report significant use of text-to-text generative AI | ✅ "Acknowledgments and disclosures" section added to both files. |
| C2 | AI tool not listed as an author | ✅ AI is not an author; the sole author is Gabriel Hill. |
| C3 | Authors take full responsibility for all content | ✅ Stated explicitly. |

## Ready-to-paste arXiv metadata

- **Title:** Fast Weights Predict, Not Evaluate: A Hebbian Associative World Model for Interception
- **Authors:** Gabriel Hill
- **Primary subject class:** `cs.LG` (Machine Learning). Cross-list `cs.NE` (Neural and
  Evolutionary Computing) and `cs.AI` (Artificial Intelligence).
- **License:** arXiv's default non-exclusive distribution license (or `CC BY 4.0` if you want
  open reuse -- your choice).
- **Comments (optional):** `11 pages, 3 tables. Code: https://github.com/llvm-x86/fast-weights-predict`

**Abstract (plain ASCII text, for the metadata field — arXiv metadata rejects Unicode):**

> The Dragon Hatchling (BDH) architecture is built on a fast-weight associative memory -- a
> matrix of stimulus-response associations maintained by outer-product plasticity and read out
> by similarity-weighted recall. We ask what this primitive is for: is it a value function, an
> evaluative map from states to expected return, or a world model, a predictive map from states
> to next states? We argue for the latter and settle the question empirically in a controlled
> interception task, where a turn-rate-limited pursuer must catch a moving target, with ten
> seeds per cell and Welch significance tests. Three findings emerge. (1) Used as a value
> function, the memory collapses: a linear TD learner with the same associative readout reaches
> 0-24 catches -- under 7% of the best analytic predictor in every environment -- and
> backpropagated DQN, PPO, and continuous-action SAC all fall below the no-prediction reflex on
> every environment. (2) Used as a world model with greedy lead-pursuit planning, the same
> memory reaches 87-99% of the best analytic predictor in every environment, and 92% of a
> hand-crafted circle-fitter on the curved prey that fitter specializes to. (3) On persistently
> curving prey -- where intercepting a trajectory, not a point velocity, is what matters -- the
> learned world model beats first- and second-order analytic extrapolation by 17% and 195%
> respectively, because rolling forward the learned dynamics is more accurate than a truncated
> Taylor series. The advantage is significant under a two-sided Welch test, survives a noise
> sweep (vanishing only where the motion becomes genuinely unpredictable), and a closed-loop
> imagination variant does not help -- pointing to a robust effect. We interpret the results as
> supporting a Dreamer-style division of labor: fast-weight memories are best understood as
> predictive substrates trained by dense self-supervised error, with sparse, outcome-driven
> planning on top, rather than as value-function approximators.

## Before you upload

1. Upload the **LaTeX source** (`paper.tex`, not the `.md`, and not the PDF alone) so arXiv can
   build it. The PDF has already been compiled and verified locally (`paper.pdf`, 11 pages).
2. If you get an endorsement prompt (common for new `cs.*` accounts), request an endorsement
   from an existing arXiv contributor.

All rubric items are now resolved. The code is at
<https://github.com/llvm-x86/fast-weights-predict>.
