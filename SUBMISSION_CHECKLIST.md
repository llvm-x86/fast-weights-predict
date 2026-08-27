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
| A2 | Carefully prepared sections, figures, tables, references | ✅ Title, Abstract, §1–§7, Table 1, 11 references, and a disclosures section are present. |
| A3 | Professional, sufficiently neutral tone | ✅ No promotional or extraneous content. |
| A4 | Originality, novelty, significance; no misrepresentation | ✅ Honest scope (toy domain stated explicitly). Fixed two internal inconsistencies: abstract said "~2% of *optimal* interception" (now "~2–10% of the *lead-pursuit ceiling*", matching the body), and the discretization ablation cited a no-reset reflex of ~1184 (now the correct reset-on-catch ~179). |
| A5 | Original work; no plagiarism; legal right to license | ✅ Original benchmark; the BDH paper is cited, not reproduced. |
| A6 | Fits a served category | ✅ Fits `cs.LG` / `cs.NE` / `cs.AI`. Note: new arXiv accounts in `cs.*` may need an **endorsement** before the first submission — request one if prompted. |
| A7 | No duplicated/versioned look-alike submissions | ➖ Single submission. |
| A8 | ≤ 3 papers/day | ➖ One paper. |
| A9 | No offensive imagery | ➖ No images. |

## B. Format requirements

| Item | Requirement | Status |
|---|---|---|
| B1 | LaTeX (or PDF) source — **Markdown is not accepted** | ✅ Added `paper.tex` (11pt article, `amsmath`/`booktabs`/`hyperref`). Compile it (`pdflatex paper.tex`, or drop into Overleaf) to produce the PDF for upload. |
| B2 | Single-spaced, 10–14pt type, ≥ 1" margins | ✅ `\documentclass[11pt]{article}` + `geometry[margin=1in]`. |
| B3 | Machine readable | ✅ Standard LaTeX, no scanned text. |
| B4 | No line numbers, watermarks, highlighted text, margin notes, referee remarks, slides, or obstructive copyright statements | ✅ None present. |
| B5 | Complete references | ✅ 11 references; fixed the "T. Widrow" → "B. Widrow" error, aligned the Dreamer year (2023), replaced a `[citations]` placeholder with real pursuit refs, and added the BDH authors (Kosowski et al.). |
| B6 | Code/data links must resolve to a **public** repository | ✅ Published at <https://github.com/llvm-x86/fast-weights-predict>; URL referenced in `paper.tex` and `paper.md`. |

## C. Generative-AI policy (arXiv requires reporting, not prohibition)

| Item | Requirement | Status |
|---|---|---|
| C1 | Report significant use of text-to-text generative AI | ✅ "Acknowledgments and disclosures" section added to both files. |
| C2 | AI tool not listed as an author | ✅ AI is not an author; human authorship is the (to-be-filled) byline. |
| C3 | Authors take full responsibility for all content | ✅ Stated explicitly. |

## Before you upload

1. Compile `paper.tex` to PDF and upload the **LaTeX source** (not the PDF alone, and never the
   `.md`) so arXiv can build it.
2. If you get an endorsement prompt (common for new `cs.*` accounts), request an endorsement
   from an existing arXiv contributor.

All rubric items are now resolved. The code is at
<https://github.com/llvm-x86/fast-weights-predict>.
