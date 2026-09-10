# world-dreamer

A *world dreamer* is a **world model** plus a **planner that optimizes inside it**:

- **World model** — a predictive map `state → next state`, trained densely on
  self-supervised prediction error (never on reward).
- **Dreamer** — a planner that *imagines* candidate plans inside the world model,
  scores them against the model, and executes the best. Reward is sparse and only
  touches the planner.

This repository instantiates that two-part skeleton on **two different substrates**,
to make the scope of the idea explicit:

| | world model (substrate) | dreamer (planner) |
|---|---|---|
| **continuous** (`pursuit.py`) | the Dragon Hatchling (BDH) fast-weight memory: `φ(s) → vₜ₊₁`, a linear Hebbian map over continuous state | receding-horizon shooting over aim headings, scored by imagined time-to-catch |
| **discrete** (`arc.py`) | an induced **program**: a map from an input grid to an output grid, composed of discrete transformation primitives | **program search**: hypothesize programs, *imagine* their output on every example, keep the one that predicts all observations |

The substrate is the only thing that differs. The architecture — *predict, then
plan inside the prediction* — is shared. That is the honest meaning of "a
generalized world dreamer": a **recipe**, not a single trained network that
transfers from pursuit to ARC.

## The continuous half (pursuit)

`pursuit.py` adapts the pursuit benchmark (`../pursuit/bench.py`) into the shared
`framework.py` interface. The world model is the BDH fast-weight memory; the
dreamer is `bench._mpc_aim`, the shooting loop used for Result 7 / Table 8 of the
paper. The full, significance-tested numbers live there. A short demo:

```bash
python3 pursuit.py          # ~2,000 steps of the pursuit loop through the framework
```

On predictable prey the dreamer beats the analytic lead (up to ~40% on curved
motion); on reactive prey it matches — but does not exceed — the short-lead
baseline, because the dreamer inherits the world model's own ceiling: the
velocity-decorrelation time. See `../pursuit/paper.md` §5.10 for the honest
accounting.

## The discrete half (ARC)

`arc.py` is the discrete analog: the world model is the induced program, and the
dreamer is program search (hypothesize → imagine on examples → verify). The
primitive library covers two kinds of transformation:

- **geometry** — identity, rotation, reflection, translation, recolor, scaling,
  tiling, substitution tiling, flood-fill, crop-to-object;
- **objectness + completion** (v2) — 4-connected object decomposition, gravity
  (four directions), mirror-image completion across the central axis, connecting
  same-colored points (orthogonal and diagonal), dilation, crop/remove the
  largest or smallest object, keep-only-one-color, recolor-objects-by-size, and
  per-object map transforms (flip / rotate each object in place);
- **painting + structure + palette** (v4) — whole-palette permutation learned
  across all examples (guarded to be bijective so it cannot collapse two colors
  onto one), border painting, dominant-color flood, symmetry-axis completion
  about a detected (half-integer) axis, and structure extraction/erasure: keep or
  erase the central row / column / cross / diagonals, checkerboard-by-parity from
  the background hole, and highlight monochromatic rows/columns;
- **numerosity + unit extraction** (v5) — count connected components into a
  row / column / diagonal of cells, single-cell numerosity (least/most frequent
  color, and the scattered 'noise' color: least frequent, ties → most
  components), crop the top-left corner to output size, 2-D periodic tiling, and
  drawing a line between same-colored points with a *new* color (filling only
  background cells, so intermediate same-colored points are preserved);
- **rectangle construction** (v6/v7) — draw the rectangle outline of the overall
  bounding box, draw a rectangle outline around *each* object, fill a bounding
  box, fill *each* object's bounding box (8-connected, because the shapes that
  need this rule touch only at a corner), and draw a full cross through each
  object's centre.  All painting touches background cells only, so object pixels
  survive.
- **layout, rank and symmetry** (v8) — 2×2 mirror tiling (plain, left-right,
  top-bottom, fourfold); the four quadrants of a 2×2 layout combined cellwise
  (or / and / xor / majority) or overlaid in a chosen order; equal-shaped panels
  delimited by a separator row/column combined cellwise, with the separator
  colour swept; stripping the separator lines; point-symmetry completion about
  the centre; erasing isolated cells that belong to no object; per-object
  outlines (4- and 8-connected, either replacing the object or painted on top of
  it); cropping to the content bounding box and then scaling; and recolouring
  every object by the *rank* of its size, with the rank palette induced from the
  first example and then verified on all of them (a rank rule survives examples
  whose size sets differ, which a size→colour map cannot).

- **a generic per-object operator** (v9) — `map_objects(f)` applies a
  size-preserving primitive `f` to *every object's own bounding-box subgrid* and
  writes it back in place. `map_flip`/`map_rotate` were the two hand-written
  special cases of this; opening it up makes per-object gravity, per-object
  mirror and point completion, per-object cross and diagonal carving, and
  dilation expressible for the first time, from one operator plus 25 inner
  primitives. It is the generic form of ARC's largest family, "apply this
  transformation to every shape";
- **panel reordering** (v9) — sort the separator-delimited panels of a grid by
  their content (cell count, sum, lexicographic order, distinct colours, ascending
  or descending) and write them back into the slots they came from. The layout
  carries no information and only the *order* of the panels is the answer. Unlike
  the crop-by-property families, this selects nothing — it is a total permutation
  of the panels in fixed slots, so it cannot be right for the wrong reason.

Two search details are worth stating because they decide results:

- **Which hypothesis wins.** Several distinct primitives can reproduce every
  training example of some task, and the search returns the first that does. The
  pair-learned colour maps (`recolor_by_size`, `recolor_by_rank_*`,
  `recolor_map`) are therefore searched in a *second* pass, and inside that pass
  the first step is ordered by how close its colour set already is to the
  target's, so an intermediate that already carries the target palette is
  preferred over one that only reaches it by re-colouring. Both passes verify
  against every training example, so this changes *which* verified program is
  returned, never *whether* one is.
- **The size gate, and why it is gone.** An intermediate grid used to be expanded
  only if a single primitive could plausibly take it to the target size. That
  filter is unsound: `tile_2d` re-tiles the detected row/column period to
  *exactly* the target size from any input, so the reachable size-relation set is
  universal and the only complete filter is no filter. An audit over both
  training sets checked 256,551 distinct intermediates, found the gate rejecting
  2.2% of them, and re-ran **all 1,211** unsolved training tasks with the gate
  forced open: none became solvable, so the fix is score-neutral — but it is the
  difference between a search that is complete with respect to its primitive
  library and one that is not, and dropping the gate measured 0.98x, i.e. free.

The dreamer is a **program-composition search**: depth-1 primitives, plus
depth-2 and depth-3 compositions `f3 ∘ f2 ∘ f1` where intermediate steps are
size-preserving/shrinking primitives (deduplicated by the intermediate grid) and
the final step is matched to the target. Every candidate program is verified
against **all** training examples before it is used on the held-out test, so a
wrong rule is filtered out rather than guessed. Evaluation is parallel across
the available cores, matched exactly to the CPU affinity (8 on the machine these
numbers come from), because oversubscribing the worker pool was measurably
slower.

### Search performance

The search is the bottleneck, so it is worth stating what a sweep costs. A full
ARC-AGI-1 training pass (max_depth=2, 400 tasks, 8 workers, one process per core)
went from **2m55s to 33s** — a **5.3x** wall-clock and 5.5x CPU-time reduction —
with identical results on a 176-task sample drawn from all four datasets. The
changes, all behaviour-preserving:

- candidate grids are compared to the target with an early-exit, row-by-row test
  instead of building a nested tuple for every candidate that fails, which is
  almost all of them;
- the ±6 translation sweep and the reflection-axis sweep test the predicate
  directly, cell by cell, instead of materialising a grid per candidate;
- the connected-component decomposition is memoised, since a single depth-1
  sweep asks for the same grid's components many times over (numerosity, crop,
  denoise, recolour);
- `_bg` returns on the first row containing black rather than counting the whole
  grid; and
- primitive sweeps that take a colour parameter only try colours that occur in
  the target (a fill or a border paints with that colour), and the numerosity
  primitives are gated on the single output shape each of them can produce.

A whole four-dataset evaluation now takes about 3.5 minutes on 8 cores, which is
what makes a parallel search over solver *variants* practical: each variant works
in its own copy of the tree, is scored on a fixed seeded sample of all four
datasets, and only the variants that survive are re-measured on the full sets.

**Result (honest, transparent), exact match on the held-out test output:**

| search depth | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| depth-1 | 56 / 400 (14.0%) | 63 / 1,000 (6.3%) |
| depth-2 (default) | **84 / 400 (21.0%)** | **107 / 1,000 (10.7%)** |

<p></p>

| dataset | solved | fits every training example but misses the test |
|---|---|---|
| ARC-AGI-1 training | 84 / 400 | 2 |
| ARC-AGI-2 training | 107 / 1,000 | 4 |
| ARC-AGI-1 evaluation | 24 / 400 | 0 |
| ARC-AGI-2 evaluation | 0 / 120 | 0 |

The last column is reported deliberately. A program that reproduces *every*
training example and still misses the held-out test is the honest signature of
program induction under-determined by its examples, and on this DSL it happens
rarely (2 and 4 tasks on the training sets) — but it does happen, and hiding it
would overstate how much the verification step actually guarantees. It costs no
score: ARC scores a wrong answer exactly as it scores no answer.

```bash
python3 eval_arc.py /tmp/arc-agi/data/training 8 2      # ARC-AGI-1, depth 2 (default)
python3 eval_arc.py /tmp/arc-agi/data/training 8 3      # ARC-AGI-1, depth 3
python3 eval_arc.py /tmp/ARC-AGI-2/data/training 8 2    # ARC-AGI-2
```

The solves are single-transformation, single-object, and short-composition tasks
(`rotate`, `translate`, `scale(2)`, `tile`, `self_substitute`, `fill_holes`,
`mirror`, `gravity`, `connect`, `crop_largest`, `recolor_by_size`, the v4
palette permutation, border/dominant painting, symmetry-axis completion,
structure keep/erase (cross, diagonal, mid-row/column), checkerboard, the v5
single-cell numerosity and connect-with-new-color rules, the v6/v7 rectangle
construction (`draw_bbox_outline`/`map_bbox_outline`/`fill_bbox_region`/
`map_bbox_fill`/`draw_object_cross`), and the v8 layout rules (mirror tiling,
quadrant logic, separator panels, point-symmetry completion, per-object outlines,
rank recolouring), and the v9 operators below — plus two-step combinations of
them). The other ~316 tasks are compositional, relational, numerosity, and
sequence-extrapolation tasks that a hand-written primitive DSL with shallow
search does not reach —
which is precisely where ARC's difficulty lies, and where the ARC-AGI-3 frontier
(symbolic world modeling / program search at scale) is aimed.

### Why the evaluation sets fail: a capability gap, not a search failure

This is worth separating, because the two failures call for completely different
fixes. For every task the solver gets wrong on either evaluation set, we asked a
sharper question than "is it solved": **is there any depth-1 primitive that
reproduces the output of every training example?** If yes, the search found a
hypothesis and the task is an *ambiguity* failure (the examples do not pin the
rule down). If no, it is a *capability* failure (the DSL cannot express the rule
at all, and depth-2 cannot help, since it composes the same primitives).

| evaluation set | solved | unsolved | …with a train-fitting depth-1 primitive |
|---|---|---|---|
| ARC-AGI-1 | 24 | 376 | **0** |
| ARC-AGI-2 | 0 | 120 | **0** |

Not one. Every remaining failure is a rule the primitive library cannot express,
not a rule the search failed to find. That is the honest reading of these
numbers, and it is also why the training-set rate is four times the evaluation
rate: the training sets contain many tasks whose rule *is* in the library, and
the evaluation sets were built to exclude exactly those. Adding more hand-written
primitives to this DSL is therefore not the lever — the discussion below is about
what is.

### Does it generalize to ARC-AGI-2? (honest, measured)

No — not in the "solved" sense, and it would be misleading to claim otherwise.
Measured on the ARC-AGI-2 public training set (1,000 tasks), the exact same
pipeline scores **102 / 1,000 (10.2%)**, down from 20.5% on ARC-AGI-1. ARC-AGI-2
was designed to remove the single-transformation tasks this DSL catches and to
stress compositional object/relation reasoning, so the number drops — exactly as
expected. The per-object map transforms and the
v4/v5/v6/v7/v8 painting, structure, numerosity, rectangle and layout primitives
are aimed at that core and recover a growing handful of tasks, but the
composition/relation core remains out of reach for a shallow hand-written DSL.

**On the held-out *evaluation* sets (the real benchmarks) the combined system
scores 24 / 400 (6.0%) on ARC-AGI-1 and 0 / 120 (0.0%) on ARC-AGI-2.** The 24
ARC-AGI-1 solves come from the hand-written DSL; the learned patch memory
contributes **zero** to either evaluation set. These are the numbers that answer
the "does it generalize" question, and the answer is: barely on ARC-AGI-1 and
not at all on ARC-AGI-2. The training-set figures above are optimistic by
roughly a factor of four — they cover the single-transformation and
short-composition tasks that the evaluation sets were explicitly built to
exclude. The evaluation tasks are larger, more colorful, and
compositional/relational, and neither the hand-written DSL nor the patch
associative memory induces them. Any claim that this repository "solves ARC-AGI-1
or ARC-AGI-2" would be false.

What *does* generalize is the architecture, not the primitives: the same
world-model-plus-dreamer recipe (predict, then plan inside the prediction) is
instantiated on both substrates and on both ARC benchmarks. Closing the gap to
ARC-AGI-2 is a program-induction research problem (LLM-guided program synthesis
with a verifier-in-the-loop, or a much larger object-centric DSL with deep
search), not a matter of adding more hand-written primitives.

### The ceiling is the dreamer, not the substrate (demonstrated)

`llm_dreamer.py` keeps the identical architecture but swaps the *dreamer*: instead
of enumerating primitives, a language model reads the examples, induces a rule,
and writes a program, which the world-model check then verifies. On ARC-AGI-2 it
solves 4/4 relational tasks that the primitive DSL fails — a shape-keyed recolor,
a split-and-overlap, a vertical pattern continuation, and a mirror-tile — each in
a few lines of induced code, verified against every training and held-out test
example. That is the honest demonstration that the world-dreamer's ceiling is set
by the planner, and that the route to high ARC scores is synthesis, not more
hand-coded primitives. It is also *not* a 90% system: it is manual induction on a
handful of tasks, not an automated solver.

### The learned (non-LLM) substrate — `learned.py`

The literal dragon-hatchling: the BDH fast-weight memory applied directly to ARC
grids. Features are position-dependent one-hot cell colors; the write is the
Hebbian outer product `W ← W + η ψ(output) φ(input)ᵀ`; the readout is the linear
map plus an argmax color per cell (computed without materializing the dense
matrix).

Seven fast-weight world models are learned — a position-bound map, a global
color-cooccurrence map, and five geometries of a **3×3 patch associative memory**
(`LocalRuleModel`) that keys on the local input neighborhood instead of a single
cell, so a rule learned at one location generalizes to every location. Three
details matter and each was measured: patches on the grid border **reflect the
grid across the edge** rather than reading a sentinel, so border patches hold
real colors; a patch and its eight dihedral transforms **share vote counts**; and
the *dreamer* selects among the seven by leave-one-*example*-out cell error plus
the full-training-set cell error, because leave-one-out alone rewards a model
that predicts "almost the input". Measured honestly on the same-size subset
(size-changing tasks are reported as skipped, not hidden):

| benchmark | same-size tasks | solved (same-size) | solved (all tasks) |
|---|---|---|---|
| ARC-AGI-1 | 130 | **15 (11.5%)** | **15 / 400 (3.8%)** |
| ARC-AGI-2 | 258 | **17 (6.6%)** | **17 / 1,000 (1.7%)** |

A second, **object-level** memory was then added, and it is the honest answer to
"can an associative memory do ARC". It decomposes the input into objects, keys
each by its (translation- and optionally dihedral-invariant) tight subgrid, learns
the output object together with its *relative offset*, and paints it back — so
this model is not restricted to same-size tasks. An unseen key leaves that object
untouched, and a size-changing task is answered only when the memory reproduces
**every** training pair exactly both full-fit and leave-one-out. The measured
reason its contribution is small is the interesting part: 72 of 129 same-size and
74 of 271 size-changing training tasks are reproduced *perfectly* on the full
training fit, but only 5 and 1 survive leave-one-out — the rest are memorizers
whose held-out object never recurs. Over half the failing objects have a
never-seen key, and of the 74 size-changing tasks whose answer is a single object,
none has an object patch that recurs across leave-one-out folds. Dropping the
leave-one-out gate would take the learned model from 15 to 18 on ARC-AGI-1
training while answering 270 tasks wrongly, so the gate stays: the "pick the odd
object out" family is structurally out of reach for a lookup table.

The patch memory is what lifts the number: it expresses the *local* ARC family
(cellular automata, region fill, symmetry completion) that neither the
position-bound map nor the global color map can reach. It is still single
digits, and that is the honest point: an associative memory — even a patch-level
one — cannot induce the compositional relational rules that are most of ARC, and
it adds nothing at all on either evaluation set. As the DSL has grown, the
overlap between the two has grown with it: the learned model now contributes
ten tasks that no induced program reaches on ARC-AGI-1 training and eleven on
ARC-AGI-2 (`0ca9ddb6`, `4258a5f9`, `a8d7556c`, `a9f96cdd`, `b60334d2`,
`b6afb2da`, `ce22a75a`, `d364b489`, `5c0a986e`, `6c434453`, plus `ad38a9d0` on
ARC-AGI-2) — up from two, after the patch memory gained
edge-reflecting padding, dihedral-shared patch keys, a selection score that also
punishes overfitting the training pair, and the object-level memory above. On
**both evaluation sets it still adds exactly zero**. One task it used to own,
`543a7ed5`, is traded away in return; its correct model has strictly worse
held-out error, and no weighting recovers it without losing two others.

### The combined substrate — `combined.py`

The two non-LLM substrates solve *disjoint* slices, so the honest realization of
"proceed with A & B in parallel" is an ensemble: try the induced program first
(exact and verified), and fall back to the learned patch map when no program
verifies. Measured as a union on the held-out test:

| benchmark | DSL alone | learned alone | **combined** |
|---|---|---|---|
| ARC-AGI-1 | 84 / 400 (21.0%) | 15 / 400 (3.8%) | **94 / 400 (23.5%)** |
| ARC-AGI-2 | 107 / 1,000 (10.7%) | 17 / 1,000 (1.7%) | **118 / 1,000 (11.8%)** |

The same point holds when the dreamer is *automated*: `verify_solution.py` runs a
language-model proposer (a solver agent per task) against the verifier. On an
unbiased 8-task sample of ARC-AGI-2 tasks the primitive DSL scores 0/8 on, a
single-shot LLM proposal (with one human repair) solves **5/8 (62.5%)** —
`3b4c2228` (count 2×2 blocks), `6fa7a44f` (append vertical flip), `a57f2f04`
(texture-fill), `bc4146bd` (mirror-scale), `00576224` (mirror-tile). Three harder
tasks (rearrangement, denoising, region-fill) were not solved in budget. That is
~14× the DSL on this sample, but a 5-task sample is not a benchmark score, and it
is still far from 90%.

## What this is not

This is **not** "an agent that solves ARC-AGI-1." No system "solves" ARC-AGI-1 in
the sense of reliably matching the human baseline (~84% on the private evaluation,
higher with retries); the compositional program-induction core remains an open
research problem. This repository demonstrates that the *same*
world-model-plus-dreamer architecture that works on continuous pursuit can be
re-instantiated on discrete program induction, with transparent coverage numbers.
The bridge from here to genuinely hard ARC is a much richer program-induction
substrate (and a search that composes over it), not a reuse of the pursuit
fast-weight matrix.

## Files

- `framework.py` — the shared `WorldModel` / `Dreamer` / `WorldDreamer` interface.
- `pursuit.py` — the continuous instantiation (BDH fast weights + shooting).
- `arc.py` — the discrete instantiation (program induction + program search).
- `eval_arc.py` — ARC evaluation harness (parallel; pass the core count, e.g. 8).
  This is the authoritative measurement; the two below are triage tooling.
- `sample_eval.py` — seeded fixed-sample scoring, for comparing solver variants
  quickly (a triage signal only — it can miss a variant's gain entirely).
- `variant.py` — spawn isolated copies of the tree for concurrent experiments,
  score them on the fixed sample, and diff them against the canonical solver.
- `llm_dreamer.py` — the LLM-as-dreamer demonstration (induced, verified rules).
- `learned.py` — the non-LLM learned fast-weight substrate (BDH on ARC grids).
- `combined.py` — the faithful ensemble: induced program first, learned patch map fallback.
- `verify_solution.py` — verifies a proposed `solve()` against an ARC task.
