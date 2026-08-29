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
  row / column / diagonal of cells, crop the top-left corner to output size,
  2-D periodic tiling, and drawing a line between same-colored points with a
  *new* color.

The dreamer is a **program-composition search**: depth-1 primitives, plus
depth-2 and depth-3 compositions `f3 ∘ f2 ∘ f1` where intermediate steps are
size-preserving/shrinking primitives (deduplicated by the intermediate grid) and
the final step is matched to the target. Every candidate program is verified
against **all** training examples before it is used on the held-out test, so a
wrong rule is filtered out rather than guessed. Evaluation is parallel across 12
processes (the tasks are pure Python, so the GIL rules out threads).

**Result (honest, transparent), exact match on the held-out test output:**

| search depth | ARC-AGI-1 | ARC-AGI-2 |
|---|---|---|
| depth-1 | 41 / 400 (10.2%) | 43 / 1,000 (4.3%) |
| depth-2 (default) | **51 / 400 (12.8%)** | **58 / 1,000 (5.8%)** |

```bash
python3 eval_arc.py /tmp/arc-agi/data/training 12 2     # ARC-AGI-1, depth 2 (default)
python3 eval_arc.py /tmp/arc-agi/data/training 12 3     # ARC-AGI-1, depth 3
python3 eval_arc.py /tmp/ARC-AGI-2/data/training 12 2   # ARC-AGI-2
```

The solves are single-transformation, single-object, and short-composition tasks
(`rotate`, `translate`, `scale(2)`, `tile`, `self_substitute`, `fill_holes`,
`mirror`, `gravity`, `connect`, `crop_largest`, `recolor_by_size`, the v4
palette permutation, border/dominant painting, symmetry-axis completion,
structure keep/erase (cross, diagonal, mid-row/column), checkerboard, and
highlight-uniform-lines — plus two-step combinations of them). The other ~350
tasks are compositional, relational, numerosity, and sequence-extrapolation
tasks that a hand-written primitive DSL with shallow search does not reach —
which is precisely where ARC's difficulty lies, and where the ARC-AGI-3 frontier
(symbolic world modeling / program search at scale) is aimed.

### Does it generalize to ARC-AGI-2? (honest, measured)

No — not in the "solved" sense, and it would be misleading to claim otherwise.
Measured on the ARC-AGI-2 public training set (1,000 tasks), the exact same
pipeline scores **58 / 1,000 (5.8%)**, down from 12.8% on ARC-AGI-1. ARC-AGI-2 was
designed to remove the single-transformation tasks this DSL catches and to stress
compositional object/relation reasoning, so the number drops — exactly as
expected. The per-object map transforms and v4/v5 painting/structure/numerosity
primitives are aimed at that core and recover a handful of tasks, but the
composition/relation core remains out of reach for a shallow hand-written DSL.

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

Three fast-weight world models are learned — a position-bound map, a
position-invariant color map, and (v2) a **3×3 patch associative memory**
(`LocalRuleModel`) that keys on the local input neighborhood instead of a single
cell, so a rule learned at one location generalizes to every location — and the
*dreamer* selects the one that generalizes best by leave-one-out error on the
training set (so a model that merely memorizes is rejected). Measured honestly on
the same-size subset (size-changing tasks are reported as skipped, not hidden):

| benchmark | same-size tasks | solved |
|---|---|---|
| ARC-AGI-1 | 129 | **4 (3.1%)** |
| ARC-AGI-2 | 257 | **2 (0.8%)** |

The patch memory is what lifts the number: it expresses the *local* ARC family
(cellular automata, region fill, symmetry completion) that neither the
position-bound map nor the global color map can reach, recovering three tasks the
symbolic DSL also misses (`3618c87e`, `543a7ed5`, `b6afb2da`). It is still single
digits, and that is the honest point: an associative memory — even a patch-level
one — cannot induce the compositional relational rules that are most of ARC.

### The combined substrate — `combined.py`

The two non-LLM substrates solve *disjoint* slices, so the honest realization of
"proceed with A & B in parallel" is an ensemble: try the induced program first
(exact and verified), and fall back to the learned patch map when no program
verifies. Measured as a union on the held-out test:

| benchmark | DSL alone | learned alone | **combined** |
|---|---|---|---|
| ARC-AGI-1 | 51 / 400 (12.8%) | 4 / 129 (3.1%) | **54 / 400 (13.5%)** |
| ARC-AGI-2 | 58 / 1,000 (5.8%) | 2 / 257 (0.8%) | **59 / 1,000 (5.9%)** |

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
- `eval_arc.py` — ARC evaluation harness (parallel, 12 processes).
- `llm_dreamer.py` — the LLM-as-dreamer demonstration (induced, verified rules).
- `learned.py` — the non-LLM learned fast-weight substrate (BDH on ARC grids).
- `combined.py` — the faithful ensemble: induced program first, learned patch map fallback.
- `verify_solution.py` — verifies a proposed `solve()` against an ARC task.
