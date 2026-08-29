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
  largest or smallest object, keep-only-one-color, and recolor-objects-by-size.

plus a two-step "extract the object, then transform it" composition. Every
candidate program is verified against **all** training examples before it is used
on the held-out test, so a wrong rule is filtered out rather than guessed.

**Result (honest, transparent): 32 / 400 (8.0%) of the ARC-AGI-1 training set,
exact match on the held-out test output.**

```bash
python3 eval_arc.py /tmp/arc-agi/data/training
```

The 32 solves are the single-transformation and single-object tasks (`rotate`,
`translate`, `scale(2)`, `tile`, `self_substitute`, `fill_holes`, `mirror`,
`gravity`, `connect`, `crop_largest`, …). The other 368 tasks are compositional,
relational, numerosity, and sequence-extrapolation tasks that a hand-written
primitive DSL does not reach — which is precisely where ARC's difficulty lies, and
where the ARC-AGI-3 frontier (symbolic world modeling / program search at scale)
is aimed.

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
- `eval_arc.py` — ARC-AGI-1 evaluation harness.
