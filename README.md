# Fast-weight associative memory as a *world model* for interception

A seeded, multi-environment pursuit benchmark investigating whether the Dragon
Hatchling (BDH) fast-weight memory is better used as a **world model** (predict
state → next-state, then plan) than as a **value function**, and whether a
learned world model beats analytic extrapolation for chasing a moving target.

## Thesis

The BDH paper's core primitive is the fast-weight associative memory — a matrix
of stimulus→response associations maintained by outer-product plasticity
(`W ← W + η·δ·xᵀ`), read out by similarity-weighted recall (`ŷ = W·x`). We argue,
and show empirically, that this primitive is a **predictive / working-memory**
substrate, not a stable value-function approximator:

1. **World model ≫ value function.** Used to predict prey motion (a world model)
   with greedy lead-pursuit planning, it reaches ~94% of optimal interception.
   Used as a value function (linear TD, or a backprop DQN), it collapses to
   2–50% of optimal — the "deadly triad" (function approximation + bootstrapping
   + off-policy) plus a turn-rate-limited, discretized action space.
2. **World model > analytic extrapolation on curved prey.** On persistently
   curving prey, the learned model beats both first-order (velocity-lead) and
   second-order (accel-lead) extrapolation, because rolling forward the learned
   dynamics is more accurate than a truncated Taylor series (the ½aτ² parabola
   overshoots a circle over long lead horizons).

The split — dense world-model learning (self-supervised prediction error) plus
greedy planning, driven by sparse catch outcomes — is exactly the Dreamer-V4
structure, instantiated with a biologically-plausible three-factor Hebbian rule.

## Environment

- Toroidal world 1200×800 px, `dt = 0.05 s`, catch radius 48 px, cooldown 0.5 s.
- Chaser: constant top speed 175 px/s, turn rate clamped to 3.6 rad/s.
- Prey: 165 px/s (comparable to the chaser, so interception — not out-running —
  is what matters). Five dynamics:
  - `const-vel` — linear (first-order is Bayes-optimal; control).
  - `circling` — constant turn rate → arcs/circles (**where prediction matters**).
  - `ou-turn`, `ou-vel` — stochastic (Martingale; future is unpredictable).
  - `jump` — velocity-jump process.
  - `flee` — reactive predator avoidance (steers away, faster when approached).
- Protocol: **reset-on-catch** — each catch teleports the prey to a random far
  location, so every catch is a fresh interception from distance (no "lock-on"
  saturation). Metric: catches per episode, mean ± sd over seeds.

## Agents

| agent | type | mechanism |
|---|---|---|
| velocity-lead | predictor | `p + v·τ` (first-order extrapolation) |
| accel-lead | predictor | `p + v·τ + ½a·τ²` (second-order Taylor) |
| kalman-lead | predictor | constant-velocity Kalman filter + lead |
| **bdh** | predictor | Hebbian fast-weight world model + imagination |
| pure-pursuit | policy | steer at current prey (reflex, no prediction) |
| MPC | policy | model-based search over 8 headings (first-order prey model) |
| linear-Q | policy | linear FA + TD(0) + ε-greedy (the deadly triad) |
| DQN | policy | MLP Q-learning (replay + target net + Adam) |
| PPO | policy | clipped-surrogate PPO (policy + value MLPs) |

The **bdh** world model is a linear content-addressable associative memory
(`W: [2×D]`, `ŷ = W·φ(s)`) over normalized features `[rel_x, rel_y, v_x, v_y]`,
updated by the normalized three-factor (error-gated) Hebbian rule
`W ← W + η·(y − ŷ)·φ(s)ᵀ/(1 + ‖φ‖²)`, and rolled forward τ steps (imagination)
to forecast the future prey position for lead pursuit. It is trained on every
observed transition (dense, self-supervised), and the policy (steer at the
forecast) is driven only by catch outcomes.

## Reproducing

```bash
node bench.js <horizon> <prey...>       # predictors + policies (SEEDS, RESET, TURN, PREY_SPEED, NACT, BDH_M, BDH_ETA env)
SEEDS=5 RESET=1 node bench.js 24000 circling
node results.js                          # full mean±sd tables → results.md
```

## Key results

All numbers below are **reset-on-catch** (mean ± sd over seeds), the protocol
that measures genuine interception from distance. Predictors: 10 seeds × 24000
steps; policies: 3 seeds × 20000 steps.

### 1. World model vs value function (the headline)

The BDH world model reaches ~94% of the lead-pursuit ceiling. The *same* memory
as a value function (linear-Q) collapses to ~0, and a backprop DQN does not even
beat the no-prediction reflex (pure-pursuit):

| prey | pure-pursuit (reflex) | linear-Q (deadly triad) | DQN | PPO | **bdh world model** |
|---|---|---|---|---|---|
| const-vel | 60 ± 10 | 19 ± 6 | 30 ± 28 | 12 | **363 ± 53** |
| circling | 179 ± 52 | 22 ± 5 | 41 ± 5 | 21 | **384 ± 38** |
| ou-turn | 129 ± 10 | 25 ± 6 | 29 ± 4 | — | **330 ± 11** |
| ou-vel | 137 ± 11 | 27 ± 5 | 29 ± 3 | — | **364 ± 12** |
| jump | 254 ± 8 | 20 ± 2 | 44 ± 9 | — | **298 ± 11** |
| flee | 160 ± 1 | 0 ± 1 | 37 ± 9 | — | **165 ± 5** |

DQN and PPO are given dense reward (−distance) plus a catch bonus; they are
still *worse* than the no-prediction reflex. (Without reset DQN climbs to
~30–50% of the reflex after 40000 steps — it never exceeds it.) The failure is
robust across action discretization (4/8/16 headings): linear-Q and DQN stay at
~22–46 catches while the reflex scores ~179 (reset-on-catch).

The clean ordering (reset, 16000 steps): pure-pursuit ~49–143 < **MPC ~195–206
(first-order model, discrete search)** < velocity-lead ~327–364 (first-order,
continuous lead) < **bdh ~384 (learned world model)** — prediction and the
learned model both matter, and planning with a model decisively beats
model-free learning.

### 2. World model vs analytic extrapolation on curved prey

| prey | velocity-lead | accel-lead | kalman-lead | **bdh** |
|---|---|---|---|---|
| const-vel | 364 ± 57 | 364 ± 57 | 148 ± 50 | **363 ± 53** |
| circling | 327 ± 35 | 131 ± 192 | 35 ± 66 | **384 ± 38** |
| ou-turn | 368 ± 17 | 364 ± 10 | 116 ± 7 | 330 ± 11 |
| ou-vel | 388 ± 8 | 385 ± 15 | 136 ± 16 | 364 ± 12 |
| jump | 342 ± 10 | 293 ± 8 | 133 ± 10 | 298 ± 11 |
| flee | 182 ± 2 | 69 ± 7 | 6 ± 3 | 165 ± 5 |

- **circling**: bdh **+17%** over first-order, **+193%** over second-order.
  Second-order extrapolation is also wildly unreliable (sd 192 vs bdh's 38) —
  the ½aτ² parabola overshoots the circle over long lead horizons.
- linear (`const-vel`): ties first-order (expected; first-order is exact).
- stochastic (`ou-*`, `jump`): no predictor beats first-order; the model adds
  small variance (honest negative).
- reactive (`flee`): comparable, open-loop (frozen-chaser) imagination is the
  documented limitation.

### 3. Ablations

- **Turn rate** (1.8→5.4 rad/s): bdh > velocity-lead at every turn rate
  (262 vs 219, 288 vs 273, 306 vs 280, 312 vs 283).
- **Speed ratio**: bdh robust (321.8@150, 306.4@165, 294.2@180); second-order
  only competes at slow prey (336.0@150) and collapses at high speed (140@165+),
  because its parabola overshoots.
- **Memory width (Fourier frequencies M)**: linear (M=0) is optimal (306.4);
  nonlinear expansion does not help on smooth dynamics (253–303).
- **Learning rate η**: slow learning is best (η=0.1 → 318.6); all η beat
  first-order (280).
- **Weight decay λ**: moderate (1e-3) is optimal (306.4); extreme (1e-2) hurts
  (285.4); all λ beat first-order (280).
- **Action discretization (4/8/16 headings)**: model-free methods stay at ~15–22
  catches vs the reflex's ~179 — the failure is *not* a discretization artifact
  (it persists at 16 actions).

## Honest scope / limitations

- The result is strongest on **curved (persistent-turn) prey** — the case where
  predicting a *trajectory* (not a point velocity) matters. On Martingale
  (unpredictable) motion no predictor can beat first-order; on reactive prey the
  open-loop imagination is a known gap.
- The world model is a **linear** associative memory (the nonlinear Fourier
  expansion did not help). The BDH "Hebbian" claim is the three-factor
  (error-gated) outer-product update — an honest, standard reward-modulated rule.
- Not yet included: SAC (continuous control), a circle-fit analytic baseline,
  walled environments, and multi-prey. These (plus a cleaner closed-loop
  imagination for reactive prey) are the remaining items needed to call this a
  full paper rather than a technical report.

## Files

- `bench.js` — seeded environment, predictors (velocity/accel/kalman/bdh) and
  model-free policies (pure-pursuit, linear-Q, DQN), forecast-error metric,
  reset-on-catch protocol, ablation knobs.
- `results.js` — computes the mean±sd tables and writes `results.md`.
