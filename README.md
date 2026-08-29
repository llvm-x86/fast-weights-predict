# Fast Weights Predict, Not Evaluate: A Hebbian Associative World Model for Interception

A seeded, multi-environment pursuit benchmark investigating whether the Dragon
Hatchling (BDH) fast-weight associative memory is better used as a **world model**
(predict state → next-state, then plan) than as a **value function**, and whether a
learned world model beats analytic extrapolation for intercepting a moving target.

`paper.md` / `paper.tex` is the full write-up; `results.md` holds the reproducible
result tables.

## Thesis

The BDH architecture's core primitive is a fast-weight associative memory — a matrix
of stimulus→response associations maintained by outer-product plasticity
(`W ← W + η·δ·xᵀ`), read out by similarity-weighted recall (`ŷ = W·x`). We argue, and
show empirically, that this primitive is a **predictive / working-memory** substrate,
not a stable value-function approximator:

1. **World model ≫ value function.** Used to predict prey motion (a world model) with
   greedy lead-pursuit planning, it stays within 14% of the best analytic predictor per
   environment. Used as a value function (linear TD, backprop DQN, PPO, or SAC), it
   falls *below the no-prediction reflex* in every environment.
2. **World model > analytic extrapolation on curved prey.** On persistently curving
   prey, the learned model beats first-order (velocity-lead) by +18% and second-order
   (accel-lead) by +196%, because rolling forward the learned dynamics is more accurate
   than a truncated Taylor series (the ½aτ² parabola overshoots a circle).

The split — dense world-model learning (self-supervised prediction error) plus greedy
planning, driven by sparse catch outcomes — is the Dreamer structure instantiated with
a biologically plausible three-factor Hebbian rule.

## Environment

- Toroidal world 1200×800 px, `dt = 0.05 s`, catch radius 48 px, cooldown 0.5 s.
- Chaser: constant top speed 175 px/s, turn rate clamped to 3.6 rad/s.
- Prey: 160 px/s (0.91× the chaser, so interception — not out-running — is what
  matters). Six dynamics:
  - `const-vel` — linear (first-order extrapolation is exact; control).
  - `circling` — constant turn rate → arcs/circles (**where prediction matters**).
  - `ou-turn`, `ou-vel` — Ornstein–Uhlenbeck (stochastic) turn and speed.
  - `jump` — piecewise-constant heading with random jumps.
  - `flee` — reactive avoidance (steers away, speeds up when approached).
  - `circling-noisy` — `circling` plus OU noise on the turn rate (noise sweep only).
- Protocol: **reset-on-catch** — each catch teleports the prey to a random far
  location, so every catch is a fresh interception from distance (no "lock-on"
  saturation). Metric: catches per episode (mean ± sd over 10 seeds).

## Agents

| agent | type | mechanism |
|---|---|---|
| velocity-lead | predictor | `p + v·τ` (first-order extrapolation) |
| accel-lead | predictor | `p + v·τ + ½a·τ²` (second-order Taylor) |
| kalman-lead | predictor | constant-velocity Kalman filter + lead |
| circle-fit | predictor | least-squares circle fit + extrapolation (analytic specialist) |
| **bdh** | predictor | Hebbian fast-weight world model + open-loop imagination |
| bdh-cl | predictor | the same world model + closed-loop imagination |
| wm-sgd | predictor | plain LMS world model (same features/rollout, no gating/decay) |
| wm-rls | predictor | recursive-least-squares world model (optimal online linear) |
| wm-mlp | predictor | one-hidden-layer MLP world model (Adam) |
| bdh-ng | predictor | natural-gradient (per-synapse) Hebbian world model + slow Polyak–Ruppert readout |
| bdh-pre | predictor | natural-gradient (per-synapse) Hebbian world model, fast readout only |
| bdh-avg | predictor | BDH (scalar NLMS) + slow Polyak–Ruppert-averaged readout |
| bdh-r | predictor | BDH + unit bearing features (reactive map representable) |
| bdh-rd | predictor | BDH + bearing features + halved lead (decorrelation-adapted) |
| velocity-lead-h | predictor | velocity-lead with halved lead horizon (ablation) |
| mpc-vel | predictor | receding-horizon search (48 aims) under a *perfect* constant-velocity model, scored by imagined time-to-catch |
| world-dreamer | predictor | the same search, but the prey is rolled forward by the *learned* BDH world model (Dreamer loop over fast weights) |
| pure-pursuit | policy | steer at current prey (reflex, no prediction) |
| MPC | policy | model-based search over 8 headings (first-order prey model) |
| linear-Q | policy | linear FA + TD(0) + ε-greedy (the evaluative reading) |
| DQN | policy | MLP Q-learning (replay + target net + Adam) |
| PPO | policy | clipped-surrogate PPO (policy + value MLPs) |
| SAC | policy | Soft Actor-Critic (continuous turn-rate action) |

The **bdh** world model is a linear content-addressable associative memory
(`W: [2×D]`, `ŷ = W·φ(s)`) over normalized features `[1, rel_x, rel_y, v_x, v_y]`,
updated by the normalized three-factor (error-gated) Hebbian rule
`W ← W + η·(y − ŷ)·φ(s)ᵀ/(1 + ‖φ‖²)`, and rolled forward τ steps (imagination) to
forecast the future prey position for lead pursuit. It is trained on every observed
transition (dense, self-supervised), and the policy (steer at the forecast) is driven
only by catch outcomes.

## Reproducing

```bash
python3 bench.py                 # full tables + significance tests → results.md (~1 hr)
NOISE_SWEEP=1 python3 bench.py   # also run the circling-noisy noise sweep
python3 ablations.py             # reproduce the §5.4 ablation numbers (~20 min)
node bench.js <horizon> <prey...>  # reference JavaScript implementation
```

Every number in `paper.md` / `paper.tex` is produced by `bench.py` (numpy, canonical)
and cross-checked against `bench.js` (reference). Predictors: 10 seeds × 24000 steps;
policies: 10 seeds × 20000 steps; reset-on-catch; Welch t-tests.

## Key results

The headline (Table 1 of the paper, 10 seeds): every model-free value learner
(linear-Q, DQN, PPO, SAC) falls below the no-prediction reflex in every environment,
while the same associative memory as a world model stays within 14% of the best analytic
predictor per environment. On `circling`, the world model beats first-order by +18% and
second-order by +196% (both significant, p ≤ 0.0031); a hand-crafted circle-fitter is
the only predictor that exceeds it there, and only on the smooth-curve prey it is
designed for. See `results.md` and §5 of `paper.md` for the complete tables, the
significance tests, the noise sweep, the closed-loop ablation, and the formulation comparison.

A **world-model formulation comparison** (Table 4, §5.6) asks whether the *specific* Hebbian
rule matters. It does **not** make BDH the accuracy optimum: recursive least squares (RLS) — the
optimal online linear estimator — ties BDH on `const-vel` and beats it on every other stationary
prey (`circling` 416 vs 385, `jump` 349 vs 296), and plain LMS is statistically
indistinguishable from BDH on `circling` (p = 0.68). But BDH is the only formulation realizable
by local, three-factor synaptic plasticity, and it *decisively* beats RLS on reactive `flee`
prey (178 vs 86, p = 2.4e-9). The MLP is worst on every stationary prey, confirming that a
linear content-addressable memory is the right model class for these smooth dynamics.

A **natural-gradient refinement** (§5.7, Table 5) replaces BDH's scalar NLMS gain with a
per-synapse (metaplastic) gain `η/(ε+√gᵢ)` — AdaGrad's diagonal preconditioner — and optionally a
slow Polyak–Ruppert-averaged readout. It recovers most of RLS's stationary edge while staying
O(D) local; the residual gap is exactly the off-diagonal co-activity term (the O(D²) price of
optimality), and on `circling` RLS is already at the analytic ceiling so that gap is unclosable.

A **nonstationary/adversarial suite** (§5.8, Table 6) adds two co-evolving evaders — `zigflee`
(a flee-er that periodically re-samples a random jink) and `adversarial` (an "evolve-as-you-evolve"
prey whose evasiveness rises each time it is caught and decays while it escapes). On every
reactive prey, RLS — the winner on stationary prey — is the *worst* world model, and BDH beats it
by nearly three-to-one (largest on `adversarial`, where the prey adapts to the pursuer's own success).

A **reactive refinement** (§5.9, Table 7) closes the last gap on the nonstationary suite. The
reactive evaders' map is a function of the *bearing* (`atan2` of the relative position), which the
linear feature set cannot represent; adding the unit bearing (`bdh-r`) closes the gap to
velocity-lead (178→197 on `flee`, parity). The remaining lever is the *lead horizon*: a reactive
evader re-aims away from the moving pursuer, so its velocity decorrelates faster than the geometric
lead `τ=d/vc` assumes; halving the lead (`bdh-rd`) beats velocity-lead on all three nonstationary
prey (p ≤ 5.8e-5). The honest decomposition: the halved horizon is the dominant lever
(`velocity-lead-h` reaches 204/239/168), while the bearing features are the Hebbian-specific fix
that closes the representational gap.

A **world dreamer** (§5.10, Table 8) completes the architecture: it replaces the analytic lead
with the Dreamer planning loop itself — receding-horizon optimization *inside* the learned world
model. On predictable prey it beats the analytic lead (up to $\sim\!40\%$ on `circling`); on
reactive prey it matches, but does not exceed, the §5.9 short-lead baseline, because the dreamer
inherits the world model's own ceiling: the velocity-decorrelation time. The 2×2 control
(`mpc-vel` vs `world-dreamer`, analytic vs search) isolates the gain — the search objective, not
the model, on straight prey; the *learned* model, not the naive one, on curved and reactive prey.

## Limitations

The result is strongest on curved, predictable motion — the case where predicting a
*trajectory*, not a point velocity, matters. On genuinely unpredictable (Martingale)
motion no predictor beats first-order; on reactive prey the world model reaches — but does not
exceed — a first-order rule (bearing features + decorrelation-adapted lead, §5.9); and a
hand-crafted circle-fitter remains the specialist ceiling on smooth curved motion. The world model is a linear associative
memory (the nonlinear Fourier expansion did not help on these smooth dynamics). The world
dreamer's *planner* is hand-configured (fixed search budget, horizon, and terminal cost), and on
reactive prey it inherits — rather than resolves — the decorrelation horizon (§5.10). See
§6.3–6.4 of `paper.md` for the full discussion and future work.

## Files

- `bench.py` — numpy-accelerated canonical benchmark: environment, predictors,
  policies (incl. DQN/PPO/SAC), significance tests, and the noise sweep.
- `bench.js` — reference JavaScript implementation (float-exact for the analytic
  predictors).
- `ablations.py` — reproduces the §5.4 ablation table.
- `paper.md`, `paper.tex`, `paper.pdf` — the paper (Markdown / LaTeX / compiled PDF).
- `results.md` — generated result tables (written by `bench.py`).
- `results.js` — legacy JS table generator (superseded by `bench.py`).
