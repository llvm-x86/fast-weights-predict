# Dragon-Hatchling pursuit benchmark — results (numpy)

Environment: toroidal 1200x800, dt=0.05s, catch radius 48px, chaser 175 px/s, prey 160 px/s (PREY_VMAX cap).
Protocol: reset-on-catch. Metric: catches per episode (mean ± sd over seeds).

## Lead-pursuit predictors (mean ± sd, reset-on-catch, 10 seeds x 24000)

| prey | velocity-lead | accel-lead | kalman-lead | circle-fit | bdh | bdh-cl |
|---|---|---|---|---|---|---|
| const-vel | 364 ± 57 | 363 ± 56 | 148 ± 50 | 232 ± 48 | 363 ± 53 | 358 ± 58 |
| circling | 327 ± 35 | 130 ± 192 | 35 ± 66 | 415 ± 15 | 383 ± 38 | 383 ± 46 |
| ou-turn | 368 ± 17 | 364 ± 10 | 116 ± 7 | 373 ± 11 | 327 ± 6 | 332 ± 11 |
| ou-vel | 388 ± 8 | 385 ± 15 | 136 ± 16 | 388 ± 18 | 362 ± 12 | 362 ± 14 |
| jump | 342 ± 10 | 293 ± 8 | 133 ± 10 | 282 ± 6 | 298 ± 11 | 274 ± 6 |
| flee | 182 ± 2 | 68 ± 7 | 6 ± 3 | 162 ± 2 | 165 ± 5 | 147 ± 14 |

## Policies (mean ± sd, reset-on-catch, 10 seeds x 20000)

| prey | pure-pursuit | mpc | linear-q | dqn | ppo | sac |
|---|---|---|---|---|---|---|
| const-vel | 57 ± 6 | 242 ± 34 | 21 ± 6 | 23 ± 13 | 29 ± 12 | 22 ± 20 |
| circling | 176 ± 67 | 256 ± 31 | 20 ± 6 | 20 ± 5 | 18 ± 5 | 29 ± 12 |
| ou-turn | 134 ± 10 | 269 ± 21 | 24 ± 5 | 18 ± 5 | 24 ± 5 | 24 ± 4 |
| ou-vel | 134 ± 7 | 301 ± 11 | 21 ± 6 | 20 ± 6 | 23 ± 6 | 25 ± 6 |
| jump | 252 ± 10 | 261 ± 8 | 20 ± 3 | 25 ± 4 | 22 ± 4 | 32 ± 10 |
| flee | 159 ± 2 | 139 ± 3 | 0 ± 0 | 1 ± 2 | 0 ± 0 | 4 ± 4 |

## Significance (Welch t-test, two-sided)

| comparison | prey | t | p |
|---|---|---|---|
| BDH vs velocity-lead | circling | 3.457 | 0.002828 |
| BDH vs accel-lead | circling | 4.082 | 0.002359 |
| BDH-cl vs BDH | flee | -3.814 | 0.002963 |
| circle-fit vs BDH | circling | 2.533 | 0.0263 |
| SAC vs reflex | circling | -6.791 | 5.906e-05 |
| DQN vs reflex | circling | -7.300 | 4.275e-05 |

## Noise sweep (circling-noisy, BDH vs velocity-lead, 10 seeds x 24000)

| PREY_NOISE | velocity-lead | bdh | bdh - vel |
|---|---|---|---|
| 0.0 | 323.0 ± 34 | 372.8 ± 47 | +49.8 |
| 0.15 | 320.2 ± 38 | 379.2 ± 39 | +59.0 |
| 0.3 | 322.8 ± 35 | 391.4 ± 36 | +68.6 |
| 0.6 | 320.6 ± 36 | 376.2 ± 31 | +55.6 |
| 1.2 | 327.5 ± 21 | 328.3 ± 19 | +0.8 |

