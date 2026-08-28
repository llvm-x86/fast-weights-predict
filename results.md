# Dragon-Hatchling pursuit benchmark — results (numpy)

Environment: toroidal 1200x800, dt=0.05s, catch radius 48px, chaser 175 px/s, prey 160 px/s.
Protocol: reset-on-catch. Metric: catches per episode (mean ± sd over seeds).

## Lead-pursuit predictors (mean ± sd, reset-on-catch, 10 seeds x 24000)

| prey | velocity-lead | accel-lead | kalman-lead | circle-fit | bdh | bdh-cl |
|---|---|---|---|---|---|---|
| const-vel | 398 ± 51 | 398 ± 51 | 148 ± 52 | 284 ± 46 | 397 ± 49 | 396 ± 48 |
| circling | 327 ± 35 | 130 ± 192 | 35 ± 66 | 415 ± 15 | 385 ± 41 | 386 ± 38 |
| ou-turn | 368 ± 17 | 364 ± 10 | 116 ± 7 | 373 ± 11 | 331 ± 15 | 327 ± 17 |
| ou-vel | 398 ± 14 | 389 ± 10 | 140 ± 9 | 398 ± 15 | 372 ± 13 | 362 ± 17 |
| jump | 342 ± 10 | 293 ± 8 | 133 ± 10 | 282 ± 6 | 296 ± 13 | 275 ± 16 |
| flee | 195 ± 3 | 71 ± 10 | 9 ± 5 | 178 ± 2 | 178 ± 3 | 160 ± 7 |

## World-model formulation (mean ± sd, reset-on-catch, 10 seeds x 24000)

| prey | bdh (Dragon Hatchling) | sgd (LMS) | rls | mlp |
|---|---|---|---|---|
| const-vel | 397 ± 49 | 398 ± 51 | 398 ± 50 | 297 ± 128 |
| circling | 385 ± 41 | 374 ± 71 | 416 ± 24 | 174 ± 19 |
| ou-turn | 331 ± 15 | 336 ± 8 | 371 ± 13 | 222 ± 27 |
| ou-vel | 372 ± 13 | 373 ± 17 | 399 ± 13 | 258 ± 33 |
| jump | 296 ± 13 | 296 ± 13 | 349 ± 9 | 254 ± 13 |
| flee | 178 ± 3 | 162 ± 4 | 86 ± 14 | 161 ± 8 |

## Policies (mean ± sd, reset-on-catch, 10 seeds x 20000)

| prey | pure-pursuit | mpc | linear-q | dqn | ppo | sac |
|---|---|---|---|---|---|---|
| const-vel | 80 ± 6 | 304 ± 37 | 18 ± 8 | 19 ± 15 | 30 ± 10 | 23 ± 13 |
| circling | 176 ± 67 | 256 ± 31 | 18 ± 3 | 23 ± 8 | 22 ± 6 | 27 ± 13 |
| ou-turn | 134 ± 10 | 269 ± 21 | 22 ± 5 | 19 ± 5 | 21 ± 3 | 25 ± 9 |
| ou-vel | 144 ± 9 | 308 ± 14 | 18 ± 5 | 21 ± 6 | 25 ± 7 | 28 ± 5 |
| jump | 252 ± 10 | 261 ± 8 | 20 ± 5 | 25 ± 4 | 24 ± 5 | 28 ± 7 |
| flee | 170 ± 3 | 152 ± 2 | 0 ± 0 | 3 ± 3 | 0 ± 0 | 7 ± 5 |

## Significance (Welch t-test, two-sided)

| comparison | prey | t | p |
|---|---|---|---|
| BDH vs velocity-lead | circling | 3.427 | 0.00309 |
| BDH vs accel-lead | circling | 4.103 | 0.002226 |
| BDH-cl vs BDH | flee | -8.069 | 4.596e-06 |
| circle-fit vs BDH | circling | 2.207 | 0.04838 |
| BDH vs SGD (LMS) | circling | 0.427 | 0.6759 |
| BDH vs RLS | circling | -2.105 | 0.05332 |
| BDH vs MLP | circling | 14.919 | 2.174e-09 |
| BDH vs RLS | flee | 20.864 | 2.411e-09 |
| BDH vs RLS | jump | -10.593 | 1.018e-08 |
| BDH vs RLS | ou-turn | -6.175 | 8.86e-06 |
| BDH vs RLS | ou-vel | -4.761 | 0.0001564 |
| BDH vs SGD (LMS) | flee | 11.394 | 3.732e-09 |
| SAC vs reflex | circling | -6.895 | 5.055e-05 |
| DQN vs reflex | circling | -7.169 | 4.617e-05 |

## Noise sweep (circling-noisy, BDH vs velocity-lead, 10 seeds x 24000)

| PREY_NOISE | velocity-lead | bdh | bdh - vel |
|---|---|---|---|
| 0.0 | 323.0 ± 34 | 378.9 ± 48 | +55.9 |
| 0.15 | 320.2 ± 38 | 388.8 ± 39 | +68.6 |
| 0.3 | 322.8 ± 35 | 392.6 ± 28 | +69.8 |
| 0.6 | 320.6 ± 36 | 379.5 ± 28 | +58.9 |
| 1.2 | 327.5 ± 21 | 326.0 ± 25 | -1.5 |

