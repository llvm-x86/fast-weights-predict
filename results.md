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
| zigflee | 227 ± 5 | 72 ± 6 | 70 ± 5 | 216 ± 4 | 208 ± 5 | 206 ± 4 |
| adversarial | 161 ± 2 | 45 ± 5 | 4 ± 2 | 144 ± 1 | 146 ± 3 | 129 ± 9 |

## World-model formulation (mean ± sd, reset-on-catch, 10 seeds x 24000)

| prey | bdh (Dragon Hatchling) | sgd (LMS) | rls | mlp |
|---|---|---|---|---|
| const-vel | 397 ± 49 | 398 ± 51 | 398 ± 50 | 297 ± 128 |
| circling | 385 ± 41 | 374 ± 71 | 416 ± 24 | 174 ± 19 |
| ou-turn | 331 ± 15 | 336 ± 8 | 371 ± 13 | 222 ± 27 |
| ou-vel | 372 ± 13 | 373 ± 17 | 399 ± 13 | 258 ± 33 |
| jump | 296 ± 13 | 296 ± 13 | 349 ± 9 | 254 ± 13 |
| flee | 178 ± 3 | 162 ± 4 | 86 ± 14 | 161 ± 8 |
| zigflee | 208 ± 5 | 201 ± 7 | 138 ± 8 | 188 ± 7 |
| adversarial | 146 ± 3 | 133 ± 2 | 51 ± 9 | 136 ± 7 |

## Improved world model: BDH-NG vs RLS (mean ± sd, reset-on-catch, 10 seeds x 24000)

| prey | bdh | rls | bdh-ng (natural-gradient + averaging) | bdh-avg (averaging only) | bdh-pre (preconditioning only) |
|---|---|---|---|---|---|
| const-vel | 397 ± 49 | 398 ± 50 | 397 ± 51 | 398 ± 51 | 398 ± 51 |
| circling | 385 ± 41 | 416 ± 24 | 394 ± 47 | 363 ± 88 | 408 ± 34 |
| ou-turn | 331 ± 15 | 371 ± 13 | 307 ± 13 | 310 ± 13 | 352 ± 15 |
| ou-vel | 372 ± 13 | 399 ± 13 | 348 ± 9 | 341 ± 11 | 381 ± 17 |
| jump | 296 ± 13 | 349 ± 9 | 294 ± 10 | 312 ± 18 | 301 ± 6 |
| flee | 178 ± 3 | 86 ± 14 | 162 ± 2 | 138 ± 5 | 173 ± 5 |
| zigflee | 208 ± 5 | 138 ± 8 | 187 ± 5 | 177 ± 4 | 208 ± 3 |
| adversarial | 146 ± 3 | 51 ± 9 | 139 ± 3 | 121 ± 5 | 143 ± 2 |

## Nonstationary and adversarial prey (Result 5, mean ± sd, 10 seeds x 24000)

| prey | velocity-lead | circle-fit | bdh | bdh-cl | rls | bdh-ng |
|---|---|---|---|---|---|---|
| flee | 195 ± 3 | 178 ± 2 | 178 ± 3 | 160 ± 7 | 86 ± 14 | 162 ± 2 |
| zigflee | 227 ± 5 | 216 ± 4 | 208 ± 5 | 206 ± 4 | 138 ± 8 | 187 ± 5 |
| adversarial | 161 ± 2 | 144 ± 1 | 146 ± 3 | 129 ± 9 | 51 ± 9 | 139 ± 3 |

## Policies (mean ± sd, reset-on-catch, 10 seeds x 20000)

| prey | pure-pursuit | mpc | linear-q | dqn | ppo | sac |
|---|---|---|---|---|---|---|
| const-vel | 80 ± 6 | 304 ± 37 | 18 ± 8 | 19 ± 15 | 30 ± 10 | 23 ± 13 |
| circling | 176 ± 67 | 256 ± 31 | 18 ± 3 | 23 ± 8 | 22 ± 6 | 27 ± 13 |
| ou-turn | 134 ± 10 | 269 ± 21 | 22 ± 5 | 19 ± 5 | 21 ± 3 | 25 ± 9 |
| ou-vel | 144 ± 9 | 308 ± 14 | 18 ± 5 | 21 ± 6 | 25 ± 7 | 28 ± 5 |
| jump | 252 ± 10 | 261 ± 8 | 20 ± 5 | 25 ± 4 | 24 ± 5 | 28 ± 7 |
| flee | 170 ± 3 | 152 ± 2 | 0 ± 0 | 3 ± 3 | 0 ± 0 | 7 ± 5 |
| zigflee | 201 ± 4 | 176 ± 3 | 1 ± 1 | 4 ± 4 | 0 ± 0 | 6 ± 6 |
| adversarial | 141 ± 1 | 124 ± 2 | 0 ± 0 | 2 ± 3 | 0 ± 0 | 4 ± 4 |

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
| BDH-NG vs RLS | circling | -1.319 | 0.2094 |
| BDH-NG vs RLS | ou-turn | -10.838 | 2.556e-09 |
| BDH-NG vs RLS | ou-vel | -10.282 | 2.63e-08 |
| BDH-NG vs RLS | jump | -12.616 | 2.614e-10 |
| BDH-NG vs RLS | const-vel | -0.049 | 0.9617 |
| BDH-NG vs RLS | flee | 17.105 | 1.781e-08 |
| BDH-NG vs BDH | circling | 0.478 | 0.6382 |
| BDH-NG vs BDH | flee | -15.155 | 1.221e-11 |
| BDH-pre vs RLS | circling | -0.608 | 0.5517 |
| BDH-pre vs RLS | ou-turn | -3.027 | 0.007369 |
| BDH-pre vs RLS | ou-vel | -2.660 | 0.01659 |
| BDH-pre vs RLS | jump | -14.033 | 2.724e-10 |
| BDH-pre vs RLS | const-vel | 0.000 | 1 |
| BDH-pre vs RLS | flee | 18.506 | 8.79e-10 |
| BDH-pre vs BDH | circling | 1.366 | 0.1895 |
| BDH-pre vs BDH | ou-turn | 3.130 | 0.00578 |
| BDH-pre vs BDH | flee | -2.712 | 0.01628 |
| BDH vs RLS | zigflee | 23.686 | 6.373e-13 |
| BDH vs RLS | adversarial | 33.218 | 2.455e-12 |
| BDH-NG vs RLS | adversarial | 30.278 | 2.102e-12 |
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

