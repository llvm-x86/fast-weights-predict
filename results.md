# Dragon-Hatchling pursuit benchmark — results

Environment: toroidal 1200×800, dt=0.05s, catch radius 48px, chaser max speed 175 px/s (constant), prey speed 165 px/s.
Protocol: reset-on-catch (each catch teleports prey to a random far location, so every catch is a fresh interception from distance).
Metric: catches per episode (mean ± sd over seeds).

## Lead-pursuit predictors (catches, mean ± sd, reset-on-catch)

| prey | velocity-lead | accel-lead | kalman-lead | bdh (world model) |
|---|---|---|---|---|
| const-vel | 364 ± 57 | 364 ± 57 | 148 ± 50 | 363 ± 53 |
| circling | 327 ± 35 | 131 ± 192 | 35 ± 66 | 384 ± 38 |
| ou-turn | 368 ± 17 | 364 ± 10 | 116 ± 7 | 330 ± 11 |
| ou-vel | 388 ± 8 | 385 ± 15 | 136 ± 16 | 364 ± 12 |
| jump | 342 ± 10 | 293 ± 8 | 133 ± 10 | 298 ± 11 |
| flee | 182 ± 2 | 69 ± 7 | 6 ± 3 | 165 ± 5 |

## Model-free policies (catches, mean ± sd, reset-on-catch)

| prey | pure-pursuit (reflex) | MPC (1st-order) | linear-Q (deadly triad) | DQN | PPO |
|---|---|---|---|---|---|
| const-vel | 60 ± 10 | 242 ± 33 | 19 ± 6 | 30 ± 28 | 18 ± 5 |
| circling | 179 ± 52 | 254 ± 42 | 22 ± 5 | 41 ± 5 | 23 ± 5 |
| ou-turn | 129 ± 10 | 286 ± 23 | 25 ± 6 | 29 ± 4 | 23 ± 1 |
| ou-vel | 137 ± 11 | 304 ± 10 | 27 ± 5 | 29 ± 3 | 21 ± 9 |
| jump | 254 ± 8 | 265 ± 11 | 20 ± 2 | 44 ± 9 | 23 ± 7 |
| flee | 160 ± 1 | 140 ± 3 | 0 ± 1 | 37 ± 9 | 8 ± 8 |
