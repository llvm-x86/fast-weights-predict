#!/usr/bin/env python3
# pursuit.py — the continuous half of the world dreamer, wired into the shared
# framework (framework.py) by importing the pursuit benchmark.
#
#   WorldModel = the BDH fast-weight memory (phi(s) -> v_{t+1}, a linear Hebbian
#                map over continuous state), trained online by NLMS.
#   Dreamer    = receding-horizon shooting: search candidate aim headings, roll
#                each forward *inside* the BDH model, score by imagined
#                time-to-catch, execute the best.
#
# This is a thin adapter over ../pursuit/bench.py; the real numbers live there
# (Result 7 / Table 8 of the paper).  Run `python3 pursuit.py` for a short demo.

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pursuit'))

import numpy as np

import bench
from framework import WorldModel, Dreamer, WorldDreamer


class BDHWorldModel(WorldModel):
    """The Dragon Hatchling fast-weight memory as a WorldModel."""

    def __init__(self, rng):
        self.base = bench._make_bdh(rng, closed_loop=False)
        self.VMAX = float(bench.PREY_VMAX)
        self.R = 700.0

    def observe(self, state, next_state):
        # state / next_state are {'prey': P, 'chaser': C} snapshots
        self.base['observe'](state['prey'], next_state['prey'], state['chaser'])

    def predict(self, state):
        # one-step prey-velocity prediction under the learned map
        p, ch = state['prey'], state['chaser']
        Wm = np.asarray(self.base['get_W'](), dtype=np.float64)
        dx = bench.wrap_delta(p['px'], ch['cx'], bench.W)
        dy = bench.wrap_delta(p['py'], ch['cy'], bench.H)
        f = np.array([1.0, dx / self.R, dy / self.R,
                      p['vx'] / self.VMAX, p['vy'] / self.VMAX], dtype=np.float64)
        o = Wm @ f
        return (float(o[0]) * self.VMAX, float(o[1]) * self.VMAX)


class ShootingDreamer(Dreamer):
    """MPC-in-imagination: shoot aim headings, roll out inside the BDH model."""

    def __init__(self, n_aims=48, hmax=40):
        self.n_aims = n_aims
        self.hmax = hmax

    def plan(self, state, model):
        p, ch = state['prey'], state['chaser']
        Wm = np.asarray(model.base['get_W'](), dtype=np.float64)
        VMAX = model.VMAX
        R = model.R

        def fwd(px, py, pvx, pvy, cx, cy):
            dx = (px - cx + bench.W / 2.0) % bench.W - bench.W / 2.0
            dy = (py - cy + bench.H / 2.0) % bench.H - bench.H / 2.0
            f = np.stack([np.ones_like(px), dx / R, dy / R, pvx / VMAX, pvy / VMAX], axis=1)
            o = f @ Wm.T
            return o[:, 0] * VMAX, o[:, 1] * VMAX

        nsteps = bench.clamp(bench.lead_steps(p, ch), 1, self.hmax)
        theta = bench._mpc_aim(fwd, p, ch, self.n_aims, nsteps)
        return theta


def make_pursuit_world_dreamer(rng):
    return WorldDreamer(BDHWorldModel(rng), ShootingDreamer())


if __name__ == '__main__':
    # Short smoke: run a few hundred steps of the pursuit loop through the
    # framework objects (world model + dreamer), reporting catches.
    rng = bench.make_rng(1)
    prey = bench.make_prey('flee', rng)
    chaser = bench.make_chaser()
    wd = make_pursuit_world_dreamer(rng)
    catches = 0
    cooldown = 0
    for _ in range(2000):
        p = prey['P']
        state = {'prey': {'px': p['px'], 'py': p['py'], 'vx': p['vx'], 'vy': p['vy']},
                 'chaser': chaser}
        prev = {'prey': dict(state['prey']), 'chaser': dict(chaser)}
        theta = wd.dreamer.plan(state, wd.world_model)
        chaser['heading'] = bench.turn_toward(chaser['heading'], theta, bench.CHASER_MAXTURN)
        chaser['cx'] = (chaser['cx'] + np.cos(chaser['heading']) * bench.CHASE_MAX * bench.DT) % bench.W
        chaser['cy'] = (chaser['cy'] + np.sin(chaser['heading']) * bench.CHASE_MAX * bench.DT) % bench.H
        prey['step'](bench.DT, chaser)
        nxt = {'prey': {'px': p['px'], 'py': p['py'], 'vx': p['vx'], 'vy': p['vy']},
               'chaser': chaser}
        wd.world_model.observe(prev, nxt)
        d = bench.dist(p['px'], p['py'], chaser['cx'], chaser['cy'])
        if cooldown > 0:
            cooldown -= 1
        if d < bench.CATCH_RADIUS and cooldown == 0:
            catches += 1
            cooldown = bench.COOLDOWN_STEPS
    print(f'pursuit world dreamer (flee, 2000 steps): {catches} catches')
