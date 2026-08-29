#!/usr/bin/env python3
# bench.py — numpy-accelerated Dragon-Hatchling pursuit benchmark.
#
# Float-exact port of bench.js for the environment, RNG, and analytic predictors
# (the deterministic predictor numbers are statistically identical to the JS
# reference; 1-ulp libm vs V8 transcendentals cause only noise-level drift).
# Model-free RL baselines (DQN, PPO, SAC) use numpy *minibatch* training, which
# is 20-50x faster than the per-sample JavaScript loop.
#
# Usage:
#   python3 bench.py                # full tables + significance (writes results.md)
#   python3 bench.py --quick        # smoke: 3 seeds, short horizon
#   NOISE_SWEEP=1 python3 bench.py  # also run the circling-noisy noise sweep

import math
import os
import sys

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------------- constants / env knobs ----------------
W, H = 1200, 800
DT = 0.05
CATCH_RADIUS = 48
COOLDOWN_STEPS = 10
CHASE_MAX = 175
CHASER_MAXTURN = float(os.environ.get('TURN', 3.6))
PREY_MAXTURN = 2.5
PREY_VMAX = 160
PREY_SPEED = float(os.environ.get('PREY_SPEED', 160))
LEAD_TAU_CAP = 40
PREY_NOISE = float(os.environ.get('PREY_NOISE', 0.3))

# ---------------- RNG: mulberry32 + Box-Muller (float-exact vs bench.js) ----------------
MASK32 = 0xFFFFFFFF

def _mulberry32(seed):
    a = seed & MASK32
    def u():
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK32
        t = ((a ^ (a >> 15)) * (1 | a)) & MASK32
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & MASK32)) & MASK32) ^ t) & MASK32
        return (t ^ (t >> 14)) / 4294967296.0
    return u

def make_rng(seed):
    u = _mulberry32(int(seed * 2654435761 + 1013904223) & MASK32)
    state = {'has_spare': False, 'spare': 0.0}
    def gauss():
        if state['has_spare']:
            state['has_spare'] = False
            return state['spare']
        while True:
            x = 2.0 * u() - 1.0
            y = 2.0 * u() - 1.0
            r = x * x + y * y
            if not (r >= 1.0 or r == 0.0):
                f = math.sqrt(-2.0 * math.log(r) / r)
                state['has_spare'] = True
                state['spare'] = y * f
                return x * f
    return {'next': u, 'gauss': gauss}

# ---------------- geometry ----------------
def wrap_delta(a, b, L):
    d = math.fmod(a - b, L)
    if d > L / 2:
        d -= L
    if d < -L / 2:
        d += L
    return d

def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def dist(px, py, cx, cy):
    dx = wrap_delta(px, cx, W)
    dy = wrap_delta(py, cy, H)
    return math.hypot(dx, dy)

def turn_toward(theta, target, max_turn):
    d = wrap_angle(target - theta)
    step = max(-max_turn * DT, min(max_turn * DT, d))
    return wrap_angle(theta + step)

def clamp(v, a, b):
    return a if v < a else (b if v > b else v)

# ---------------- prey ----------------
def make_prey(type_, rng):
    heading = rng['next']() * 2 * math.pi
    px = W / 2 + (rng['next']() - 0.5) * W * 0.6
    py = H / 2 + (rng['next']() - 0.5) * H * 0.6
    h = heading
    sp = PREY_SPEED
    omega = 0.0
    noise_state = 0.0
    jump_t = 30
    noise = PREY_NOISE
    e = 0.0          # adversarial: persistent evasiveness (rises on catch, decays on escape)
    jink = 0         # zigflee: steps until the next random heading jink
    bias = 0.0       # zigflee: current random heading offset

    P = {'px': px, 'py': py, 'vx': math.cos(h) * sp, 'vy': math.sin(h) * sp,
         'speed': sp, 'heading': h}

    if type_ in ('circling', 'circling-noisy'):
        omega = (rng['next']() - 0.5) * 2 * 1.2

    def set_vel(hh, ss):
        P['heading'] = wrap_angle(hh)
        P['speed'] = clamp(ss, 10, PREY_VMAX)
        P['vx'] = math.cos(P['heading']) * P['speed']
        P['vy'] = math.sin(P['heading']) * P['speed']

    def move():
        P['px'] = (P['px'] + P['vx'] * DT) % W
        P['py'] = (P['py'] + P['vy'] * DT) % H

    def step(dt, chaser):
        nonlocal h, sp, omega, noise_state, jump_t, e, jink, bias
        t = type_
        if t == 'const-vel':
            move()
        elif t == 'ou-turn':
            omega += (-1.0 * omega) * dt + 0.8 * math.sqrt(dt) * rng['gauss']()
            h = wrap_angle(h + omega * dt)
            set_vel(h, PREY_SPEED)
            move()
        elif t == 'circling':
            h = wrap_angle(h + omega * dt)
            set_vel(h, PREY_SPEED)
            move()
        elif t == 'circling-noisy':
            noise_state += (-1.0 * noise_state) * dt + noise * math.sqrt(dt) * rng['gauss']()
            h = wrap_angle(h + (omega + noise_state) * dt)
            set_vel(h, PREY_SPEED)
            move()
        elif t == 'ou-vel':
            omega += (-1.0 * omega) * dt + 0.6 * math.sqrt(dt) * rng['gauss']()
            sp += 1.5 * (PREY_SPEED - sp) * dt + 30 * math.sqrt(dt) * rng['gauss']()
            h = wrap_angle(h + omega * dt)
            set_vel(h, sp)
            move()
        elif t == 'jump':
            jump_t -= 1
            if jump_t <= 0:
                h = rng['next']() * 2 * math.pi
                jump_t = 15 + math.floor(rng['next']() * 45)
            set_vel(h, PREY_SPEED)
            move()
        elif t == 'flee':
            dx = wrap_delta(P['px'], chaser['cx'], W)
            dy = wrap_delta(P['py'], chaser['cy'], H)
            d = math.hypot(dx, dy)
            away = math.atan2(dy, dx)
            noise_v = 0.5 * rng['gauss']()
            target_h = away + noise_v
            h = turn_toward(h, target_h, PREY_MAXTURN)
            flee = clamp(PREY_SPEED * 0.42 + PREY_SPEED * 0.6 * max(0.0, 1 - d / 300), 30, PREY_SPEED * 1.03)
            set_vel(h, flee)
            move()
        elif t == 'zigflee':
            # Reactive evader with nonstationary jinks: flee directly away from the chaser, but
            # periodically re-sample a random heading offset, so the reactive state->velocity map
            # changes throughout the episode.
            jink -= 1
            if jink <= 0:
                bias = (rng['next']() - 0.5) * 2.0
                jink = 40 + math.floor(rng['next']() * 80)
            dx = wrap_delta(P['px'], chaser['cx'], W)
            dy = wrap_delta(P['py'], chaser['cy'], H)
            d = math.hypot(dx, dy)
            away = math.atan2(dy, dx)
            h = turn_toward(h, away + bias + 0.3 * rng['gauss'](), PREY_MAXTURN)
            flee = clamp(PREY_SPEED * 0.42 + PREY_SPEED * 0.6 * max(0.0, 1 - d / 300), 30, PREY_SPEED * 1.03)
            set_vel(h, flee)
            move()
        elif t == 'adversarial':
            # Co-adapting "evolve-as-you-evolve" prey: flee from the chaser, but a persistent
            # evasiveness state rises each time the prey is caught and decays while it escapes,
            # so the reactive map's intensity co-evolves with the chaser's own success.
            if chaser.get('_caught'):
                e = min(1.0, e + 0.2)
                chaser['_caught'] = False
            e *= (1.0 - dt / 8.0)                # 8 s escape decay
            dx = wrap_delta(P['px'], chaser['cx'], W)
            dy = wrap_delta(P['py'], chaser['cy'], H)
            d = math.hypot(dx, dy)
            away = math.atan2(dy, dx)
            h = turn_toward(h, away + 0.4 * rng['gauss'](), PREY_MAXTURN)
            flee = clamp(PREY_SPEED * 0.42 + PREY_SPEED * 0.6 * max(0.0, 1 - d / 300), 30, PREY_SPEED * 1.03)
            set_vel(h, flee * (1.0 + 0.4 * e))
            move()
        else:
            raise ValueError('unknown prey ' + t)

    return {'P': P, 'step': step}

# ---------------- chaser ----------------
def make_chaser():
    return {'cx': W * 0.2, 'cy': H * 0.2, 'heading': 0.0, 'speed': CHASE_MAX}

# ---------------- predictors ----------------
def lead_steps(p, chaser):
    d = dist(p['px'], p['py'], chaser['cx'], chaser['cy'])
    s = chaser['speed'] if chaser['speed'] > 0 else CHASE_MAX
    return clamp(round((d / s) / DT), 1, LEAD_TAU_CAP)

def _kalman_step(x, v, Pxx, Pxv, Pvv, z, q, r):
    x = x + v * DT
    Pxx = Pxx + 2 * DT * Pxv + DT * DT * Pvv + q
    Pxv = Pxv + DT * Pvv
    Pvv = Pvv + q
    S = Pxx + r
    Kx = Pxx / S
    Kv = Pxv / S
    y = z - x
    x = x + Kx * y
    v = v + Kv * y
    Pxx = (1 - Kx) * Pxx
    Pxv = (1 - Kx) * Pxv
    Pvv = Pvv - Kv * Pxv
    return x, v, Pxx, Pxv, Pvv

def _make_velocity_lead():
    return {'observe': lambda prev, nxt, ch: None,
            'predict_at': lambda p, ch, steps: {'x': p['px'] + p['vx'] * steps * DT,
                                                 'y': p['py'] + p['vy'] * steps * DT}}

def _make_accel_lead():
    hist = []
    def observe(prev, nxt, ch):
        hist.append((nxt['vx'], nxt['vy']))
        if len(hist) > 6:
            hist.pop(0)
    def predict_at(p, ch, steps):
        tau = steps * DT
        ax = ay = 0.0
        if len(hist) >= 2:
            a = hist[-1]; b = hist[0]
            dt_span = max(DT * (len(hist) - 1), DT)
            ax = (a[0] - b[0]) / dt_span
            ay = (a[1] - b[1]) / dt_span
        return {'x': p['px'] + p['vx'] * tau + 0.5 * ax * tau * tau,
                'y': p['py'] + p['vy'] * tau + 0.5 * ay * tau * tau}
    return {'observe': observe, 'predict_at': predict_at}

def _make_kalman_lead():
    st = {'x': 0.0, 'vx': 0.0, 'Pxx': 1.0, 'Pxv': 0.0, 'Pvv': 1.0,
          'y': 0.0, 'vy': 0.0, 'Pyy': 1.0, 'Pyv': 0.0, 'Pvv2': 1.0}
    q, r = 5.0, 400.0
    def observe(prev, nxt, ch):
        st['x'], st['vx'], st['Pxx'], st['Pxv'], st['Pvv'] = _kalman_step(
            st['x'], st['vx'], st['Pxx'], st['Pxv'], st['Pvv'], nxt['px'], q, r)
        st['y'], st['vy'], st['Pyy'], st['Pyv'], st['Pvv2'] = _kalman_step(
            st['y'], st['vy'], st['Pyy'], st['Pyv'], st['Pvv2'], nxt['py'], q, r)
    def predict_at(p, ch, steps):
        tau = steps * DT
        return {'x': st['x'] + st['vx'] * tau, 'y': st['y'] + st['vy'] * tau}
    return {'observe': observe, 'predict_at': predict_at}

def _make_circle_fit():
    hist = []
    def observe(prev, nxt, ch):
        if hist:
            lx, ly = hist[-1]
            hist.append((lx + wrap_delta(nxt['px'], prev['px'], W),
                         ly + wrap_delta(nxt['py'], prev['py'], H)))
        else:
            hist.append((prev['px'], prev['py']))
            hist.append((prev['px'] + wrap_delta(nxt['px'], prev['px'], W),
                         prev['py'] + wrap_delta(nxt['py'], prev['py'], H)))
        if len(hist) > 20:
            hist.pop(0)
    def fallback(p, steps):
        return {'x': p['px'] + p['vx'] * steps * DT, 'y': p['py'] + p['vy'] * steps * DT}
    def reset():
        hist.clear()

    def predict_at(p, ch, steps):
        if len(hist) < 5:
            return fallback(p, steps)
        # Kasa circle fit via normal equations (3x3 solve, pure Python — hot path)
        n = len(hist)
        Sxx = Sxy = Sx = Syy = Sy = b1 = b2 = b3 = 0.0
        for hx, hy in hist:
            r2 = hx * hx + hy * hy
            Sxx += hx * hx
            Sxy += hx * hy
            Sx += hx
            Syy += hy * hy
            Sy += hy
            b1 += hx * r2
            b2 += hy * r2
            b3 += r2
        # solve [[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,n]] x = [b1,b2,b3]
        det = (Sxx * (Syy * n - Sy * Sy)
               - Sxy * (Sxy * n - Sy * Sx)
               + Sx * (Sxy * Sy - Syy * Sx))
        if abs(det) < 1e-12:
            return fallback(p, steps)
        A = (b1 * (Syy * n - Sy * Sy) - Sxy * (b2 * n - b3 * Sy) + Sx * (b2 * Sy - b3 * Syy)) / det
        B = (Sxx * (b2 * n - b3 * Sy) - b1 * (Sxy * n - Sy * Sx) + Sx * (Sxy * b3 - b2 * Sx)) / det
        C = (Sxx * (Syy * b3 - b2 * Sy) - Sxy * (Sxy * b3 - b2 * Sx) + b1 * (Sxy * Sy - Syy * Sx)) / det
        a = A / 2.0
        b = B / 2.0
        r = math.sqrt(max(0.0, C + a * a + b * b))
        if r < 1.0 or r > 2000.0:
            return fallback(p, steps)
        angs = [math.atan2(hy - b, hx - a) for hx, hy in hist]
        deltas = [wrap_angle(angs[i + 1] - angs[i]) for i in range(n - 1)]
        omega = sum(deltas) / (len(deltas) * DT) if deltas else 0.0
        theta0 = math.atan2(hist[-1][1] - b, hist[-1][0] - a)
        theta = theta0 + omega * steps * DT
        return {'x': a + r * math.cos(theta), 'y': b + r * math.sin(theta)}
    return {'observe': observe, 'predict_at': predict_at, 'reset': reset}

def make_predictor(name, rng):
    lead_scale = 1.0
    if name == 'velocity-lead':
        pred = _make_velocity_lead()
    elif name == 'velocity-lead-h':
        pred = _make_velocity_lead(); lead_scale = 0.5
    elif name == 'accel-lead':
        pred = _make_accel_lead()
    elif name == 'kalman-lead':
        pred = _make_kalman_lead()
    elif name == 'circle-fit':
        pred = _make_circle_fit()
    elif name == 'bdh':
        pred = _make_bdh(rng, closed_loop=False)
    elif name == 'bdh-cl':
        pred = _make_bdh(rng, closed_loop=True)
    elif name == 'bdh-r':
        pred = _make_bdh(rng, closed_loop=False, reactive=True)
    elif name == 'bdh-rd':
        pred = _make_bdh(rng, closed_loop=False, reactive=True); lead_scale = 0.5
    elif name == 'wm-sgd':
        pred = _make_wm_sgd(rng)
    elif name == 'wm-rls':
        pred = _make_wm_rls(rng)
    elif name == 'wm-mlp':
        pred = _make_wm_mlp(rng)
    elif name == 'bdh-ng':
        pred = _make_bdh_ng(rng, pre=True, avg=True)
    elif name == 'bdh-avg':
        pred = _make_bdh_ng(rng, pre=False, avg=True)
    elif name == 'bdh-pre':
        pred = _make_bdh_ng(rng, pre=True, avg=False)
    elif name == 'world-dreamer':
        pred = _make_world_dreamer(rng)
    elif name == 'mpc-vel':
        pred = _make_mpc_vel(rng)
    else:
        raise ValueError('unknown predictor ' + name)
    if 'predict_lead' not in pred:
        pred['predict_lead'] = lambda p, ch: pred['predict_at'](p, ch, max(1, round(lead_steps(p, ch) * lead_scale)))
    return pred

# ---------------- learned forward world models ----------------
# Every world-model variant learns the same one-step mapping  phi(s_t) -> v_{t+1}
# (next prey velocity) from the same per-step observations, then rolls the prey
# forward tau steps under the learned dynamics. The variants differ ONLY in the
# update rule (the learning formulation); the feature space, the rollout, and the
# lead-pursuit planner that consumes the forecast are identical.

WM_R = 700.0

def _rollout_wm(fwd, p, chaser, steps, closed_loop=False):
    steps = clamp(steps, 1, LEAD_TAU_CAP)
    px, py = p['px'], p['py']
    vx, vy = p['vx'], p['vy']
    cx, cy = chaser['cx'], chaser['cy']
    ch_h = chaser['heading']
    for _ in range(steps):
        vx, vy = fwd(px, py, vx, vy, cx, cy)
        px = (px + vx * DT) % W
        py = (py + vy * DT) % H
        if closed_loop:
            target = math.atan2(wrap_delta(py, cy, H), wrap_delta(px, cx, W))
            ch_h = turn_toward(ch_h, target, CHASER_MAXTURN)
            cx = (cx + math.cos(ch_h) * CHASE_MAX * DT) % W
            cy = (cy + math.sin(ch_h) * CHASE_MAX * DT) % H
    return {'x': px, 'y': py}

def _wm_feature(px, py, vx, vy, cx, cy):
    # linear feature vector (bias + 4 raw features); equals BDH with M=0
    return [1.0,
            wrap_delta(px, cx, W) / WM_R,
            wrap_delta(py, cy, H) / WM_R,
            vx / PREY_VMAX,
            vy / PREY_VMAX]

def _wm_raw4(px, py, vx, vy, cx, cy):
    # raw 4-dim input for the MLP variant (the MLP supplies its own bias)
    return [wrap_delta(px, cx, W) / WM_R,
            wrap_delta(py, cy, H) / WM_R,
            vx / PREY_VMAX,
            vy / PREY_VMAX]


# --- BDH (Dragon Hatchling): error-gated Hebbian fast-weight memory ---
def _make_bdh(rng, closed_loop=False, reactive=False):
    R = 700.0
    VMAX = float(PREY_VMAX)
    M = int(os.environ.get('BDH_M', 0))
    eta = float(os.environ.get('BDH_ETA', 0.5))
    lam = float(os.environ.get('BDH_LAM', 1e-3))
    # reactive: append the distance and the unit bearing away from the chaser, so the
    # linear readout can represent the evader's "turn toward away" map (atan2 of the
    # relative position, which is nonlinear in the raw (dx, dy) alone).
    RAW = 7 if reactive else 4
    D = 1 + RAW + 2 * M
    freq = [(rng['gauss']() * 1.4, rng['gauss']() * 1.4, rng['gauss']() * 1.4, rng['gauss']() * 1.4)
            for _ in range(M)]
    WM = [[0.0] * D, [0.0] * D]  # plain lists: tiny hot path, avoid numpy overhead
    err_sq = 0.0
    err_n = 0

    def raw(p, chaser):
        dx = wrap_delta(p['px'], chaser['cx'], W)
        dy = wrap_delta(p['py'], chaser['cy'], H)
        if reactive:
            d = math.hypot(dx, dy) + 1e-3
            return [dx / R, dy / R, d / R, dx / d, dy / d, p['vx'] / VMAX, p['vy'] / VMAX]
        return [dx / R, dy / R, p['vx'] / VMAX, p['vy'] / VMAX]

    def phi(s):
        out = [0.0] * D
        out[0] = 1.0
        for i in range(RAW):
            out[1 + i] = s[i]
        for j in range(M):
            k = freq[j]
            dot = k[0] * s[0] + k[1] * s[1] + k[2] * s[2] + k[3] * s[3]
            out[1 + RAW + 2 * j] = math.cos(dot)
            out[2 + RAW + 2 * j] = math.sin(dot)
        return out

    def _dot(row, f):
        s = 0.0
        for i in range(D):
            s += row[i] * f[i]
        return s

    def observe(prevP, nextP, chaser):
        nonlocal err_sq, err_n
        s = raw(prevP, chaser)
        f = phi(s)
        yx = nextP['vx'] / VMAX
        yy = nextP['vy'] / VMAX
        dx = _dot(WM[0], f)
        dy = _dot(WM[1], f)
        ex = yx - dx
        ey = yy - dy
        err_sq += ex * ex + ey * ey
        err_n += 1
        nrm = 0.0
        for i in range(D):
            nrm += f[i] * f[i]
        lr = eta / (1.0 + nrm)
        wd = 1.0 - lr * lam
        for i in range(D):
            WM[0][i] = WM[0][i] * wd + lr * ex * f[i]
            WM[1][i] = WM[1][i] * wd + lr * ey * f[i]

    def fwd(px, py, vx, vy, cx, cy):
        dx = wrap_delta(px, cx, W)
        dy = wrap_delta(py, cy, H)
        if reactive:
            d = math.hypot(dx, dy) + 1e-3
            s = [dx / R, dy / R, d / R, dx / d, dy / d, vx / VMAX, vy / VMAX]
        else:
            s = [dx / R, dy / R, vx / VMAX, vy / VMAX]
        f = phi(s)
        return _dot(WM[0], f) * VMAX, _dot(WM[1], f) * VMAX

    def predict_at(p, chaser, steps):
        return _rollout_wm(fwd, p, chaser, steps, closed_loop)

    return {'observe': observe, 'predict_at': predict_at,
            'get_W': lambda: WM,
            'err': lambda: (math.sqrt(err_sq / err_n) if err_n else 0.0)}


# ---------------- imagination-based planning (world dreamer) ----------------
# Shared receding-horizon shooting loop: the "dreamer" half of the world dreamer.
# model(px, py, pvx, pvy, cx, cy) maps vectorized (n_aims,) state arrays to the
# imagined next prey velocity; the chaser's own unicycle kinematics are exact.
# Each candidate aim heading is rolled forward nsteps inside the model and scored
# by imagined time-to-catch (with a closing-rate terminal cost for aims that do not
# catch within the horizon), and the best aim is returned.  This replaces the
# analytic intercept formula with optimization inside a world model.
def _mpc_aim(model, p, chaser, n_aims, nsteps):
    cx0, cy0 = chaser['cx'], chaser['cy']
    h0 = chaser['heading']
    aims = np.linspace(0.0, 2.0 * math.pi, n_aims, endpoint=False)
    hx = np.full(n_aims, h0)
    cx = np.full(n_aims, cx0)
    cy = np.full(n_aims, cy0)
    px = np.full(n_aims, p['px'])
    py = np.full(n_aims, p['py'])
    pvx = np.full(n_aims, p['vx'])
    pvy = np.full(n_aims, p['vy'])
    caught = np.full(n_aims, nsteps + 1, dtype=np.int64)
    turn_step = CHASER_MAXTURN * DT
    for k in range(1, nsteps + 1):
        delta = (aims - hx + math.pi) % (2.0 * math.pi) - math.pi
        hx = hx + np.clip(delta, -turn_step, turn_step)
        cx = (cx + np.cos(hx) * CHASE_MAX * DT) % W
        cy = (cy + np.sin(hx) * CHASE_MAX * DT) % H
        pvx, pvy = model(px, py, pvx, pvy, cx, cy)
        px = (px + pvx * DT) % W
        py = (py + pvy * DT) % H
        ddx = (px - cx + W / 2.0) % W - W / 2.0
        ddy = (py - cy + H / 2.0) % H - H / 2.0
        d = np.sqrt(ddx * ddx + ddy * ddy)
        caught = np.where((d < CATCH_RADIUS) & (caught == nsteps + 1), k, caught)
    # Terminal cost for aims that did not catch within the horizon: estimate the
    # remaining time-to-catch from the horizon state using the instantaneous
    # closing speed along the line of sight (constant-closing-rate extrapolation),
    # instead of a fixed upper-bound speed.  This keeps the imagined objective
    # close to the true time-to-catch even when the real catch lies past the horizon.
    ch_vx = np.cos(hx) * CHASE_MAX
    ch_vy = np.sin(hx) * CHASE_MAX
    rx = (px - cx + W / 2.0) % W - W / 2.0
    ry = (py - cy + H / 2.0) % H - H / 2.0
    dend = np.sqrt(rx * rx + ry * ry)
    closing = -(rx * (pvx - ch_vx) + ry * (pvy - ch_vy)) / np.maximum(dend, 1e-3)
    closing = np.maximum(closing, 1e-6)
    rem = dend / closing
    score = np.where(caught <= nsteps, caught.astype(np.float64), nsteps + rem)
    return aims[int(np.argmin(score))]


def _make_world_dreamer(rng):
    # Fast-weight world model (BDH) + imagination-based optimization: the world
    # dreamer.  The BDH weights are trained online exactly as in `bdh`, but instead
    # of an analytic lead, the chaser shoots candidate aim headings forward through
    # the learned model and executes the aim that maximizes imagined success.
    base = _make_bdh(rng, closed_loop=False)
    N = int(os.environ.get('WD_N', 48))
    HMAX = int(os.environ.get('WD_HMAX', 40))
    VMAX = float(PREY_VMAX)
    R = 700.0

    def predict_lead(p, chaser):
        Wm = np.asarray(base['get_W'](), dtype=np.float64)  # (2, D), D=5 for plain BDH (M=0)

        def model(px, py, pvx, pvy, cx, cy):
            dx = (px - cx + W / 2.0) % W - W / 2.0
            dy = (py - cy + H / 2.0) % H - H / 2.0
            f = np.stack([np.ones_like(px), dx / R, dy / R, pvx / VMAX, pvy / VMAX], axis=1)
            o = f @ Wm.T
            return o[:, 0] * VMAX, o[:, 1] * VMAX

        nsteps = clamp(lead_steps(p, chaser), 1, HMAX)
        theta = _mpc_aim(model, p, chaser, N, nsteps)
        return {'x': chaser['cx'] + math.cos(theta) * 100.0,
                'y': chaser['cy'] + math.sin(theta) * 100.0}

    return {'observe': base['observe'], 'predict_at': base['predict_at'],
            'predict_lead': predict_lead, 'err': base['err']}


def _make_mpc_vel(rng):
    # Control ablation: the same imagination/search loop, but with a *perfect*
    # straight-line prey model (prey keeps its current velocity).  This isolates the
    # planner from the world model — it should reproduce velocity-lead on straight
    # prey, so any dreamer gains trace to the learned model, not the search.
    N = int(os.environ.get('WD_N', 48))
    HMAX = int(os.environ.get('WD_HMAX', 40))

    def model(px, py, pvx, pvy, cx, cy):
        return pvx, pvy

    def predict_lead(p, chaser):
        nsteps = clamp(lead_steps(p, chaser), 1, HMAX)
        theta = _mpc_aim(model, p, chaser, N, nsteps)
        return {'x': chaser['cx'] + math.cos(theta) * 100.0,
                'y': chaser['cy'] + math.sin(theta) * 100.0}

    vel = _make_velocity_lead()
    return {'observe': lambda prev, nxt, ch: None, 'predict_at': vel['predict_at'],
            'predict_lead': predict_lead, 'err': lambda: 0.0}


# --- SGD (LMS): plain online least-mean-squares, no error gating or decay ---
def _make_wm_sgd(rng):
    VMAX = float(PREY_VMAX)
    D = 5
    lr = float(os.environ.get('WM_SGD_LR', 0.1))
    Wm = np.zeros((2, D), dtype=np.float64)
    err_sq = 0.0
    err_n = 0

    def observe(prevP, nextP, chaser):
        nonlocal err_sq, err_n, Wm
        f = np.asarray(_wm_feature(prevP['px'], prevP['py'], prevP['vx'], prevP['vy'],
                                   chaser['cx'], chaser['cy']), dtype=np.float64)
        y = np.array([nextP['vx'] / VMAX, nextP['vy'] / VMAX], dtype=np.float64)
        e = y - Wm @ f
        err_sq += float(e[0] * e[0] + e[1] * e[1])
        err_n += 1
        Wm += lr * np.outer(e, f)

    def fwd(px, py, vx, vy, cx, cy):
        f = np.asarray(_wm_feature(px, py, vx, vy, cx, cy), dtype=np.float64)
        o = Wm @ f
        return float(o[0]) * VMAX, float(o[1]) * VMAX

    def predict_at(p, chaser, steps):
        return _rollout_wm(fwd, p, chaser, steps, False)

    return {'observe': observe, 'predict_at': predict_at,
            'err': lambda: (math.sqrt(err_sq / err_n) if err_n else 0.0)}


# --- RLS: recursive least squares with exponential forgetting (optimal online linear) ---
def _make_wm_rls(rng):
    VMAX = float(PREY_VMAX)
    D = 5
    lam = float(os.environ.get('WM_RLS_LAM', 0.999))
    delta = float(os.environ.get('WM_RLS_DELTA', 1.0))
    P = np.eye(D, dtype=np.float64) / delta
    Wm = np.zeros((2, D), dtype=np.float64)
    err_sq = 0.0
    err_n = 0

    def observe(prevP, nextP, chaser):
        nonlocal P, Wm, err_sq, err_n
        f = np.asarray(_wm_feature(prevP['px'], prevP['py'], prevP['vx'], prevP['vy'],
                                   chaser['cx'], chaser['cy']), dtype=np.float64)
        y = np.array([nextP['vx'] / VMAX, nextP['vy'] / VMAX], dtype=np.float64)
        yhat = Wm @ f
        e = y - yhat
        err_sq += float(e[0] * e[0] + e[1] * e[1])
        err_n += 1
        Pf = P @ f
        denom = lam + float(f @ Pf)
        k = Pf / denom
        Wm += np.outer(e, k)
        P = (P - np.outer(k, Pf)) / lam

    def fwd(px, py, vx, vy, cx, cy):
        f = np.asarray(_wm_feature(px, py, vx, vy, cx, cy), dtype=np.float64)
        o = Wm @ f
        return float(o[0]) * VMAX, float(o[1]) * VMAX

    def predict_at(p, chaser, steps):
        return _rollout_wm(fwd, p, chaser, steps, False)

    return {'observe': observe, 'predict_at': predict_at,
            'err': lambda: (math.sqrt(err_sq / err_n) if err_n else 0.0)}


# --- MLP: one-hidden-layer network trained online with Adam ---
def _make_wm_mlp(rng):
    VMAX = float(PREY_VMAX)
    hidden = int(os.environ.get('WM_MLP_HIDDEN', 16))
    lr = float(os.environ.get('WM_MLP_LR', 1e-3))
    net = MLP([4, hidden, 2], rng)
    err_sq = 0.0
    err_n = 0

    def _in4(px, py, vx, vy, cx, cy):
        return np.asarray(_wm_raw4(px, py, vx, vy, cx, cy), dtype=np.float64).reshape(1, 4)

    def observe(prevP, nextP, chaser):
        nonlocal err_sq, err_n
        x = _in4(prevP['px'], prevP['py'], prevP['vx'], prevP['vy'], chaser['cx'], chaser['cy'])
        acts, zs = net.forward(x)
        yhat = acts[-1][0]
        y = np.array([nextP['vx'] / VMAX, nextP['vy'] / VMAX], dtype=np.float64)
        e = y - yhat
        err_sq += float(e[0] * e[0] + e[1] * e[1])
        err_n += 1
        d_out = (yhat - y).reshape(1, 2)  # gradient of 1/2 MSE wrt output
        net.backward(acts, zs, d_out, lr)

    def fwd(px, py, vx, vy, cx, cy):
        x = _in4(px, py, vx, vy, cx, cy)
        acts, _ = net.forward(x)
        o = acts[-1][0]
        return float(o[0]) * VMAX, float(o[1]) * VMAX

    def predict_at(p, chaser, steps):
        return _rollout_wm(fwd, p, chaser, steps, False)

    return {'observe': observe, 'predict_at': predict_at,
            'err': lambda: (math.sqrt(err_sq / err_n) if err_n else 0.0)}


# --- BDH-NG (Result 4): a natural-gradient, two-timescale Hebbian world model ---
# Two purely local, biologically-motivated refinements of the BDH rule, each of which
# supplies part of what RLS gets from its O(D^2) covariance matrix:
#   (i)  per-synapse (diagonal natural-gradient) learning rates: replace the scalar NLMS
#        gain with a running per-feature second-moment estimate (metaplasticity), so each
#        synapse's step is scaled by the inverse of that feature's local curvature --- the
#        diagonal of the covariance RLS inverts;
#   (ii) a slow Polyak--Ruppert-averaged readout: a slowly-consolidated copy of the fast
#        weights is used for prediction, averaging out the fast weights' misadjustment and
#        converging to the least-squares solution (RLS's fixed point) without any matrix.
# The fast weight still updates on every sample, so single-sample tracking on nonstationary
# prey is preserved.  pre=False reproduces BDH's scalar NLMS gain; avg=False reads out the
# fast weight directly, giving the two one-ingredient ablations.
def _make_bdh_ng(rng, pre=True, avg=True):
    VMAX = float(PREY_VMAX)
    D = 5
    eta = float(os.environ.get('WM_NG_ETA', 0.2))
    lam = float(os.environ.get('WM_NG_LAM', 0.0))
    beta = float(os.environ.get('WM_NG_BETA', 0.05))
    kappa = float(os.environ.get('WM_NG_KAPPA', 0.05))
    eps = float(os.environ.get('WM_NG_EPS', 1e-3))
    uni = os.environ.get('WM_NG_UNI', '0') == '1'   # uniform (1/t) averaging instead of fixed kappa
    Wf = np.zeros((2, D), dtype=np.float64)   # fast weights (tracking memory)
    Ws = np.zeros((2, D), dtype=np.float64)   # slow weights (consolidated readout)
    g = np.ones(D, dtype=np.float64)          # per-synapse second moment
    err_sq = 0.0
    err_n = 0
    t = 0

    def observe(prevP, nextP, chaser):
        nonlocal err_sq, err_n, Wf, Ws, g, t
        t += 1
        f = np.asarray(_wm_feature(prevP['px'], prevP['py'], prevP['vx'], prevP['vy'],
                                   chaser['cx'], chaser['cy']), dtype=np.float64)
        y = np.array([nextP['vx'] / VMAX, nextP['vy'] / VMAX], dtype=np.float64)
        e_update = y - (Wf @ f)   # self-consistent error for the fast recursion (stable LMS)
        yhat = (Ws @ f) if avg else (Wf @ f)
        e_read = y - yhat         # error of the prediction actually used by the planner
        err_sq += float(e_read[0] * e_read[0] + e_read[1] * e_read[1])
        err_n += 1
        if pre:
            if beta > 0.0:
                g = (1.0 - beta) * g + beta * (f * f)   # leaky second moment (constant lr floor)
            else:
                g = g + (f * f)                          # AdaGrad accumulation (anneals to zero)
            lr = eta / (eps + np.sqrt(g))                # per-synapse natural-gradient gain, shape (D,)
            wd = 1.0 - lr * lam
            Wf = Wf * wd[None, :] + np.outer(e_update, lr * f)
        else:
            nrm = float(f @ f)
            lr = eta / (1.0 + nrm)             # BDH's scalar NLMS gain
            wd = 1.0 - lr * lam
            Wf = Wf * wd + lr * np.outer(e_update, f)
        if avg:
            kt = (1.0 / float(t)) if uni else kappa
            Ws = (1.0 - kt) * Ws + kt * Wf

    def fwd(px, py, vx, vy, cx, cy):
        f = np.asarray(_wm_feature(px, py, vx, vy, cx, cy), dtype=np.float64)
        o = (Ws if avg else Wf) @ f
        return float(o[0]) * VMAX, float(o[1]) * VMAX

    def predict_at(p, chaser, steps):
        return _rollout_wm(fwd, p, chaser, steps, False)

    return {'observe': observe, 'predict_at': predict_at,
            'err': lambda: (math.sqrt(err_sq / err_n) if err_n else 0.0)}

# ---------------- episode (lead-pursuit predictors) ----------------
PRED_HORIZONS = [20, 40]

def run_episode(prey_type, predictor_name, seed, horizon, reset_on_catch=True):
    rng = make_rng(seed)
    prey = make_prey(prey_type, rng)
    chaser = make_chaser()
    pred = make_predictor(predictor_name, rng)
    catches = 0
    cooldown = 0
    sum_dist = 0.0
    q = [{'h': h, 'buf': []} for h in PRED_HORIZONS]
    f_err_sum = 0.0
    f_err_n = 0

    for t in range(horizon):
        p = prey['P']
        prev_snap = {'px': p['px'], 'py': p['py'], 'vx': p['vx'], 'vy': p['vy']}

        for e in q:
            pr = pred['predict_at'](p, chaser, e['h'])
            e['buf'].append({'x': pr['x'], 'y': pr['y'], 'due': t + e['h']})
        for e in q:
            while e['buf'] and e['buf'][0]['due'] <= t:
                f_err_sum += dist(e['buf'][0]['x'], e['buf'][0]['y'], p['px'], p['py'])
                f_err_n += 1
                e['buf'].pop(0)

        lead = pred['predict_lead'](p, chaser)
        target_h = math.atan2(wrap_delta(lead['y'], chaser['cy'], H), wrap_delta(lead['x'], chaser['cx'], W))
        chaser['heading'] = turn_toward(chaser['heading'], target_h, CHASER_MAXTURN)
        chaser['speed'] = CHASE_MAX
        chaser['cx'] = (chaser['cx'] + math.cos(chaser['heading']) * chaser['speed'] * DT) % W
        chaser['cy'] = (chaser['cy'] + math.sin(chaser['heading']) * chaser['speed'] * DT) % H

        prey['step'](DT, chaser)
        pred['observe'](prev_snap, prey['P'], chaser)

        d2 = dist(p['px'], p['py'], chaser['cx'], chaser['cy'])
        sum_dist += d2
        if cooldown > 0:
            cooldown -= 1
        if d2 < CATCH_RADIUS and cooldown == 0:
            catches += 1
            cooldown = COOLDOWN_STEPS
            chaser['_caught'] = True
            if reset_on_catch:
                ang = rng['next']() * 2 * math.pi
                dd = 400 + rng['next']() * 200
                prey['P']['px'] = (chaser['cx'] + math.cos(ang) * dd) % W
                prey['P']['py'] = (chaser['cy'] + math.sin(ang) * dd) % H
                for e in q:
                    e['buf'].clear()
                if 'reset' in pred:
                    pred['reset']()

    return {'catches': catches, 'meanDist': sum_dist / horizon,
            'err': pred['err']() if 'err' in pred else None,
            'forecastErr': (f_err_sum / f_err_n if f_err_n else 0.0)}

# ---------------- numpy MLP (minibatch forward/backward) ----------------
class MLP:
    def __init__(self, sizes, rng):
        self.L = len(sizes) - 1
        self.W = []
        self.b = []
        for l in range(self.L):
            n_in, n_out = sizes[l], sizes[l + 1]
            scale = math.sqrt(2.0 / n_in)
            Wl = np.empty((n_out, n_in), dtype=np.float64)
            for j in range(n_out):
                for k in range(n_in):
                    Wl[j, k] = rng['gauss']() * scale
            self.W.append(Wl)
            self.b.append(np.zeros(n_out, dtype=np.float64))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        # x: (B, in)
        acts = [x]
        zs = []
        a = x
        for l in range(self.L):
            z = a @ self.W[l].T + self.b[l]
            zs.append(z)
            a = np.maximum(z, 0.0) if l < self.L - 1 else z
            acts.append(a)
        return acts, zs

    def backward(self, acts, zs, d_out, lr):
        # d_out: (B, out); minibatch Adam (mean gradient over the batch)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1c = 1.0 - b1 ** self.t
        b2c = 1.0 - b2 ** self.t
        B = d_out.shape[0]
        grads = d_out
        for l in range(self.L - 1, -1, -1):
            z = zs[l]
            a_prev = acts[l]
            dz = grads if l == self.L - 1 else ((z > 0.0) * grads)
            grads_prev = dz @ self.W[l]
            gW = (dz.T @ a_prev) / B
            gb = dz.mean(axis=0)
            self.mW[l] = b1 * self.mW[l] + (1 - b1) * gW
            self.vW[l] = b2 * self.vW[l] + (1 - b2) * (gW * gW)
            self.W[l] -= lr * (self.mW[l] / b1c) / (np.sqrt(self.vW[l] / b2c) + eps)
            self.mb[l] = b1 * self.mb[l] + (1 - b1) * gb
            self.vb[l] = b2 * self.vb[l] + (1 - b2) * (gb * gb)
            self.b[l] -= lr * (self.mb[l] / b1c) / (np.sqrt(self.vb[l] / b2c) + eps)
            grads = grads_prev

def _copy_mlp(src, dst):
    for l in range(src.L):
        dst.W[l] = src.W[l].copy()
        dst.b[l] = src.b[l].copy()

# ---------------- model-free policies ----------------
N_ACTIONS = int(os.environ.get('NACT', 8))
ACTIONS = [a * (2 * math.pi / N_ACTIONS) for a in range(N_ACTIONS)]

def state_of(p, chaser):
    dx = wrap_delta(p['px'], chaser['cx'], W)
    dy = wrap_delta(p['py'], chaser['cy'], H)
    d = math.hypot(dx, dy)
    return [dx / 600, dy / 600,
            p['vx'] / PREY_VMAX, p['vy'] / PREY_VMAX,
            math.cos(chaser['heading']), math.sin(chaser['heading']),
            d / 600]

def _softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))

def make_policy(name, rng):
    if name == 'linear-q':
        D = 8
        w = [np.zeros(D, dtype=np.float64) for _ in range(N_ACTIONS)]
        state = {'eps': 1.0, 'step_c': 0}
        def phi(s):
            return [1.0, s[0], s[1], s[2], s[3], s[4], s[5], s[6]]
        def q(s, a):
            f = phi(s)
            return sum(w[a][i] * f[i] for i in range(D))
        def act(s):
            if rng['next']() < state['eps']:
                return int(rng['next']() * N_ACTIONS)
            return max(range(N_ACTIONS), key=lambda a: q(s, a))
        def observe(prevS, a, r, nextS):
            state['step_c'] += 1
            state['eps'] = max(0.05, 1 - state['step_c'] / 8000)
            qmax = max(q(nextS, b) for b in range(N_ACTIONS))
            td = r + 0.95 * qmax - q(prevS, a)
            f = phi(prevS)
            for i in range(D):
                w[a][i] += 0.01 * td * f[i]
        return {'act': act, 'observe': observe, 'continuous': False}

    if name == 'dqn':
        net = MLP([7, 32, 32, N_ACTIONS], rng)
        target = MLP([7, 32, 32, N_ACTIONS], rng)
        replay = []
        state = {'eps': 1.0, 'step_c': 0}
        def qvals(s):
            return net.forward(s)[0][net.L]
        def act(s):
            if rng['next']() < state['eps']:
                return int(rng['next']() * N_ACTIONS)
            return int(np.argmax(qvals(np.asarray([s], dtype=np.float64))[0]))
        def observe(prevS, a, r, nextS):
            state['step_c'] += 1
            state['eps'] = max(0.05, 1 - state['step_c'] / 8000)
            replay.append((prevS, a, r, nextS))
            if len(replay) > 10000:
                replay.pop(0)
            if state['step_c'] % 8 == 0 and len(replay) >= 64:
                batch = [replay[int(rng['next']() * len(replay))] for _ in range(32)]
                S = np.asarray([t[0] for t in batch], dtype=np.float64)
                A = np.asarray([t[1] for t in batch], dtype=np.int64)
                R = np.asarray([t[2] for t in batch], dtype=np.float64)
                NS = np.asarray([t[3] for t in batch], dtype=np.float64)
                fwd = net.forward(S)
                Q = fwd[0][net.L]
                Qn = target.forward(NS)[0][target.L]
                y = R + 0.95 * Qn.max(axis=1)
                d_out = np.zeros_like(Q)
                d_out[np.arange(len(batch)), A] = 2.0 * (Q[np.arange(len(batch)), A] - y) / len(batch)
                net.backward(fwd[0], fwd[1], d_out, 3e-4)
            if state['step_c'] % 500 == 0:
                _copy_mlp(net, target)
        return {'act': act, 'observe': observe, 'continuous': False}

    if name == 'ppo':
        pi = MLP([7, 64, 64, N_ACTIONS], rng)
        vf = MLP([7, 64, 64, 1], rng)
        buf = []
        GAMMA, LAM, CLIP, LR, EPOCHS, BATCH = 0.99, 0.95, 0.2, 3e-4, 4, 256
        def logits(s):
            return pi.forward(s)[0][pi.L]
        def value(s):
            return vf.forward(s)[0][vf.L]
        def probs_of(lg):
            mx = lg.max(axis=1, keepdims=True)
            ex = np.exp(lg - mx)
            return ex / ex.sum(axis=1, keepdims=True)
        def act(s):
            pr = probs_of(logits(np.asarray([s], dtype=np.float64)))[0]
            r = rng['next']()
            c = 0.0
            for i in range(N_ACTIONS):
                c += pr[i]
                if r <= c:
                    return i
            return N_ACTIONS - 1
        def observe(prevS, a, r, nextS):
            pr = probs_of(logits(np.asarray([prevS], dtype=np.float64)))[0]
            buf.append({'s': prevS, 'a': a, 'r': r, 'logp': math.log(pr[a] + 1e-12),
                        'v': float(value(np.asarray([prevS], dtype=np.float64)).item())})
            if len(buf) >= BATCH:
                T = len(buf)
                adv = np.zeros(T, dtype=np.float64)
                ret = np.zeros(T, dtype=np.float64)
                v_next = float(value(np.asarray([nextS], dtype=np.float64)).item())
                gae = 0.0
                for t in range(T - 1, -1, -1):
                    vn = v_next if t == T - 1 else buf[t + 1]['v']
                    delta = buf[t]['r'] + GAMMA * vn - buf[t]['v']
                    gae = delta + GAMMA * LAM * gae
                    adv[t] = gae
                    ret[t] = gae + buf[t]['v']
                am = float(np.mean(adv))
                sd = math.sqrt(float(np.mean((adv - am) ** 2)) + 1e-8)
                S = np.asarray([b['s'] for b in buf], dtype=np.float64)
                A = np.asarray([b['a'] for b in buf], dtype=np.int64)
                logp_old = np.asarray([b['logp'] for b in buf], dtype=np.float64)
                for _ in range(EPOCHS):
                    fwd = pi.forward(S)
                    lg = fwd[0][pi.L]
                    p2 = probs_of(lg)
                    ratio = np.exp(np.log(p2[np.arange(T), A] + 1e-12) - logp_old)
                    clip_mask = (np.abs(ratio - 1.0) < CLIP).astype(np.float64)
                    onehot = np.zeros_like(p2)
                    onehot[np.arange(T), A] = 1.0
                    d_out = -adv[:, None] * (onehot - p2) * clip_mask[:, None] / T
                    pi.backward(fwd[0], fwd[1], d_out, LR)
                    fv = vf.forward(S)
                    d_out_v = 2.0 * (fv[0][vf.L] - ret[:, None]) / T
                    vf.backward(fv[0], fv[1], d_out_v, LR)
                buf.clear()
        return {'act': act, 'observe': observe, 'continuous': False}

    if name == 'sac':
        return _make_sac(rng)

    raise ValueError('unknown policy ' + name)

def _make_sac(rng):
    ACT_DIM = 1
    actor = MLP([7, 64, 64, 2 * ACT_DIM], rng)
    q1 = MLP([7 + ACT_DIM, 64, 64, 1], rng)
    q2 = MLP([7 + ACT_DIM, 64, 64, 1], rng)
    q1t = MLP([7 + ACT_DIM, 64, 64, 1], rng)
    q2t = MLP([7 + ACT_DIM, 64, 64, 1], rng)
    _copy_mlp(q1, q1t); _copy_mlp(q2, q2t)
    GAMMA, TAU, ALPHA, LR = 0.99, 0.005, 0.1, 3e-4
    replay = []
    step_c = [0]

    def _concat(s, a):
        return np.concatenate([s, a], axis=1)

    def _qvals(net, s, a):
        return net.forward(_concat(s, a))[0][net.L]

    def _sample(S):
        # S: (B,7) -> a:(B,1), logp:(B,1), m, ls, std, eps, u
        out = actor.forward(S)[0][actor.L]  # (B, 2)
        m = out[:, 0:1]
        ls = out[:, 1:2]
        std = np.exp(ls)
        eps = np.asarray([rng['gauss']() for _ in range(S.shape[0])], dtype=np.float64).reshape(-1, 1)
        u = m + std * eps
        a = np.tanh(u)
        logp = (-ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((u - m) / std) ** 2
                - (2 * math.log(2) - 2 * u - 2 * _softplus(-2 * u)))
        return a, logp, m, ls, std, eps, u

    def act(s):
        a, _, _, _, _, _, _ = _sample(np.asarray([s], dtype=np.float64))
        return float(a[0, 0])

    def observe(prevS, a, r, nextS):
        step_c[0] += 1
        replay.append((prevS, a, r, nextS))
        if len(replay) > 50000:
            replay.pop(0)
        if step_c[0] < 1000 or step_c[0] % 4 != 0 or len(replay) < 64:
            return
        # critic update (minibatch)
        batch = [replay[int(rng['next']() * len(replay))] for _ in range(64)]
        S = np.asarray([t[0] for t in batch], dtype=np.float64)
        A = np.asarray([t[1] for t in batch], dtype=np.float64).reshape(-1, 1)
        R = np.asarray([t[2] for t in batch], dtype=np.float64).reshape(-1, 1)
        NS = np.asarray([t[3] for t in batch], dtype=np.float64)
        an, logpn, _, _, _, _, _ = _sample(NS)
        qt1 = _qvals(q1t, NS, an)
        qt2 = _qvals(q2t, NS, an)
        y = R + GAMMA * (np.minimum(qt1, qt2) - ALPHA * logpn)
        f1 = q1.forward(_concat(S, A))
        f2 = q2.forward(_concat(S, A))
        q1.backward(f1[0], f1[1], 2.0 * (f1[0][q1.L] - y) / len(batch), LR)
        q2.backward(f2[0], f2[1], 2.0 * (f2[0][q2.L] - y) / len(batch), LR)
        # actor update (reparameterized, finite-difference dQ/da)
        a2, _, m, ls, std, eps, u = _sample(S)
        h = 1e-3
        ap = np.clip(a2 + h, -1.0, 1.0)
        am = np.clip(a2 - h, -1.0, 1.0)
        qp = np.minimum(_qvals(q1, S, ap), _qvals(q2, S, ap))
        qm = np.minimum(_qvals(q1, S, am), _qvals(q2, S, am))
        dq_da = (qp - qm) / (2 * h)
        d_loss_du = ALPHA * (-(u - m) / (std ** 2) + 2.0 * a2) - dq_da * (1.0 - a2 ** 2)
        d_out = np.concatenate([d_loss_du, d_loss_du * std * eps], axis=1)
        af = actor.forward(S)
        actor.backward(af[0], af[1], d_out / len(batch), LR)
        # soft target update
        for l in range(q1.L):
            q1t.W[l] = (1 - TAU) * q1t.W[l] + TAU * q1.W[l]
            q1t.b[l] = (1 - TAU) * q1t.b[l] + TAU * q1.b[l]
            q2t.W[l] = (1 - TAU) * q2t.W[l] + TAU * q2.W[l]
            q2t.b[l] = (1 - TAU) * q2t.b[l] + TAU * q2.b[l]

    return {'act': act, 'observe': observe, 'continuous': True}

# ---------------- policy episode ----------------
def run_policy_episode(prey_type, policy_name, seed, horizon, reset_on_catch=True):
    rng = make_rng(seed)
    prey = make_prey(prey_type, rng)
    chaser = make_chaser()
    policy = None if policy_name in ('pure-pursuit', 'mpc') else make_policy(policy_name, rng)
    catches = 0
    cooldown = 0
    sum_dist = 0.0

    for _ in range(horizon):
        p = prey['P']
        s = state_of(p, chaser)
        a = 0.0
        if policy_name == 'pure-pursuit':
            chaser['heading'] = turn_toward(chaser['heading'],
                math.atan2(wrap_delta(p['py'], chaser['cy'], H), wrap_delta(p['px'], chaser['cx'], W)),
                CHASER_MAXTURN)
        elif policy_name == 'mpc':
            d0 = dist(p['px'], p['py'], chaser['cx'], chaser['cy'])
            HZ = clamp(round((d0 / CHASE_MAX) / DT), 1, 40)
            best, bestD = 0, float('inf')
            for aa in range(N_ACTIONS):
                h = chaser['heading']
                cx, cy = chaser['cx'], chaser['cy']
                px, py = p['px'], p['py']
                for _ in range(HZ):
                    h = turn_toward(h, ACTIONS[aa], CHASER_MAXTURN)
                    cx = (cx + math.cos(h) * CHASE_MAX * DT) % W
                    cy = (cy + math.sin(h) * CHASE_MAX * DT) % H
                    px = (px + p['vx'] * DT) % W
                    py = (py + p['vy'] * DT) % H
                d = dist(px, py, cx, cy)
                if d < bestD:
                    bestD = d
                    best = aa
            a = best
            chaser['heading'] = turn_toward(chaser['heading'], ACTIONS[best], CHASER_MAXTURN)
        else:
            a = policy['act'](s)
            if policy['continuous']:
                chaser['heading'] = wrap_angle(chaser['heading'] + a * CHASER_MAXTURN * DT)
            else:
                chaser['heading'] = turn_toward(chaser['heading'], ACTIONS[a], CHASER_MAXTURN)

        chaser['cx'] = (chaser['cx'] + math.cos(chaser['heading']) * CHASE_MAX * DT) % W
        chaser['cy'] = (chaser['cy'] + math.sin(chaser['heading']) * CHASE_MAX * DT) % H
        prey['step'](DT, chaser)

        d2 = dist(p['px'], p['py'], chaser['cx'], chaser['cy'])
        sum_dist += d2
        reward = -d2 / 600
        if cooldown > 0:
            cooldown -= 1
        if d2 < CATCH_RADIUS and cooldown == 0:
            catches += 1
            cooldown = COOLDOWN_STEPS
            reward += 1
            chaser['_caught'] = True
            if reset_on_catch:
                ang = rng['next']() * 2 * math.pi
                dd = 400 + rng['next']() * 200
                prey['P']['px'] = (chaser['cx'] + math.cos(ang) * dd) % W
                prey['P']['py'] = (chaser['cy'] + math.sin(ang) * dd) % H
        if policy is not None:
            policy['observe'](s, a, reward, state_of(prey['P'], chaser))

    return {'catches': catches, 'meanDist': sum_dist / horizon}

# ---------------- stats ----------------
def _mean_sd(arr):
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    return mean, sd

def _welch(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if _HAVE_SCIPY and len(a) > 1 and len(b) > 1:
        t, p = _scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    return float('nan'), float('nan')

# ---------------- main ----------------
PREY = ['const-vel', 'circling', 'ou-turn', 'ou-vel', 'jump', 'flee',
        'zigflee', 'adversarial']
NONSTATIONARY_PREY = ['flee', 'zigflee', 'adversarial']
PREDICTORS = ['velocity-lead', 'accel-lead', 'kalman-lead', 'circle-fit', 'bdh', 'bdh-cl']
WM_VARIANTS = ['bdh', 'wm-sgd', 'wm-rls', 'wm-mlp']
WM_IMPROVED = ['bdh-ng', 'bdh-avg', 'bdh-pre']
WM_REACTIVE = ['bdh-r', 'bdh-rd']
POLICIES = ['pure-pursuit', 'mpc', 'linear-q', 'dqn', 'ppo', 'sac']

def main():
    quick = '--quick' in sys.argv
    do_sweep = os.environ.get('NOISE_SWEEP') == '1'
    p_seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    m_seeds = [1, 2, 3] if quick else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    horizon = 6000 if quick else 24000
    m_horizon = 6000 if quick else 20000

    out = []
    out.append('# Dragon-Hatchling pursuit benchmark — results (numpy)')
    out.append('')
    out.append(f'Environment: toroidal {W}x{H}, dt={DT}s, catch radius {CATCH_RADIUS}px, chaser {CHASE_MAX} px/s, prey {PREY_SPEED:.0f} px/s.')
    out.append('Protocol: reset-on-catch. Metric: catches per episode (mean ± sd over seeds).')
    out.append('')

    all_preds = PREDICTORS + [p for p in WM_VARIANTS if p not in PREDICTORS] + WM_IMPROVED + WM_REACTIVE + ['velocity-lead-h', 'mpc-vel', 'world-dreamer']
    pred_data = {}
    for pt in PREY:
        for pr in all_preds:
            pred_data[(pt, pr)] = [run_episode(pt, pr, s, horizon, True)['catches'] for s in p_seeds]

    out.append('## Lead-pursuit predictors (mean ± sd, reset-on-catch, 10 seeds x 24000)')
    out.append('')
    out.append('| prey | velocity-lead | accel-lead | kalman-lead | circle-fit | bdh | bdh-cl |')
    out.append('|---|---|---|---|---|---|---|')
    for pt in PREY:
        row = [f'| {pt}']
        for pr in PREDICTORS:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## World-model formulation (mean ± sd, reset-on-catch, 10 seeds x 24000)')
    out.append('')
    out.append('| prey | bdh (Dragon Hatchling) | sgd (LMS) | rls | mlp |')
    out.append('|---|---|---|---|---|')
    for pt in PREY:
        row = [f'| {pt}']
        for pr in WM_VARIANTS:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## Improved world model: BDH-NG vs RLS (mean ± sd, reset-on-catch, 10 seeds x 24000)')
    out.append('')
    out.append('| prey | bdh | rls | bdh-ng (natural-gradient + averaging) | bdh-avg (averaging only) | bdh-pre (preconditioning only) |')
    out.append('|---|---|---|---|---|---|')
    for pt in PREY:
        row = [f'| {pt}']
        for pr in ['bdh', 'wm-rls'] + WM_IMPROVED:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## Nonstationary and adversarial prey (Result 5, mean ± sd, 10 seeds x 24000)')
    out.append('')
    out.append('| prey | velocity-lead | circle-fit | bdh | bdh-cl | rls | bdh-ng |')
    out.append('|---|---|---|---|---|---|---|')
    for pt in NONSTATIONARY_PREY:
        row = [f'| {pt}']
        for pr in ['velocity-lead', 'circle-fit', 'bdh', 'bdh-cl', 'wm-rls', 'bdh-ng']:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## Reactive world model: bearing features and decorrelation-adapted lead (Result 6, mean ± sd, 10 seeds x 24000)')
    out.append('')
    out.append('| prey | velocity-lead | velocity-lead-h | bdh | bdh-r (bearing) | bdh-rd (bearing + short lead) | circle-fit |')
    out.append('|---|---|---|---|---|---|---|')
    for pt in NONSTATIONARY_PREY:
        row = [f'| {pt}']
        for pr in ['velocity-lead', 'velocity-lead-h', 'bdh', 'bdh-r', 'bdh-rd', 'circle-fit']:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## World dreamer: imagination-based optimization (Result 7, mean ± sd, 10 seeds x 24000)')
    out.append('')
    out.append('Planner x world-model 2x2: velocity-lead = analytic + perfect; mpc-vel = search + perfect;')
    out.append('bdh = analytic + learned; world-dreamer = search + learned. bdh-rd is the reactive-adapted short-lead baseline.')
    out.append('')
    out.append('| prey | velocity-lead | mpc-vel | bdh | bdh-rd | world-dreamer |')
    out.append('|---|---|---|---|---|---|')
    for pt in PREY:
        row = [f'| {pt}']
        for pr in ['velocity-lead', 'mpc-vel', 'bdh', 'bdh-rd', 'world-dreamer']:
            mean, sd = _mean_sd(pred_data[(pt, pr)])
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## Policies (mean ± sd, reset-on-catch, 10 seeds x 20000)')
    out.append('')
    out.append('| prey | pure-pursuit | mpc | linear-q | dqn | ppo | sac |')
    out.append('|---|---|---|---|---|---|---|')
    pol_data = {}
    for pt in PREY:
        row = [f'| {pt}']
        for pl in POLICIES:
            cats = [run_policy_episode(pt, pl, s, m_horizon, True)['catches'] for s in m_seeds]
            pol_data[(pt, pl)] = cats
            mean, sd = _mean_sd(cats)
            row.append(f' {mean:.0f} ± {sd:.0f}')
        out.append(' |'.join(row) + ' |')
    out.append('')

    out.append('## Significance (Welch t-test, two-sided)')
    out.append('')
    out.append('| comparison | prey | t | p |')
    out.append('|---|---|---|---|')
    sigs = [
        ('bdh', 'velocity-lead', 'circling', 'BDH vs velocity-lead', pred_data),
        ('bdh', 'accel-lead', 'circling', 'BDH vs accel-lead', pred_data),
        ('bdh-cl', 'bdh', 'flee', 'BDH-cl vs BDH', pred_data),
        ('circle-fit', 'bdh', 'circling', 'circle-fit vs BDH', pred_data),
        ('bdh', 'wm-sgd', 'circling', 'BDH vs SGD (LMS)', pred_data),
        ('bdh', 'wm-rls', 'circling', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-mlp', 'circling', 'BDH vs MLP', pred_data),
        ('bdh', 'wm-rls', 'flee', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-rls', 'jump', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-rls', 'ou-turn', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-rls', 'ou-vel', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-sgd', 'flee', 'BDH vs SGD (LMS)', pred_data),
        ('bdh-ng', 'wm-rls', 'circling', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'ou-turn', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'ou-vel', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'jump', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'const-vel', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'flee', 'BDH-NG vs RLS', pred_data),
        ('bdh-ng', 'bdh', 'circling', 'BDH-NG vs BDH', pred_data),
        ('bdh-ng', 'bdh', 'flee', 'BDH-NG vs BDH', pred_data),
        ('bdh-pre', 'wm-rls', 'circling', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'wm-rls', 'ou-turn', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'wm-rls', 'ou-vel', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'wm-rls', 'jump', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'wm-rls', 'const-vel', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'wm-rls', 'flee', 'BDH-pre vs RLS', pred_data),
        ('bdh-pre', 'bdh', 'circling', 'BDH-pre vs BDH', pred_data),
        ('bdh-pre', 'bdh', 'ou-turn', 'BDH-pre vs BDH', pred_data),
        ('bdh-pre', 'bdh', 'flee', 'BDH-pre vs BDH', pred_data),
        ('bdh', 'wm-rls', 'zigflee', 'BDH vs RLS', pred_data),
        ('bdh', 'wm-rls', 'adversarial', 'BDH vs RLS', pred_data),
        ('bdh-ng', 'wm-rls', 'adversarial', 'BDH-NG vs RLS', pred_data),
        ('bdh-r', 'bdh', 'flee', 'BDH-r vs BDH', pred_data),
        ('bdh-r', 'velocity-lead', 'flee', 'BDH-r vs velocity-lead', pred_data),
        ('bdh-rd', 'velocity-lead', 'flee', 'BDH-rd vs velocity-lead', pred_data),
        ('bdh-rd', 'velocity-lead', 'zigflee', 'BDH-rd vs velocity-lead', pred_data),
        ('bdh-rd', 'velocity-lead', 'adversarial', 'BDH-rd vs velocity-lead', pred_data),
        ('bdh-rd', 'bdh-r', 'flee', 'BDH-rd vs BDH-r', pred_data),
        ('bdh-rd', 'circle-fit', 'flee', 'BDH-rd vs circle-fit', pred_data),
        ('velocity-lead-h', 'velocity-lead', 'flee', 'velocity-lead-h vs velocity-lead', pred_data),
        ('sac', 'pure-pursuit', 'circling', 'SAC vs reflex', pol_data),
        ('dqn', 'pure-pursuit', 'circling', 'DQN vs reflex', pol_data),
        ('world-dreamer', 'velocity-lead', 'const-vel', 'world-dreamer vs velocity-lead', pred_data),
        ('world-dreamer', 'velocity-lead', 'circling', 'world-dreamer vs velocity-lead', pred_data),
        ('world-dreamer', 'velocity-lead', 'ou-turn', 'world-dreamer vs velocity-lead', pred_data),
        ('world-dreamer', 'velocity-lead', 'ou-vel', 'world-dreamer vs velocity-lead', pred_data),
        ('world-dreamer', 'bdh', 'const-vel', 'world-dreamer vs BDH', pred_data),
        ('world-dreamer', 'bdh', 'circling', 'world-dreamer vs BDH', pred_data),
        ('world-dreamer', 'mpc-vel', 'circling', 'world-dreamer vs mpc-vel (learned vs naive model)', pred_data),
        ('world-dreamer', 'mpc-vel', 'flee', 'world-dreamer vs mpc-vel (learned vs naive model)', pred_data),
        ('world-dreamer', 'mpc-vel', 'adversarial', 'world-dreamer vs mpc-vel (learned vs naive model)', pred_data),
        ('mpc-vel', 'velocity-lead', 'const-vel', 'mpc-vel vs velocity-lead', pred_data),
        ('world-dreamer', 'bdh-rd', 'zigflee', 'world-dreamer vs bdh-rd (reactive boundary)', pred_data),
        ('world-dreamer', 'bdh-rd', 'adversarial', 'world-dreamer vs bdh-rd (reactive boundary)', pred_data),
    ]
    for a, b, pt, label, d in sigs:
        t, p = _welch(d.get((pt, a), []), d.get((pt, b), []))
        out.append(f'| {label} | {pt} | {t:.3f} | {p:.4g} |')
    out.append('')

    if do_sweep:
        out.append('## Noise sweep (circling-noisy, BDH vs velocity-lead, 10 seeds x 24000)')
        out.append('')
        out.append('| PREY_NOISE | velocity-lead | bdh | bdh - vel |')
        out.append('|---|---|---|---|')
        global PREY_NOISE
        for noise in [0.0, 0.15, 0.3, 0.6, 1.2]:
            PREY_NOISE = noise
            vl = [run_episode('circling-noisy', 'velocity-lead', s, horizon, True)['catches'] for s in p_seeds]
            bd = [run_episode('circling-noisy', 'bdh', s, horizon, True)['catches'] for s in p_seeds]
            vm, vsd = _mean_sd(vl)
            bm, bsd = _mean_sd(bd)
            out.append(f'| {noise} | {vm:.1f} ± {vsd:.0f} | {bm:.1f} ± {bsd:.0f} | {bm - vm:+.1f} |')
        out.append('')

    text = '\n'.join(out) + '\n'
    with open('results.md', 'w') as f:
        f.write(text)
    print(text)

if __name__ == '__main__':
    main()
