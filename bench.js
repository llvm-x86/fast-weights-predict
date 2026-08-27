'use strict';
// bench.js — Dragon-Hatchling pursuit benchmark
// Seeded, single prey vs single chaser, toroidal world.
// The planner is FIXED (lead pursuit: steer at the predicted future prey
// position at horizon tau = distance / chaser_speed). Only the PREDICTOR varies:
//   velocity-lead (1st order), accel-lead (2nd order), kalman-lead,
//   bdh (content-addressable fast-weight world model, 3-factor Hebbian).
// This isolates "does a learned world model beat analytic extrapolation?"

const W = 1200, H = 800;
const DT = 0.05;
const CATCH_RADIUS = 48;
const COOLDOWN_STEPS = 10;          // ~0.5s no re-catch after a catch
const CHASE_MAX = 175;              // chaser sprints at constant top speed
const CHASER_MAXTURN = process.env.TURN != null ? parseFloat(process.env.TURN) : 3.6;   // rad/s
const PREY_MAXTURN = 2.5;           // rad/s (chaser has the edge)
const PREY_VMAX = 160;
const PREY_SPEED = process.env.PREY_SPEED != null ? parseFloat(process.env.PREY_SPEED) : 165;
const LEAD_TAU_CAP = 40;            // max imagination steps

// ---------------- RNG (mulberry32 + Box-Muller) ----------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeRng(seed) {
  const u = mulberry32(seed * 2654435761 + 1013904223);
  let hasSpare = false, spare = 0;
  function gauss() {
    if (hasSpare) { hasSpare = false; return spare; }
    let x, y, r;
    do { x = 2 * u() - 1; y = 2 * u() - 1; r = x * x + y * y; } while (r >= 1 || r === 0);
    const f = Math.sqrt(-2 * Math.log(r) / r);
    hasSpare = true; spare = y * f; return x * f;
  }
  return { next: u, gauss };
}

// ---------------- geometry ----------------
function wrapDelta(a, b, L) { let d = (a - b) % L; if (d > L / 2) d -= L; if (d < -L / 2) d += L; return d; }
function wrapAngle(a) { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; }
function dist(px, py, cx, cy) {
  const dx = wrapDelta(px, cx, W), dy = wrapDelta(py, cy, H);
  return Math.hypot(dx, dy);
}
function turnToward(theta, target, maxTurn) {
  const d = wrapAngle(target - theta);
  const step = Math.max(-maxTurn * DT, Math.min(maxTurn * DT, d));
  return wrapAngle(theta + step);
}
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

// ---------------- prey ----------------
// Prey state: {px, py, vx, vy, speed, heading}
// step(dt, chaserPos) advances one tick; returns nothing (mutates state).
function makePrey(type, rng) {
  const heading = rng.next() * 2 * Math.PI;
  const speed0 = PREY_SPEED;
  let px = W / 2 + (rng.next() - 0.5) * W * 0.6;
  let py = H / 2 + (rng.next() - 0.5) * H * 0.6;
  let h = heading, sp = speed0, omega = 0;
  let jumpT = 30;

  const P = { px, py, vx: Math.cos(h) * sp, vy: Math.sin(h) * sp, speed: sp, heading: h };

  // persistent-curvature prey: constant turn rate drawn per episode
  if (type === 'circling') omega = (rng.next() - 0.5) * 2 * 1.2;

  function setVel(hh, ss) {
    P.heading = wrapAngle(hh); P.speed = clamp(ss, 10, PREY_VMAX);
    P.vx = Math.cos(P.heading) * P.speed; P.vy = Math.sin(P.heading) * P.speed;
  }
  function move() {
    P.px = ((P.px + P.vx * DT) % W + W) % W;
    P.py = ((P.py + P.vy * DT) % H + H) % H;
  }

  const step = {
    'const-vel'(dt, chaser) { move(); },

    // persistent turning: angular velocity is OU -> smooth curved paths
    'ou-turn'(dt, chaser) {
      omega += (-1.0 * omega) * dt + 0.8 * Math.sqrt(dt) * rng.gauss();
      h = wrapAngle(h + omega * dt);
      setVel(h, 165);
      move();
    },

    // constant turn rate -> runs in arcs/circles (first-order extrapolation fails)
    'circling'(dt, chaser) {
      h = wrapAngle(h + omega * dt);
      setVel(h, PREY_SPEED);
      move();
    },

    // OU on both speed and angular velocity (accelerating, curved)
    'ou-vel'(dt, chaser) {
      omega += (-1.0 * omega) * dt + 0.6 * Math.sqrt(dt) * rng.gauss();
      sp += 1.5 * (PREY_SPEED - sp) * dt + 30 * Math.sqrt(dt) * rng.gauss();
      h = wrapAngle(h + omega * dt);
      setVel(h, sp);
      move();
    },

    // velocity-jump: piecewise constant heading, random jumps
    'jump'(dt, chaser) {
      jumpT -= 1;
      if (jumpT <= 0) { h = rng.next() * 2 * Math.PI; jumpT = 15 + Math.floor(rng.next() * 45); }
      setVel(h, PREY_SPEED);
      move();
    },

    // reactive fleeing: steers away from the chaser, faster when chaser near
    'flee'(dt, chaser) {
      const dx = wrapDelta(P.px, chaser.cx, W), dy = wrapDelta(P.py, chaser.cy, H);
      const d = Math.hypot(dx, dy);
      const away = Math.atan2(dy, dx);           // heading directly away
      const noise = 0.5 * rng.gauss();
      const targetH = away + noise;
      h = turnToward(h, targetH, PREY_MAXTURN);
      const flee = clamp(PREY_SPEED * 0.42 + PREY_SPEED * 0.6 * Math.max(0, 1 - d / 300), 30, PREY_SPEED * 1.03);
      setVel(h, flee);
      move();
    },
  }[type];

  return { P, step };
}

// ---------------- chaser ----------------
function makeChaser() {
  return { cx: W * 0.2, cy: H * 0.2, heading: 0, speed: CHASE_MAX };
}

// ---------------- predictors ----------------
// Each predictor exposes:
//   observe(p, chaser)     — train on the observed tick (filters / world model)
//   predictAt(p, chaser, steps) -> {x,y}  predicted prey position `steps` ahead
//   predictLead(p, chaser) — predictAt at the pursuit lead horizon (for steering)
// The planner (lead pursuit) is shared: steer at predicted future position.

function makePredictor(name, rng) {
  if (name === 'velocity-lead') {
    return {
      observe() {},
      predictAt(p, chaser, steps) {
        const tau = steps * DT;
        return { x: p.px + p.vx * tau, y: p.py + p.vy * tau };
      },
      predictLead(p, chaser) { return this.predictAt(p, chaser, leadSteps(p, chaser)); },
    };
  }

  if (name === 'accel-lead') {
    const hist = []; // keep last few velocities (world coords)
    return {
      observe(prev, next) {
        hist.push({ x: next.vx, y: next.vy });
        if (hist.length > 6) hist.shift();
      },
      predictAt(p, chaser, steps) {
        const tau = steps * DT;
        let ax = 0, ay = 0;
        if (hist.length >= 2) {
          const a = hist[hist.length - 1], b = hist[0];
          const dtSpan = Math.max(DT * (hist.length - 1), DT);
          ax = (a.x - b.x) / dtSpan; ay = (a.y - b.y) / dtSpan;
        }
        return { x: p.px + p.vx * tau + 0.5 * ax * tau * tau, y: p.py + p.vy * tau + 0.5 * ay * tau * tau };
      },
      predictLead(p, chaser) { return this.predictAt(p, chaser, leadSteps(p, chaser)); },
    };
  }

  if (name === 'kalman-lead') {
    // 2D constant-velocity Kalman (uncorrelated axes)
    let x = 0, vx = 0, Pxx = 1, Pxv = 0, Pvv = 1;
    let y = 0, vy = 0, Pyy = 1, Pyv = 0, Pvv2 = 1;
    const q = 5, r = 400;
    return {
      observe(prev, next) {
        ({ x, v: vx, Pxx, Pxv, Pvv } = kalmanStep(x, vx, Pxx, Pxv, Pvv, next.px, q, r));
        ({ x: y, v: vy, Pxx: Pyy, Pxv: Pyv, Pvv: Pvv2 } = kalmanStep(y, vy, Pyy, Pyv, Pvv2, next.py, q, r));
      },
      predictAt(p, chaser, steps) {
        const tau = steps * DT;
        return { x: x + vx * tau, y: y + vy * tau };
      },
      predictLead(p, chaser) { return this.predictAt(p, chaser, leadSteps(p, chaser)); },
    };
  }

  if (name === 'bdh') return makeBdh(rng);

  throw new Error('unknown predictor ' + name);
}

function leadSteps(p, chaser) {
  const d = dist(p.px, p.py, chaser.cx, chaser.cy);
  const s = chaser.speed > 0 ? chaser.speed : CHASE_MAX;
  return clamp(Math.round((d / s) / DT), 1, LEAD_TAU_CAP);
}
function leadTau(p, chaser) { return leadSteps(p, chaser) * DT; }

function kalmanStep(x, v, Pxx, Pxv, Pvv, z, q, r) {
  // predict
  x = x + v * DT;
  Pxx = Pxx + 2 * DT * Pxv + DT * DT * Pvv + q;
  Pxv = Pxv + DT * Pvv;
  Pvv = Pvv + q;
  // update
  const S = Pxx + r;
  const Kx = Pxx / S, Kv = Pxv / S;
  const y = z - x;
  x = x + Kx * y; v = v + Kv * y;
  Pxx = (1 - Kx) * Pxx; Pxv = (1 - Kx) * Pxv; Pvv = Pvv - Kv * Pxv;
  return { x, v, Pxx, Pxv, Pvv };
}

// ---------------- BDH content-addressable world model ----------------
// Learns the prey's one-step transition f: state_t -> v_{t+1} (the world model),
// then rolls it forward (imagination) to forecast the future prey position.
// Raw state s (normalized ~[-1,1]) = [relx/R, rely/R, vx/VMAX, vy/VMAX]
//   (relative geometry + current velocity).
// Features phi(s) = [1, s.., cos(k·s), sin(k·s)]  (random Fourier features for
//   nonlinear content-addressable generalization; M=0 is a pure linear associator).
// Fast-weight memory WM: [2 x D], readout yhat = WM·phi(s)  (content-addressable recall)
// Three-factor (error-gated) Hebbian update, normalized (NLMS) for stability:
//   WM += eta·(y - yhat)·phi(s)^T / (1 + phi·phi), plus synaptic weight decay.
function makeBdh(rng) {
  const R = 700, VMAX = PREY_VMAX;
  const M = process.env.BDH_M != null ? parseInt(process.env.BDH_M, 10) : 0;    // Fourier frequencies (0 = pure linear associator)
  const eta = process.env.BDH_ETA != null ? parseFloat(process.env.BDH_ETA) : 0.5;
  const lam = process.env.BDH_LAM != null ? parseFloat(process.env.BDH_LAM) : 1e-3;  // synaptic weight decay
  const RAW = 4;
  const D = 1 + RAW + 2 * M;
  const TAU_CAP_STEPS = LEAD_TAU_CAP;

  // random Fourier frequencies (deterministic given rng)
  const freq = [];
  for (let i = 0; i < M; i++) {
    freq.push([rng.gauss() * 1.4, rng.gauss() * 1.4, rng.gauss() * 1.4, rng.gauss() * 1.4]);
  }
  let WM = [new Float64Array(D), new Float64Array(D)];  // fast-weight matrix rows = output dims (dx, dy)
  let errSq = 0, errN = 0;

  function raw(p, chaser) {
    return [
      wrapDelta(p.px, chaser.cx, W) / R,
      wrapDelta(p.py, chaser.cy, H) / R,
      p.vx / VMAX, p.vy / VMAX,
    ];
  }

  function phi(s) {
    const out = new Float64Array(D);
    out[0] = 1;
    for (let i = 0; i < RAW; i++) out[1 + i] = s[i];
    for (let j = 0; j < M; j++) {
      const k = freq[j];
      let dot = 0;
      for (let i = 0; i < RAW; i++) dot += k[i] * s[i];
      out[1 + RAW + 2 * j] = Math.cos(dot);
      out[2 + RAW + 2 * j] = Math.sin(dot);
    }
    return out;
  }

  function read(s) {
    const f = phi(s);
    let dx = 0, dy = 0;
    for (let i = 0; i < D; i++) { dx += WM[0][i] * f[i]; dy += WM[1][i] * f[i]; }
    return [dx, dy];
  }

  function observe(prevP, nextP, chaser) {
    // learn the transition: state_t (prevP) -> next velocity (nextP)
    const s = raw(prevP, chaser);
    const f = phi(s);
    const yx = nextP.vx / VMAX, yy = nextP.vy / VMAX;
    const [dx, dy] = read(s);
    const ex = yx - dx, ey = yy - dy;
    errSq += ex * ex + ey * ey; errN++;
    // normalized three-factor Hebbian update with weight decay
    let nrm = 0;
    for (let i = 0; i < D; i++) nrm += f[i] * f[i];
    const lr = eta / (1 + nrm);
    const wd = 1 - lr * lam;
    for (let i = 0; i < D; i++) {
      WM[0][i] = WM[0][i] * wd + lr * ex * f[i];
      WM[1][i] = WM[1][i] * wd + lr * ey * f[i];
    }
  }

  function predictAt(p, chaser, steps) {
    steps = clamp(steps, 1, TAU_CAP_STEPS);
    let px = p.px, py = p.py, vx = p.vx, vy = p.vy;
    for (let i = 0; i < steps; i++) {
      const s = [
        wrapDelta(px, chaser.cx, W) / R,
        wrapDelta(py, chaser.cy, H) / R,
        vx / VMAX, vy / VMAX,
      ];
      const [ux, uy] = read(s);       // predicted next velocity (normalized)
      vx = ux * VMAX; vy = uy * VMAX;
      px = ((px + vx * DT) % W + W) % W;
      py = ((py + vy * DT) % H + H) % H;
    }
    return { x: px, y: py };
  }
  function predictLead(p, chaser) { return predictAt(p, chaser, leadSteps(p, chaser)); }

  return { observe, predictAt, predictLead, err: () => (errN ? Math.sqrt(errSq / errN) : 0) };
}

// ---------------- episode ----------------
const PRED_HORIZONS = [20, 40];      // steps (1s, 2s) for the forecast-error metric

function runEpisode(preyType, predictorName, seed, horizon, opts = {}) {
  const rng = makeRng(seed);
  const prey = makePrey(preyType, rng);
  const chaser = makeChaser();
  const predictor = makePredictor(predictorName, rng);
  let catches = 0, cooldown = 0;
  let sumDist = 0;

  // lead-horizon forecast error: |predicted future prey position - actual|,
  // measured at fixed horizons (independent of catch/cooldown mechanics).
  const q = PRED_HORIZONS.map(h => ({ h, buf: [] }));
  let fErrSum = 0, fErrN = 0;

  for (let t = 0; t < horizon; t++) {
    const p = prey.P;
    const prevSnap = { px: p.px, py: p.py, vx: p.vx, vy: p.vy };

    // record fixed-horizon forecasts, settle those that have matured
    for (const e of q) {
      const pr = predictor.predictAt(p, chaser, e.h);
      e.buf.push({ x: pr.x, y: pr.y, due: t + e.h });
    }
    for (const e of q) {
      while (e.buf.length && e.buf[0].due <= t) {
        fErrSum += dist(e.buf[0].x, e.buf[0].y, p.px, p.py);
        fErrN++;
        e.buf.shift();
      }
    }

    // plan: steer at predicted future prey position (lead pursuit)
    const lead = predictor.predictLead(p, chaser);
    const targetH = Math.atan2(wrapDelta(lead.y, chaser.cy, H), wrapDelta(lead.x, chaser.cx, W));
    chaser.heading = turnToward(chaser.heading, targetH, CHASER_MAXTURN);

    // move chaser
    chaser.speed = CHASE_MAX;
    chaser.cx = ((chaser.cx + Math.cos(chaser.heading) * chaser.speed * DT) % W + W) % W;
    chaser.cy = ((chaser.cy + Math.sin(chaser.heading) * chaser.speed * DT) % H + H) % H;

    // move prey, then train the world model on the observed transition
    prey.step(DT, chaser);
    predictor.observe(prevSnap, prey.P, chaser);

    // catch check
    const d2 = dist(p.px, p.py, chaser.cx, chaser.cy);
    sumDist += d2;
    if (cooldown > 0) cooldown--;
    if (d2 < CATCH_RADIUS && cooldown === 0) {
      catches++; cooldown = COOLDOWN_STEPS;
      if (opts.resetOnCatch) {
        // teleport prey far away -> each catch is a fresh interception from distance
        const ang = rng.next() * 2 * Math.PI, dd = 400 + rng.next() * 200;
        prey.P.px = ((chaser.cx + Math.cos(ang) * dd) % W + W) % W;
        prey.P.py = ((chaser.cy + Math.sin(ang) * dd) % H + H) % H;
        for (const e of q) e.buf.length = 0;   // invalidate forecasts across the teleport
      }
    }
  }
  return {
    catches,
    meanDist: sumDist / horizon,
    err: predictor.err ? predictor.err() : null,
    forecastErr: fErrN ? fErrSum / fErrN : 0,
  };
}

// ================= Model-free value/policy baselines =================
// These learn state -> action (a discrete target heading) directly, with no
// predictive world model. The reward is dense (negative distance) plus a sparse
// +1 catch bonus, so failure is a credit-assignment/interception failure, not
// merely "no reward signal".
const N_ACTIONS = process.env.NACT != null ? parseInt(process.env.NACT, 10) : 8;
const ACTIONS = Array.from({ length: N_ACTIONS }, (_, a) => a * (2 * Math.PI / N_ACTIONS));

function stateOf(p, chaser) {
  const dx = wrapDelta(p.px, chaser.cx, W), dy = wrapDelta(p.py, chaser.cy, H);
  const d = Math.hypot(dx, dy);
  return [
    dx / 600, dy / 600,
    p.vx / PREY_VMAX, p.vy / PREY_VMAX,
    Math.cos(chaser.heading), Math.sin(chaser.heading),
    d / 600,
  ];
}

// --- tiny MLP (ReLU hidden, linear output) with Adam ---
function makeMlp(sizes, rng) {
  const L = sizes.length - 1;
  const W = [], b = [], mW = [], vW = [], mb = [], vb = [];
  for (let l = 0; l < L; l++) {
    const nIn = sizes[l], nOut = sizes[l + 1], scale = Math.sqrt(2 / nIn);
    const Wl = [], mWl = [], vWl = [];
    for (let j = 0; j < nOut; j++) {
      const row = new Float64Array(nIn), mrow = new Float64Array(nIn), vrow = new Float64Array(nIn);
      for (let k = 0; k < nIn; k++) row[k] = rng.gauss() * scale;
      Wl.push(row); mWl.push(mrow); vWl.push(vrow);
    }
    W.push(Wl); b.push(new Float64Array(nOut));
    mW.push(mWl); vW.push(vWl); mb.push(new Float64Array(nOut)); vb.push(new Float64Array(nOut));
  }
  return { W, b, mW, vW, mb, vb, L, t: 0 };
}
function mlpForward(net, x) {
  const acts = [x], zs = [];
  let a = x;
  for (let l = 0; l < net.L; l++) {
    const Wl = net.W[l], bl = net.b[l];
    const z = new Float64Array(Wl.length);
    for (let j = 0; j < Wl.length; j++) {
      let s = bl[j]; const row = Wl[j];
      for (let k = 0; k < a.length; k++) s += row[k] * a[k];
      z[j] = s;
    }
    zs.push(z);
    a = l < net.L - 1 ? z.map(v => (v > 0 ? v : 0)) : z;
    acts.push(a);
  }
  return { acts, zs };
}
function mlpBackward(net, acts, zs, dOut, lr) {
  net.t++;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const b1c = 1 - Math.pow(b1, net.t), b2c = 1 - Math.pow(b2, net.t);
  let grads = dOut;
  for (let l = net.L - 1; l >= 0; l--) {
    const Wl = net.W[l], bl = net.b[l], z = zs[l], aPrev = acts[l];
    let dz = grads;
    if (l < net.L - 1) {
      dz = new Float64Array(z.length);
      for (let j = 0; j < z.length; j++) dz[j] = z[j] > 0 ? grads[j] : 0;
    }
    const gradsPrev = new Float64Array(aPrev.length);
    for (let k = 0; k < aPrev.length; k++) {
      let s = 0;
      for (let j = 0; j < Wl.length; j++) s += Wl[j][k] * dz[j];
      gradsPrev[k] = s;
    }
    for (let j = 0; j < Wl.length; j++) {
      const row = Wl[j], mrow = net.mW[l][j], vrow = net.vW[l][j];
      for (let k = 0; k < row.length; k++) {
        const g = dz[j] * aPrev[k];
        mrow[k] = b1 * mrow[k] + (1 - b1) * g;
        vrow[k] = b2 * vrow[k] + (1 - b2) * g * g;
        row[k] -= lr * (mrow[k] / b1c) / (Math.sqrt(vrow[k] / b2c) + eps);
      }
      const gb = dz[j];
      net.mb[l][j] = b1 * net.mb[l][j] + (1 - b1) * gb;
      net.vb[l][j] = b2 * net.vb[l][j] + (1 - b2) * gb * gb;
      bl[j] -= lr * (net.mb[l][j] / b1c) / (Math.sqrt(net.vb[l][j] / b2c) + eps);
    }
    grads = gradsPrev;
  }
}

function makePolicy(name, rng) {
  if (name === 'linear-q') {
    // linear FA + TD(0) bootstrapping + off-policy (the "deadly triad")
    const D = 8, w = Array.from({ length: N_ACTIONS }, () => new Float64Array(D));
    let eps = 1.0, stepC = 0;
    const phi = s => [1, s[0], s[1], s[2], s[3], s[4], s[5], s[6]];
    const q = (s, a) => { const f = phi(s); let z = 0; for (let i = 0; i < D; i++) z += w[a][i] * f[i]; return z; };
    return {
      act(s) {
        if (rng.next() < eps) return Math.floor(rng.next() * N_ACTIONS);
        let best = 0; for (let a = 1; a < N_ACTIONS; a++) if (q(s, a) > q(s, best)) best = a;
        return best;
      },
      observe(prevS, a, r, nextS) {
        stepC++; eps = Math.max(0.05, 1 - stepC / 8000);
        let qmax = -Infinity; for (let b = 0; b < N_ACTIONS; b++) qmax = Math.max(qmax, q(nextS, b));
        const td = r + 0.95 * qmax - q(prevS, a);
        const f = phi(prevS);
        for (let i = 0; i < D; i++) w[a][i] += 0.01 * td * f[i];
      },
    };
  }

  if (name === 'dqn') {
    const net = makeMlp([7, 32, 32, N_ACTIONS], rng);
    const target = makeMlp([7, 32, 32, N_ACTIONS], rng);
    const copy = () => { for (let l = 0; l < net.L; l++) { target.W[l] = net.W[l].map(r => r.slice()); target.b[l] = net.b[l].slice(); } };
    const replay = [];
    let eps = 1.0, stepC = 0;
    const qvals = s => mlpForward(net, s).acts[net.L];
    return {
      act(s) {
        if (rng.next() < eps) return Math.floor(rng.next() * N_ACTIONS);
        const q = qvals(s); let best = 0; for (let a = 1; a < N_ACTIONS; a++) if (q[a] > q[best]) best = a;
        return best;
      },
      observe(prevS, a, r, nextS) {
        stepC++; eps = Math.max(0.05, 1 - stepC / 8000);
        replay.push({ s: prevS, a, r, next: nextS });
        if (replay.length > 10000) replay.shift();
        if (stepC % 8 === 0 && replay.length >= 64) {
          for (let b = 0; b < 32; b++) {
            const tr = replay[Math.floor(rng.next() * replay.length)];
            const fwd = mlpForward(net, tr.s);
            const qn = mlpForward(target, tr.next).acts[target.L];
            const targetVal = tr.r + 0.95 * Math.max(...qn);
            const dOut = new Float64Array(N_ACTIONS);
            dOut[tr.a] = 2 * (fwd.acts[net.L][tr.a] - targetVal);
            mlpBackward(net, fwd.acts, fwd.zs, dOut, 3e-4);
          }
        }
        if (stepC % 500 === 0) copy();
      },
    };
  }

  if (name === 'ppo') {
    // clipped-surrogate PPO (discrete), policy + value MLPs
    const pi = makeMlp([7, 64, 64, N_ACTIONS], rng);
    const vf = makeMlp([7, 64, 64, 1], rng);
    let buf = [];
    const GAMMA = 0.99, LAM = 0.95, CLIP = 0.2, LR = 3e-4, EPOCHS = 4, BATCH = 256;
    const logits = s => mlpForward(pi, s).acts[pi.L];
    const value = s => mlpForward(vf, s).acts[vf.L][0];
    function probsOf(lg) {
      const mx = Math.max(...lg);
      const ex = lg.map(l => Math.exp(l - mx));
      const sum = ex.reduce((a, c) => a + c, 0);
      return ex.map(e => e / sum);
    }
    return {
      act(s) {
        const p = probsOf(logits(s));
        let r = rng.next(), c = 0, a = 0;
        for (let i = 0; i < N_ACTIONS; i++) { c += p[i]; if (r <= c) { a = i; break; } }
        return a;
      },
      observe(prevS, a, r, nextS) {
        const p = probsOf(logits(prevS));
        buf.push({ s: prevS, a, r, logp: Math.log(p[a] + 1e-12), v: value(prevS) });
        if (buf.length >= BATCH) {
          const T = buf.length;
          const adv = new Float64Array(T), ret = new Float64Array(T);
          const vNext = value(nextS);
          let gae = 0;
          for (let t = T - 1; t >= 0; t--) {
            const vn = t === T - 1 ? vNext : buf[t + 1].v;
            const delta = buf[t].r + GAMMA * vn - buf[t].v;
            gae = delta + GAMMA * LAM * gae;
            adv[t] = gae; ret[t] = gae + buf[t].v;
          }
          const am = adv.reduce((x, y) => x + y, 0) / T;
          const sd = Math.sqrt(adv.reduce((x, y) => x + (y - am) ** 2, 0) / T + 1e-8);
          for (let e = 0; e < EPOCHS; e++) {
            for (let t = 0; t < T; t++) {
              const A = (adv[t] - am) / sd;
              const p2 = probsOf(logits(buf[t].s));
              const ratio = Math.exp(Math.log(p2[buf[t].a] + 1e-12) - buf[t].logp);
              if (ratio > 1 - CLIP && ratio < 1 + CLIP) {
                const fwd = mlpForward(pi, buf[t].s);
                const dOut = new Float64Array(N_ACTIONS);
                for (let i = 0; i < N_ACTIONS; i++) dOut[i] = -A * ((i === buf[t].a ? 1 : 0) - p2[i]);
                mlpBackward(pi, fwd.acts, fwd.zs, dOut, LR);
              }
              const vfFwd = mlpForward(vf, buf[t].s);
              const dOutV = new Float64Array(1); dOutV[0] = 2 * (vfFwd.acts[vf.L][0] - ret[t]);
              mlpBackward(vf, vfFwd.acts, vfFwd.zs, dOutV, LR);
            }
          }
          buf = [];
        }
      },
    };
  }

  throw new Error('unknown policy ' + name);
}

function runPolicyEpisode(preyType, policyName, seed, horizon, opts = {}) {
  const rng = makeRng(seed);
  const prey = makePrey(preyType, rng);
  const chaser = makeChaser();
  const policy = (policyName === 'pure-pursuit' || policyName === 'mpc') ? null : makePolicy(policyName, rng);
  let catches = 0, cooldown = 0, sumDist = 0;

  for (let t = 0; t < horizon; t++) {
    const p = prey.P;
    const s = stateOf(p, chaser);
    let a = 0;
    if (policyName === 'pure-pursuit') {
      chaser.heading = turnToward(chaser.heading,
        Math.atan2(wrapDelta(p.py, chaser.cy, H), wrapDelta(p.px, chaser.cx, W)), CHASER_MAXTURN);
    } else if (policyName === 'mpc') {
      // model-predictive control: search headings, simulate H steps with a
      // first-order (constant-velocity) prey model, minimize final distance.
      const d0 = dist(p.px, p.py, chaser.cx, chaser.cy);
      const HZ = clamp(Math.round((d0 / CHASE_MAX) / DT), 1, 40);
      let best = 0, bestD = Infinity;
      for (let aa = 0; aa < N_ACTIONS; aa++) {
        let h = chaser.heading, cx = chaser.cx, cy = chaser.cy;
        let px = p.px, py = p.py;
        for (let i = 0; i < HZ; i++) {
          h = turnToward(h, ACTIONS[aa], CHASER_MAXTURN);
          cx = ((cx + Math.cos(h) * CHASE_MAX * DT) % W + W) % W;
          cy = ((cy + Math.sin(h) * CHASE_MAX * DT) % H + H) % H;
          px = ((px + p.vx * DT) % W + W) % W;
          py = ((py + p.vy * DT) % H + H) % H;
        }
        const d = dist(px, py, cx, cy);
        if (d < bestD) { bestD = d; best = aa; }
      }
      a = best;
      chaser.heading = turnToward(chaser.heading, ACTIONS[best], CHASER_MAXTURN);
    } else {
      a = policy.act(s);
      chaser.heading = turnToward(chaser.heading, ACTIONS[a], CHASER_MAXTURN);
    }

    // move chaser
    chaser.cx = ((chaser.cx + Math.cos(chaser.heading) * CHASE_MAX * DT) % W + W) % W;
    chaser.cy = ((chaser.cy + Math.sin(chaser.heading) * CHASE_MAX * DT) % H + H) % H;

    // move prey
    prey.step(DT, chaser);

    // reward + train
    const d2 = dist(p.px, p.py, chaser.cx, chaser.cy);
    sumDist += d2;
    let reward = -d2 / 600;
    if (cooldown > 0) cooldown--;
    if (d2 < CATCH_RADIUS && cooldown === 0) {
      catches++; cooldown = COOLDOWN_STEPS; reward += 1;
      if (opts.resetOnCatch) {
        const ang = rng.next() * 2 * Math.PI, dd = 400 + rng.next() * 200;
        prey.P.px = ((chaser.cx + Math.cos(ang) * dd) % W + W) % W;
        prey.P.py = ((chaser.cy + Math.sin(ang) * dd) % H + H) % H;
      }
    }
    if (policy) policy.observe(s, a, reward, stateOf(prey.P, chaser));
  }
  return { catches, meanDist: sumDist / horizon };
}

// ---------------- CLI / smoke test ----------------
if (require.main === module) {
  const horizon = parseInt(process.argv[2] || '24000', 10);
  const preyTypes = process.argv.slice(3).length ? process.argv.slice(3) : ['const-vel', 'circling', 'ou-turn', 'ou-vel', 'jump', 'flee'];
  const predictors = ['velocity-lead', 'accel-lead', 'kalman-lead', 'bdh'];
  const policies = ['pure-pursuit', 'mpc', 'linear-q', 'dqn', 'ppo'];
  const nSeeds = parseInt(process.env.SEEDS || '3', 10);
  const seeds = Array.from({ length: nSeeds }, (_, i) => i + 1);
  const reset = process.env.RESET === '1';
  console.log(`horizon=${horizon} steps | seeds=${seeds.length} | resetOnCatch=${reset}`);
  console.log('--- lead-pursuit predictors (catches | forecastErr px) ---');
  for (const pt of preyTypes) {
    const row = [pt];
    for (const pr of predictors) {
      let cat = 0, fErr = 0;
      for (const s of seeds) {
        const r = runEpisode(pt, pr, s, horizon, { resetOnCatch: reset });
        cat += r.catches; fErr += r.forecastErr;
      }
      row.push(`${pr} ${(cat / seeds.length).toFixed(1)} | ${(fErr / seeds.length).toFixed(1)}px`);
    }
    console.log(row.join('\t'));
  }
  if (process.env.NO_POLICY !== '1') {
    console.log('--- model-free policies (catches) ---');
    for (const pt of preyTypes) {
      const row = [pt];
      for (const pl of policies) {
        let cat = 0;
        for (const s of seeds) {
          cat += runPolicyEpisode(pt, pl, s, horizon, { resetOnCatch: reset }).catches;
        }
        row.push(`${pl} ${(cat / seeds.length).toFixed(1)}`);
      }
      console.log(row.join('\t'));
    }
  }
}

module.exports = { runEpisode, runPolicyEpisode, makeRng, makePrey, makeChaser, makePredictor, makePolicy, W, H, DT, CATCH_RADIUS, PREY_VMAX, CHASER_MAXTURN };
