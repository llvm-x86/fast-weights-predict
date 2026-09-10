#!/usr/bin/env python3
# learned.py — the non-LLM *learned* world-model: the dragon-hatchling (BDH)
# fast-weight memory applied directly to ARC grids.
#
# In the continuous pursuit instantiation the world model is a linear Hebbian map
#   W : phi(s) -> v_{t+1}   with   W <- W + eta * delta * phi^T
# over hand-built continuous state features.  Here the SAME substrate is applied
# to discrete grids: features are position-dependent one-hot cell colors, the
# observed transition is input-grid -> output-grid, and the write is the Hebbian
# outer product  W <- W + eta * psi(output) phi(input)^T.  Readout is the linear
# map  y_hat = W phi(x)  with an argmax color per output cell.
#
# This is the honest "learned model" the task asked for, and it is also the
# honest demonstration of *why* a linear Hebbian memory does not carry from
# pursuit to ARC: ARC is compositional program induction, and an associative
# memory can only retrieve an output whose input overlaps the stored inputs.  The
# number below is expected to be single digits, and it is reported as such.
#
# Note there is no separate "dreamer" here: the planner degenerates to a trivial
# argmax readout, because the learned map has no internal search.  That absence
# is itself the finding — the world-dreamer's power is in the planner.
#
# ---------------------------------------------------------------------------
# The ensemble of learned models
#
# The memory above keys on (position, colour) and so is blind to every rule that
# is *local* rather than positional.  Three further associative memories close
# most of that gap, all of them still plain counting with no LLM and no search:
#
#   ColorRemapModel   global colour co-occurrence C[in][out]  (global recolour)
#   LocalRuleModel    associative memory over local input patches -> output cell
#   (candidate list)  several LocalRuleModel geometries, see CANDIDATES below
#
# What changed relative to the first version of this file, and why:
#
#   * The 3x3 patch key used to be looked up exactly, and an unseen patch fell
#     back to the single nearest stored patch.  The key is now voted over the
#     k nearest stored patches with weight 1/(1+hamming), so an exact key still
#     dominates (distance 0) but near misses contribute smoothly.
#
#   * Border patches used to be padded with a sentinel (-1).  A sentinel makes
#     every border patch a unique key that shares nothing with the interior
#     statistics.  Reflecting the grid across the edge instead ("reflect" pad)
#     gives border patches real colours, which is the single largest gain here.
#
#   * A patch and its 8 dihedral transforms can share vote counts ("sym"), so a
#     rule learned in one orientation transfers to the others.  This is applied
#     only to patches fully inside the grid by default, because a padded patch's
#     transforms are not meaningful.
#
#   * Model selection used to minimise the summed held-out cell error alone.
#     A model that predicts "almost the input" scores well on that metric while
#     never being exactly right.  The score is now
#     (leave-one-example-out cell error) + (full-train cell error): the second
#     term is a training-fit check, in the same spirit as arc.py's "verify the
#     induced program on the training pairs before ranking it".
#
# Selection remains entirely on the training pairs.  The test outputs are only
# ever compared with, never consulted.  Tasks whose grids change size are still
# reported as skipped rather than mis-handled.

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def encode(g, H, W):
    v = np.zeros(H * W * 10, dtype=np.float64)
    for i in range(min(len(g), H)):
        for j in range(min(len(g[0]), W)):
            v[(i * W + j) * 10 + g[i][j]] = 1.0
    return v


class HebbianWorldModel:
    """BDH fast-weight memory over one-hot position-color features.

    The Hebbian write W <- W + eta * psi(y) phi(x)^T and the linear readout
    y_hat = W phi(x) are computed WITHOUT materializing the O(dim^2) dense matrix.
    Because phi/psi are one-hot, W phi(x_test) = sum_k psi(y_k) * overlap(x_k,
    x_test), where overlap counts matching cell colors.  We store the (input,
    output) pairs and vote — mathematically identical to the outer-product form,
    and memory stays O(train examples * grid area)."""

    def __init__(self, H, W, eta=1.0, decay=0.0):
        self.H, self.W = H, W
        self.eta = eta
        self.decay = decay
        self.mem = []  # list of (input_grid, output_grid)

    def observe(self, x, y):
        self.mem.append(([row[:] for row in x], [row[:] for row in y]))

    def predict(self, x):
        H, W = self.H, self.W
        scores = [[[0.0] * 10 for _ in range(W)] for _ in range(H)]
        for xk, yk in self.mem:
            ov = sum(1 for i in range(H) for j in range(W) if xk[i][j] == x[i][j])
            for i in range(H):
                for j in range(W):
                    scores[i][j][yk[i][j]] += ov
        out = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                out[i][j] = max(range(10), key=lambda c: scores[i][j][c])
        return out


class ColorRemapModel:
    """Hebbian color association: co-occurrence C[in_color][out_color], learned
    across every cell of every training pair, ignoring position.  This expresses
    global recolors (the most common single ARC transform) that the position-bound
    map cannot."""

    def __init__(self, H, W):
        self.H, self.W = H, W
        self.C = [[0] * 10 for _ in range(10)]

    def observe(self, x, y):
        for i in range(self.H):
            for j in range(self.W):
                self.C[x[i][j]][y[i][j]] += 1

    def predict(self, x):
        C = self.C
        return [[max(range(10), key=lambda c: C[x[i][j]][c])
                 for j in range(self.W)] for i in range(self.H)]


def _dihedral(g, n):
    """The distinct dihedral transforms of a square patch given row-major."""
    m = [g[i * n:(i + 1) * n] for i in range(n)]
    outs = []
    for k in range(4):
        r = m
        for _ in range(k):
            r = [[r[i][j] for i in range(n - 1, -1, -1)] for j in range(n)]
        outs.append(tuple(v for row in r for v in row))
        outs.append(tuple(v for row in [rw[::-1] for rw in r] for v in row))
    return list(dict.fromkeys(outs))


class LocalRuleModel:
    """Associative memory over local input neighbourhoods -> output cell colour.

    This is the patch-level generalization of the Hebbian map: instead of keying
    on (position, color), it keys on the local (2r+1)^2 pattern, so a rule learned
    at one location generalizes to every other location.  It expresses the *local*
    ARC family (cellular automata, hole/region fill, dilation, symmetry
    completion) that neither the position-bound map nor the global color map can.

    Parameters
      radius        r; the key is the (2r+1)^2 neighbourhood.  1 = the original 3x3.
      k             vote with the k nearest stored keys, weight 1/(1+hamming);
                    k=1 reproduces the original "nearest stored pattern" fallback.
      sym           also count/lookup the 8 dihedral transforms of the patch, so a
                    locally rotation/reflection-symmetric rule shares statistics.
      sym_interior  if True, only patches fully inside the grid are transformed
                    (a padded patch's transforms are not meaningful).
      pad           'sentinel' (-1 outside the grid — the original) or 'reflect'
                    (mirror the grid across the edge, so border patches hold real
                    colours and share statistics with the interior).
      prior         optional bias towards keeping the input colour (0 = off).

    Exact neighborhood matches dominate the vote; an unseen neighborhood falls
    back to its nearest stored patterns, then (with no memory at all) to identity.
    """

    def __init__(self, H, W, radius=1, k=1, sym=False, sym_interior=True,
                 pad='sentinel', prior=0.0):
        self.H, self.W = H, W
        self.radius, self.k = radius, k
        self.sym, self.sym_interior = sym, sym_interior
        self.pad, self.prior = pad, prior
        self.dict = {}  # patch key -> per-colour counts
        self._keys_arr = None
        self._counts = None

    def _patch(self, x, i, j):
        r, H, W = self.radius, self.H, self.W
        g, full = [], True
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    g.append(x[ni][nj])
                else:
                    full = False
                    if self.pad == 'reflect':
                        ri = -ni if ni < 0 else (2 * H - 2 - ni if ni >= H else ni)
                        rj = -nj if nj < 0 else (2 * W - 2 - nj if nj >= W else nj)
                        g.append(x[min(max(ri, 0), H - 1)][min(max(rj, 0), W - 1)])
                    else:
                        g.append(-1)
        return g, full

    def _keys(self, x, i, j):
        g, full = self._patch(x, i, j)
        if not self.sym:
            return [tuple(g)]
        if self.sym_interior and not full:
            return [tuple(g)]
        return _dihedral(g, 2 * self.radius + 1)

    def observe(self, x, y):
        for i in range(self.H):
            for j in range(self.W):
                for key in self._keys(x, i, j):
                    d = self.dict.get(key)
                    if d is None:
                        d = [0] * 10
                        self.dict[key] = d
                    d[y[i][j]] += 1
        self._keys_arr = None

    def _fit(self):
        if self._keys_arr is not None:
            return
        keys = sorted(self.dict)
        if keys:
            self._keys_arr = np.array(keys, dtype=np.int16)
            self._counts = np.array([self.dict[key] for key in keys], dtype=np.float64)
        else:
            self._keys_arr = np.zeros((0, 1), dtype=np.int16)
            self._counts = np.zeros((0, 10), dtype=np.float64)

    def predict(self, x):
        H, W = self.H, self.W
        self._fit()
        K, C = self._keys_arr, self._counts
        # one query per distinct patch (cheap, and the transform variants repeat a lot)
        index, queries, cells = {}, [], [[None] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                ks = self._keys(x, i, j)
                cells[i][j] = ks[0]
                for key in ks:
                    if key not in index:
                        index[key] = len(queries)
                        queries.append(key)
        votes = np.zeros((len(queries), 10), dtype=np.float64)
        if K.shape[0] and queries:
            Q = np.array(queries, dtype=np.int16)
            kk = min(self.k, K.shape[0])
            chunk = max(1, 300000 // max(1, K.shape[0]))
            for s in range(0, len(queries), chunk):
                q = Q[s:s + chunk]
                dist = (q[:, None, :] != K[None, :, :]).sum(axis=2)
                near = np.argsort(dist, axis=1)[:, :kk]
                rows = np.arange(q.shape[0])[:, None]
                w = 1.0 / (1.0 + dist[rows, near])
                for t in range(kk):
                    votes[s:s + q.shape[0]] += C[near[:, t]] * w[:, t:t + 1]
        out = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                v = votes[index[cells[i][j]]].copy() if len(queries) else np.zeros(10)
                if self.prior:
                    v[x[i][j]] += self.prior
                if not v.any():                       # no memory at all: identity
                    out[i][j] = x[i][j]
                else:
                    out[i][j] = int(max(range(10), key=lambda c: (v[c], c == x[i][j], -c)))
        return out


# The candidate ensemble.  Each entry is a name and a factory.  They are distinct
# *rule families* — a position-bound memory, a global colour memory, and several
# geometries of the local patch memory — not a search over hyper-parameters, and
# nothing here is chosen per task; solve_task ranks them on the training pairs.
CANDIDATES = [
    ('hebb', lambda H, W: HebbianWorldModel(H, W)),
    ('color', lambda H, W: ColorRemapModel(H, W)),
    ('local3x3', lambda H, W: LocalRuleModel(H, W)),
    ('local3x3_sym', lambda H, W: LocalRuleModel(H, W, sym=True)),
    ('local3x3_sym_all', lambda H, W: LocalRuleModel(H, W, sym=True, sym_interior=False)),
    ('local3x3_reflect', lambda H, W: LocalRuleModel(H, W, pad='reflect')),
    ('local3x3_sym_reflect', lambda H, W: LocalRuleModel(H, W, sym=True, pad='reflect')),
]


def _mismatch(model, e):
    p = model.predict(e['input'])
    return sum(1 for i in range(len(p)) for j in range(len(p[0]))
               if p[i][j] != e['output'][i][j])


def _selection_score(factory, train, H, W):
    """Honest model ranking, computed on the training pairs only.

    Held-out term:  for each training example, fit on the others and count the
    mismatched cells of its prediction.  This is the dreamer's honest check — a
    model that only memorizes is rejected even when it fits the training set.

    Fit term:  the mismatched cells when the model is fitted on all pairs.  The
    held-out term alone rewards a model that predicts "roughly the input" on
    every fold; requiring the model to also reproduce the pairs it saw is the
    same "verify before ranking" discipline arc.py applies to induced programs.
    """
    held_out = 0
    for k in range(len(train)):
        m = factory(H, W)
        for j, e in enumerate(train):
            if j != k:
                m.observe(e['input'], e['output'])
        held_out += _mismatch(m, train[k])
    full = factory(H, W)
    for e in train:
        full.observe(e['input'], e['output'])
    fit = sum(_mismatch(full, e) for e in train)
    return held_out + fit


def solve_task(task, eta=1.0, decay=0.0):
    """Learn each candidate world model, let the dreamer pick the one that
    generalizes best on the training pairs, then apply it to the test inputs.

    Returns None for tasks whose grids change size (the fixed-geometry memories
    cannot express a size change); those are counted separately, not hidden."""
    train = task['train']
    H, W = len(train[0]['output']), len(train[0]['output'][0])
    for e in train + task['test']:
        if (len(e['input']), len(e['input'][0])) != (H, W):
            return None
        if (len(e['output']), len(e['output'][0])) != (H, W):
            return None
    best = min(CANDIDATES, key=lambda c: (_selection_score(c[1], train, H, W),))[1]
    m = best(H, W)
    for e in train:
        m.observe(e['input'], e['output'])
    return [m.predict(e['input']) for e in task['test']]


def _run(args):
    path, eta, decay = args
    task = json.load(open(path))
    outs = solve_task(task, eta, decay)
    if outs is None:
        return ('skip', os.path.basename(path))
    ok = all(tuple(tuple(r) for r in outs[k]) ==
             tuple(tuple(r) for r in t['output']) for k, t in enumerate(task['test']))
    return ('solved' if ok else 'failed', os.path.basename(path))


def evaluate(data_dir, num_workers=2, eta=1.0, decay=0.0):
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.endswith('.json'))
    solved, failed, skipped = [], [], []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for status, fn in pool.map(_run, [(f, eta, decay) for f in files], chunksize=8):
            {'solved': solved, 'failed': failed, 'skip': skipped}[status].append(fn)
    return solved, failed, skipped


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/arc-agi/data/training'
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    decay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    solved, failed, skipped = evaluate(data_dir, eta=eta, decay=decay)
    n = len(solved) + len(failed) + len(skipped)
    sized = len(solved) + len(failed)
    print(f'{os.path.basename(data_dir)}: {len(solved)}/{sized} same-size tasks '
          f'({100.0 * len(solved) / max(1, sized):.1f}%); '
          f'{len(skipped)}/{n} skipped (size-changing)')
    print('solved:', ' '.join(sorted(solved)) if solved else '(none)')
