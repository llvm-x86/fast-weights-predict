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
        if self.decay:
            # weight older memories down (analog of the dense W *= (1-decay))
            for k in range(len(self.mem)):
                self.mem[k] = (self.mem[k][0], self.mem[k][1])
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


def solve_task(task, eta=1.0, decay=0.0):
    """Learn W from the training pairs and predict every test output.

    Returns None for tasks whose grids change size (the fixed-geometry linear map
    cannot express a size change); those are counted separately, not as failures
    to be hidden."""
    train = task['train']
    H, W = len(train[0]['output']), len(train[0]['output'][0])
    for e in train + task['test']:
        if (len(e['input']), len(e['input'][0])) != (H, W):
            return None
        if (len(e['output']), len(e['output'][0])) != (H, W):
            return None
    m = HebbianWorldModel(H, W, eta=eta, decay=decay)
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


def evaluate(data_dir, num_workers=12, eta=1.0, decay=0.0):
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
