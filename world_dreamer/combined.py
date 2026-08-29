#!/usr/bin/env python3
# combined.py — the faithful ensemble of the two non-LLM substrates.
#
# The symbolic world model (arc.py: induced program + program search) and the
# learned fast-weight world model (learned.py: BDH associative memory) solve
# *disjoint* slices of ARC: the DSL reaches single/compound transformations, the
# learned patch memory reaches local cell rules.  A "world dreamer" that keeps
# both and lets the dreamer choose is the honest realization of the user's
# "proceed with A & B in parallel": try the induced program first (it is exact and
# verifiable), and fall back to the learned map when no program verifies.
#
# Nothing here is an LLM.  Both components are non-LLM; the union is the number
# reported as "combined".

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import arc
import learned


def solve_task(task):
    """Induced program first, learned fast-weight map as fallback."""
    outs = arc.solve_task(task, max_depth=2, beam=16)
    if outs is not None:
        return outs
    return learned.solve_task(task)


def _run(path):
    task = json.load(open(path))
    outs = solve_task(task)
    if outs is None:
        return (False, os.path.basename(path))
    ok = all(tuple(tuple(r) for r in outs[k]) == tuple(tuple(r) for r in t['output'])
             for k, t in enumerate(task['test']))
    return (ok, os.path.basename(path))


def evaluate(data_dir, num_workers=12):
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.endswith('.json'))
    solved, failed = [], []
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for ok, fn in pool.map(_run, files, chunksize=8):
            (solved if ok else failed).append(fn)
    return solved, failed


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/arc-agi/data/training'
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    solved, failed = evaluate(data_dir, num_workers)
    n = len(solved) + len(failed)
    print(f'{os.path.basename(data_dir)} combined (DSL + learned): '
          f'{len(solved)}/{n} tasks solved ({100.0 * len(solved) / n:.1f}%)')
