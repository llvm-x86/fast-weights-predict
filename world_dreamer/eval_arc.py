#!/usr/bin/env python3
# eval_arc.py — evaluate the discrete world-dreamer (arc.py) on ARC-AGI-1.
#
# Usage:  python3 eval_arc.py /path/to/arc/data/training [num_workers]
#
# Reports the fraction of tasks whose held-out test output is reproduced exactly.
# This is an honest, transparent coverage number over the ARC-AGI-1 *training*
# set (the evaluation set's test outputs are the leaderboard's held-out target,
# so offline reports use the training set or submit to the leaderboard).
#
# Tasks are CPU-bound pure Python, so parallelism is processes (the GIL makes
# threads useless here), not threads.

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import arc


def _solve_one(args):
    """Load one task, solve it, and report whether every held-out test output is
    reproduced exactly.  Runs in a worker process."""
    path, max_depth, beam = args
    try:
        task = json.load(open(path))
    except Exception:
        return os.path.basename(path), False
    outs = arc.solve_task(task, max_depth=max_depth, beam=beam)
    ok = outs is not None and all(
        arc._tup(outs[k]) == arc._tup(t['output']) for k, t in enumerate(task['test']))
    return os.path.basename(path), ok


def evaluate(data_dir, num_workers=12, max_depth=2, beam=16):
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.endswith('.json'))
    solved = []
    failed = []
    t0 = time.time()
    jobs = [(f, max_depth, beam) for f in files]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for i, (fn, ok) in enumerate(pool.map(_solve_one, jobs, chunksize=8)):
            (solved if ok else failed).append(fn)
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(files)}  solved={len(solved)}  '
                      f'({time.time()-t0:.0f}s)', flush=True)
    return solved, failed


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/training'
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    beam = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    solved, failed = evaluate(data_dir, num_workers, max_depth, beam)
    n = len(solved) + len(failed)
    print(f'\nARC-AGI-1 training set (depth={max_depth}, beam={beam}): '
          f'{len(solved)}/{n} tasks solved ({100.0 * len(solved) / n:.1f}%)')
    print('\nSolved:', ' '.join(sorted(solved)) if solved else '(none)')
    print('\nFailed:', ' '.join(sorted(failed)))
