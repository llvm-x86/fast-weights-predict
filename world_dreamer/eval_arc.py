#!/usr/bin/env python3
# eval_arc.py — evaluate the discrete world-dreamer (arc.py) on ARC-AGI-1.
#
# Usage:  python3 eval_arc.py /path/to/arc/data/training
#
# Reports the fraction of tasks whose held-out test output is reproduced exactly.
# This is an honest, transparent coverage number over the ARC-AGI-1 *training*
# set (the evaluation set's test outputs are the leaderboard's held-out target,
# so offline reports use the training set or submit to the leaderboard).

import json
import os
import sys
import time

import arc


def evaluate(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    solved = []
    failed = []
    t0 = time.time()
    for i, fn in enumerate(files):
        task = json.load(open(os.path.join(data_dir, fn)))
        outs = arc.solve_task(task)
        ok = outs is not None and all(
            arc._tup(outs[k]) == arc._tup(t['output']) for k, t in enumerate(task['test']))
        (solved if ok else failed).append(fn)
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)}  solved={len(solved)}  ({time.time()-t0:.0f}s)', flush=True)
    return solved, failed


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/training'
    solved, failed = evaluate(data_dir)
    n = len(solved) + len(failed)
    print(f'\nARC-AGI-1 training set: {len(solved)}/{n} tasks solved '
          f'({100.0 * len(solved) / n:.1f}%)')
    print('\nSolved:', ' '.join(sorted(solved)) if solved else '(none)')
    print('\nFailed:', ' '.join(sorted(failed)))
