#!/usr/bin/env python3
"""pass_at_k.py — score the solver by how many attempts it needs.

The solver's search produces a *stream* of programs that each reproduce every
training example (`arc.iter_programs`).  Reporting only the first one throws most
of that away: where the examples under-determine the rule, the first verified
program is a guess, and a later one is often the right guess.

Two honest ways to use the stream:

  pass@k  — accept the answer if ANY of the first k distinct predictions is
            correct.  The official ARC metric allows two attempts per test input,
            so pass@2 is the number comparable to a leaderboard score; pass@1 is
            what this README reports elsewhere.
  vote    — pick the prediction that the largest number of verified programs
            agree on, i.e. use consensus instead of enumeration order, and then
            score that single answer.  This is a genuine change to pass@1, not a
            free extra attempt.

Usage:
    python3 pass_at_k.py [module_dir] [workers] [k]
"""
import json
import glob
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

MODDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sys.path.insert(0, MODDIR)

DATASETS = [
    ('/tmp/arc-agi/data/training', 'ARC-AGI-1 train'),
    ('/tmp/ARC-AGI-2/data/training', 'ARC-AGI-2 train'),
    ('/tmp/arc-agi/data/evaluation', 'ARC-AGI-1 eval'),
    ('/tmp/ARC-AGI-2/data/evaluation', 'ARC-AGI-2 eval'),
]
MAX_PROGRAMS = 24


def one(path):
    import arc
    task = json.load(open(path))
    truth = tuple(arc._tup(t['output']) for t in task['test'])
    try:
        cands = arc.solve_task_multi(task, 2, 16, want=K, max_programs=MAX_PROGRAMS)
    except Exception:
        return ([], None)
    return (cands, truth)


if __name__ == '__main__':
    for d, label in DATASETS:
        files = sorted(glob.glob(d + '/*.json'))
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            res = list(pool.map(one, files, chunksize=2))
        n = len(files)
        passk = [0] * (K + 1)
        vote = 0
        none = 0
        for cands, truth in res:
            if not cands:
                none += 1
                continue
            for k in range(1, K + 1):
                if truth in cands[:k]:
                    passk[k] += 1
            top = Counter(cands).most_common(1)[0][0]
            if top == truth:
                vote += 1
        line = '  '.join('pass@%d=%d' % (k, passk[k]) for k in range(1, K + 1))
        pct = '  '.join('%.1f%%' % (100.0 * passk[k] / n) for k in range(1, K + 1))
        print('%-18s n=%4d' % (label, n))
        print('    attempts  %s' % line)
        print('    percent   %s' % pct)
        print('    consensus vote %d (%.1f%%)   tasks with no candidate %d'
              % (vote, 100.0 * vote / n, none))
        sys.stdout.flush()
