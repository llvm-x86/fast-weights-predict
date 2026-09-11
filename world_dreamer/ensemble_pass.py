#!/usr/bin/env python3
"""ensemble_pass.py — score the two substrates as a 2-attempt submission.

`combined.py` is a *cascade*: it returns the DSL's answer whenever the DSL has
one, and only falls back to the learned model when the DSL returns nothing.  That
throws away the learned answer on exactly the tasks where the DSL answered and
was **wrong** — and this project has measured, repeatedly, that the DSL answers
wrongly while fitting every training example (2 tasks on ARC-AGI-1 training, 4 on
ARC-AGI-2, 1 on ARC-AGI-1 evaluation).

The official ARC metric accepts two attempts per test input, so those tasks are
recoverable without any new capability: submit the DSL's answer *and* the learned
answer.  This script measures exactly how much that recovers, and reports it
separately from the cascade so the two are never conflated.

Usage:
    python3 ensemble_pass.py [module_dir] [workers]
"""
import json
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

MODDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sys.path.insert(0, MODDIR)

DATASETS = [
    ('/tmp/arc-agi/data/training', 'ARC-AGI-1 train'),
    ('/tmp/ARC-AGI-2/data/training', 'ARC-AGI-2 train'),
    ('/tmp/arc-agi/data/evaluation', 'ARC-AGI-1 eval'),
    ('/tmp/ARC-AGI-2/data/evaluation', 'ARC-AGI-2 eval'),
]


def one(path):
    import arc
    import learned
    task = json.load(open(path))
    truth = tuple(arc._tup(t['output']) for t in task['test'])

    def preds_of(outs):
        if outs is None:
            return None
        return tuple(arc._tup(g) for g in outs)

    dsl = preds_of(arc.solve_task(task, 2, 16))
    lrn = preds_of(learned.solve_task(task))
    return (dsl == truth, lrn == truth, dsl is not None, lrn is not None)


if __name__ == '__main__':
    for d, label in DATASETS:
        files = sorted(glob.glob(d + '/*.json'))
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            res = list(pool.map(one, files, chunksize=2))
        n = len(files)
        dsl_ok = sum(1 for a, _, _, _ in res if a)
        lrn_ok = sum(1 for _, b, _, _ in res if b)
        # cascade: DSL answer if it has one, else learned (what combined.py does)
        casc = sum(1 for a, b, has_d, _ in res if a or (not has_d and b))
        # 2-attempt submission: both answers count
        pass2 = sum(1 for a, b, _, _ in res if a or b)
        # tasks where the DSL answered WRONG and the learned model is right
        recov = sorted(os.path.basename(f) for f, (a, b, hd, hl) in
                       zip(files, res) if (not a) and b and hd)
        print('%-18s n=%4d' % (label, n))
        print('    DSL alone            %3d (%.1f%%)' % (dsl_ok, 100.0 * dsl_ok / n))
        print('    learned alone        %3d (%.1f%%)' % (lrn_ok, 100.0 * lrn_ok / n))
        print('    cascade (combined)   %3d (%.1f%%)' % (casc, 100.0 * casc / n))
        print('    pass@2 (both)        %3d (%.1f%%)   recovered %d'
              % (pass2, 100.0 * pass2 / n, pass2 - casc))
        if recov:
            print('    recovered by attempt 2:', recov[:20])
        sys.stdout.flush()
