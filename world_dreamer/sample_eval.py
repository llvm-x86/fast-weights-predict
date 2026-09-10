#!/usr/bin/env python3
"""sample_eval.py — fast, seeded, directly comparable scoring for solver variants.

The full four-dataset evaluation takes about 3.5 minutes at 8 workers, which is
still too slow to run inside every iteration of a variant search.  This samples a
FIXED subset of each dataset (deterministic stride over sorted filenames), so two
variants are always scored on exactly the same tasks and their numbers can be
compared line by line.

It is a *triage* tool, not a benchmark: it under-reports variants whose gain sits
on tasks outside the sample.  Anything that survives triage must be re-measured
on the full sets with `eval_arc.py` / `combined.py` before it is believed.  That
happened in practice — a gate change that triage scored as exactly neutral was
worth +2 and +4 tasks on the full training sets.

Usage:
    python3 sample_eval.py [module_dir] [workers]
    PIN=e0fb7511.json python3 sample_eval.py . 4

`PIN` force-includes named task files in every dataset that has them, so a
variant targeting one specific task can actually be measured on it.
"""
import json
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

MODDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sys.path.insert(0, MODDIR)

# (dataset directory, stride).  Strides keep each sample at 100-150 tasks.
SPEC = [
    ('/tmp/arc-agi/data/training', 4),       # 400  -> 100
    ('/tmp/ARC-AGI-2/data/training', 7),     # 1000 -> 143
    ('/tmp/arc-agi/data/evaluation', 4),     # 400  -> 100
    ('/tmp/ARC-AGI-2/data/evaluation', 1),   # 120  -> 120
]
PIN = [x.strip() for x in os.environ.get('PIN', '').split(',') if x.strip()]


def one(path):
    import arc
    task = json.load(open(path))
    try:
        out = arc.solve_task(task, max_depth=2, beam=16)
    except Exception as exc:                      # a crash must not kill the sweep
        return (os.path.basename(path), 'CRASH', repr(exc)[:120])
    if out is None:
        return (os.path.basename(path), 'none', None)
    ok = all(arc._tup(out[k]) == arc._tup(t['output'])
             for k, t in enumerate(task['test']))
    return (os.path.basename(path), 'ok' if ok else 'WRONG', None)


def sample(d, stride):
    files = sorted(glob.glob(os.path.join(d, '*.json')))[::stride]
    have = {os.path.basename(f) for f in files}
    for name in PIN:
        f = os.path.join(d, name)
        if name not in have and os.path.exists(f):
            files.append(f)
    return sorted(files)


if __name__ == '__main__':
    total_ok = total_n = 0
    for d, stride in SPEC:
        if not os.path.isdir(d):
            continue
        files = sample(d, stride)
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            res = list(pool.map(one, files, chunksize=2))
        good = [n for n, s, _ in res if s == 'ok']
        bad = [n for n, s, _ in res if s == 'WRONG']
        crash = [(n, e) for n, s, e in res if s == 'CRASH']
        total_ok += len(good)
        total_n += len(files)
        tag = os.path.basename(os.path.dirname(os.path.dirname(d))) + '/' + os.path.basename(d)
        print('%-32s solved=%3d/%3d  wrong=%d' % (tag, len(good), len(files), len(bad)))
        if bad:
            print('    wrong :', sorted(bad))
        if crash:
            print('    crash :', crash[:3])
        sys.stdout.flush()
    print('TOTAL solved=%d/%d (%.1f%%)' % (total_ok, total_n,
                                           100.0 * total_ok / max(total_n, 1)))
