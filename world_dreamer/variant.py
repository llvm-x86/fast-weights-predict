#!/usr/bin/env python3
"""variant.py — spawn isolated solver copies and score them on a fixed sample.

The workflow this exists for: when several independent ideas are worth trying,
give each one its own copy of the tree so they can be developed and measured
concurrently without touching the canonical solver, then keep only the winners.

    python3 variant.py spawn A_gate B_depth3 C_battery      # isolated copies
    python3 variant.py measure A_gate 4                      # fixed-sample score
    python3 variant.py measure A_gate 4 -- pin=e0fb7511.json  # include a task
    python3 variant.py diff A_gate                           # what it changed
    python3 variant.py list

Copies live under $WD_VARIANTS (default /tmp/wd_variants) and contain the whole
package, so a variant is measured with its own `arc.py`.  Scoring uses
`sample_eval.py`, which is a triage signal only: a variant that looks neutral or
positive there must be re-measured on the full sets with `eval_arc.py` /
`combined.py` before it is believed, because the sample can miss its gain
entirely.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('WD_VARIANTS', '/tmp/wd_variants')
FILES = ['arc.py', 'learned.py', 'combined.py', 'eval_arc.py', 'framework.py',
         'pursuit.py', 'sample_eval.py']


def spawn(names):
    os.makedirs(ROOT, exist_ok=True)
    for name in names:
        dst = os.path.join(ROOT, name)
        if os.path.exists(dst):
            print('exists, leaving alone:', dst)
            continue
        os.makedirs(dst)
        for f in FILES:
            src = os.path.join(HERE, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print('spawned', dst)


def measure(name, workers=4, pin=None):
    d = os.path.join(ROOT, name)
    if not os.path.isdir(d):
        raise SystemExit('no such variant: ' + d)
    env = dict(os.environ)
    if pin:
        env['PIN'] = pin
    subprocess.run([sys.executable, os.path.join(HERE, 'sample_eval.py'), d,
                    str(workers)], env=env, check=False)


def diff(name):
    d = os.path.join(ROOT, name)
    for f in FILES:
        a, b = os.path.join(HERE, f), os.path.join(d, f)
        if os.path.exists(a) and os.path.exists(b):
            r = subprocess.run(['diff', '-u', a, b], capture_output=True, text=True)
            if r.stdout:
                print('==== %s' % f)
                print(r.stdout)


def listing():
    if not os.path.isdir(ROOT):
        print('(no variants yet)')
        return
    for n in sorted(os.listdir(ROOT)):
        print(n, os.path.join(ROOT, n))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, rest = sys.argv[1], sys.argv[2:]
    pin = None
    rest = [a for a in rest if not (a.startswith('--pin=') and (pin := a[6:]) is not None)]
    if cmd == 'spawn':
        spawn(rest)
    elif cmd == 'measure':
        if not rest:
            raise SystemExit('measure needs a variant name')
        measure(rest[0], int(rest[1]) if len(rest) > 1 else 4, pin)
    elif cmd == 'diff':
        diff(rest[0])
    elif cmd == 'list':
        listing()
    else:
        raise SystemExit(__doc__)
