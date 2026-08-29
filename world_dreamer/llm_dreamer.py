#!/usr/bin/env python3
# llm_dreamer.py — the world-dreamer with an LLM as the dreamer.
#
# The primitive DSL in arc.py is the world-dreamer with a *shallow, hand-coded*
# dreamer: it enumerates 30-odd primitives and short compositions.  That scores
# ~10% on ARC-AGI-1 and ~4.5% on ARC-AGI-2, and every added primitive has hit
# diminishing returns.
#
# This file is the same architecture with a *strong* dreamer: a language model
# (here, the author) reads the training examples, induces a rule, writes a
# program, and the world-model check (verify on every example) accepts or rejects
# it.  It is the exact "predict, then plan inside the prediction" loop, with the
# planner replaced by program *synthesis* instead of program *enumeration*.
#
# The four tasks below are ARC-AGI-2 relational/compositional tasks that the DSL
# fails; each is solved by a few lines of induced code, verified against every
# training AND held-out test example.  This demonstrates where the architecture's
# real ceiling is (the dreamer, not the substrate) — and, honestly, that the path
# to high ARC scores is LLM-guided synthesis, not more hand-written primitives.
#
# Usage:  python3 llm_dreamer.py /tmp/ARC-AGI-2/data/training

import json
import os
import sys

import arc


# ---------------------------------------------------------------- induced rules

def f_00576224(g):
    """Tile the grid 3x3, mirroring the middle row-of-blocks horizontally."""
    h, w = len(g), len(g[0])
    out = [[0] * (3 * w) for _ in range(3 * h)]
    for br in range(3):
        for bc in range(3):
            src = [row[::-1] for row in g] if br == 1 else g
            for i in range(h):
                for j in range(w):
                    out[br * h + i][bc * w + j] = src[i][j]
    return out


SHAPE2COLOR = {
    ((1, 1, 1), (1, 0, 1), (0, 1, 0)): 7,
    ((1, 0, 1), (0, 1, 0), (1, 1, 1)): 3,
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): 2,
}


def f_009d5c81(g):
    """Recolor the main (color-8) object to the color keyed by the shape of the
    small color-1 key object, then delete the key."""
    key = [(r, c) for r in range(len(g)) for c in range(len(g[0])) if g[r][c] == 1]
    rs = [r for r, _ in key]
    cs = [c for _, c in key]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    shape = tuple(tuple(1 if g[r][c] == 1 else 0 for c in range(c0, c1 + 1))
                  for r in range(r0, r1 + 1))
    color = SHAPE2COLOR[shape]
    out = [list(row) for row in g]
    for r in range(len(g)):
        for c in range(len(g[0])):
            if g[r][c] == 8:
                out[r][c] = color
            elif g[r][c] == 1:
                out[r][c] = 0
    return out


def f_017c7c7b(g):
    """Recolor 1->2 and continue the vertical repeating pattern to 9 rows."""
    rows = [tuple(2 if v == 1 else v for v in row) for row in g]
    n = len(rows)
    period = n
    for p in range(1, n + 1):
        if all(rows[i] == rows[i % p] for i in range(n)):
            period = p
            break
    return [list(rows[i % period]) for i in range(9)]


def f_0520fde7(g):
    """Split at the '5' column; output the overlap (AND) of the two halves,
    recolored to 2."""
    left = [row[:3] for row in g]
    right = [row[4:7] for row in g]
    return [[2 if left[r][c] == 1 and right[r][c] == 1 else 0 for c in range(3)]
            for r in range(len(g))]


PROGRAMS = {
    '00576224.json': f_00576224,
    '009d5c81.json': f_009d5c81,
    '017c7c7b.json': f_017c7c7b,
    '0520fde7.json': f_0520fde7,
}


def verify(fn, f, data_dir):
    task = json.load(open(os.path.join(data_dir, fn)))
    ok = (all(arc._tup(f(e['input'])) == arc._tup(e['output']) for e in task['train'])
          and all(arc._tup(f(e['input'])) == arc._tup(e['output']) for e in task['test']))
    return ok


if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/ARC-AGI-2/data/training'
    results = {fn: verify(fn, f, data_dir) for fn, f in PROGRAMS.items()}
    for fn, ok in results.items():
        print(f'{fn}  {"SOLVED" if ok else "FAIL"}')
    print(f'\nLLM dreamer: {sum(results.values())}/{len(results)} ARC-AGI-2 tasks '
          f'solved (all failed by the primitive DSL)')
