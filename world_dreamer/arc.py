#!/usr/bin/env python3
# arc.py — the discrete half of the world dreamer.
#
# A "world dreamer" is a world model plus a planner that optimizes *inside* the
# model.  In the continuous pursuit instantiation the world model is the BDH
# fast-weight memory (state -> next velocity) and the dreamer is receding-horizon
# shooting over aim headings.  In the discrete ARC instantiation the roles map as
# follows:
#
#   world model  = the induced program  (a map from input grid to output grid)
#   dreamer      = program search: hypothesize candidate programs, *imagine*
#                  their output on every example, and keep the one that predicts
#                  all observed (input, output) pairs
#   act          = run the induced program on the held-out test input
#
# The "world model" for ARC has no time axis — the transition is the single-shot
# input -> output function — so "imagination" is applying a candidate program and
# checking it against the examples.  This is the symbolic-world-modeling recipe
# that ARC-AGI-3's frontier is converging on, instantiated in miniature.

from collections import deque

# ---------------------------------------------------------------- grid helpers

def _tup(g):
    return tuple(tuple(row) for row in g)


def _colors(g):
    s = set()
    for row in g:
        s.update(row)
    return s


def _bbox_of_nonzero(g):
    h, w = len(g), len(g[0])
    r0 = c0 = None
    r1 = c1 = -1
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                if r0 is None:
                    r0, c0 = r, c
                r1 = max(r1, r)
                c1 = max(c1, c)
    if r0 is None:
        return 0, 0, h - 1, w - 1
    return r0, c0, r1, c1


# ---------------------------------------------------------------- primitives
# Each primitive takes (grid, param) and returns a new grid (or None if the
# parameter is out of range / the output size would not match).

def rotate(g, k):
    # k in {0,1,2,3} clockwise quarter turns
    out = [list(row) for row in g]
    for _ in range(k % 4):
        out = [list(r) for r in zip(*out[::-1])]
    return out


def flip(g, axis):
    # axis in {'h','v','main','anti'}
    h, w = len(g), len(g[0])
    if axis == 'h':
        return [row[::-1] for row in g]
    if axis == 'v':
        return g[::-1]
    if axis == 'main':
        return [list(r) for r in zip(*g)]
    if axis == 'anti':
        # reflect across the anti-diagonal (square grids only): out[r][c] = g[n-1-c][n-1-r]
        n = h
        return [[g[n - 1 - c][n - 1 - r] for c in range(n)] for r in range(n)]
    raise ValueError(axis)


def translate(g, dx, dy, pad=0):
    h, w = len(g), len(g[0])
    out = [[pad] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            rr, cc = r + dy, c + dx
            if 0 <= rr < h and 0 <= cc < w:
                out[rr][cc] = g[r][c]
    return out


def scale(g, k):
    # each cell -> k x k block
    h, w = len(g), len(g[0])
    return [[g[r // k][c // k] for c in range(w * k)] for r in range(h * k)]


def tile(g, n, m):
    # repeat the whole grid n times vertically, m times horizontally
    return [[g[r % len(g)][c % len(g[0])] for c in range(m * len(g[0]))]
            for r in range(n * len(g))]


def self_substitute(g):
    # each foreground cell -> a copy of g; each background cell -> a zero block of
    # the same shape as g (fractal / substitution tiling).  Output is (h*h) x (w*w).
    h, w = len(g), len(g[0])
    out = [[0] * (w * w) for _ in range(h * h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] != 0:
                for i in range(h):
                    for j in range(w):
                        out[r * h + i][c * w + j] = g[i][j]
    return out


def recolor(g, c1, c2):
    return [[c2 if v == c1 else v for v in row] for row in g]


def crop_to_bbox(g):
    r0, c0, r1, c1 = _bbox_of_nonzero(g)
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


def fill_from_border(g, c):
    # flood-fill every background cell connected to the border with color c
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    seen = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for cc in (0, w - 1):
            if out[r][cc] == 0 and not seen[r][cc]:
                seen[r][cc] = True; q.append((r, cc))
    for cc in range(w):
        for r in (0, h - 1):
            if out[r][cc] == 0 and not seen[r][cc]:
                seen[r][cc] = True; q.append((r, cc))
    while q:
        r, cc = q.popleft()
        out[r][cc] = c
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, c2 = r + dr, cc + dc
            if 0 <= rr < h and 0 <= c2 < w and not seen[rr][c2] and out[rr][c2] == 0:
                seen[rr][c2] = True; q.append((rr, c2))
    return out


def fill_holes(g, c):
    # flood-fill every background cell NOT connected to the border (enclosed 0s)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    border_reached = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for cc in (0, w - 1):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                border_reached[r][cc] = True; q.append((r, cc))
    for cc in range(w):
        for r in (0, h - 1):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                border_reached[r][cc] = True; q.append((r, cc))
    while q:
        r, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, c2 = r + dr, cc + dc
            if 0 <= rr < h and 0 <= c2 < w and not border_reached[rr][c2] and out[rr][c2] == 0:
                border_reached[rr][c2] = True; q.append((rr, c2))
    for r in range(h):
        for cc in range(w):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                out[r][cc] = c
    return out


# ---------------------------------------------------------------- program search

# A "program" is a closed-over function grid -> grid.  Search enumerates programs
# from the primitive library (and a small set of two-step compositions) that map
# the first example input to its output, then verifies the survivors on every
# example.  This is the "dreamer": imagine each candidate program's output and
# keep the one that predicts all observations.

def _enumerate_depth1(in0, out0):
    h, w = len(in0), len(in0[0])
    H, W = len(out0), len(out0[0])
    target = _tup(out0)
    cols = _colors(in0)

    # identity
    if (h, w) == (H, W) and _tup(in0) == target:
        yield ('identity', lambda g: [list(r) for r in g])

    # rotations / reflections (size-preserving; 'anti' requires square)
    if (h, w) == (H, W):
        for k in (1, 2, 3):
            if _tup(rotate(in0, k)) == target:
                yield ('rotate%d' % (90 * k), (lambda kk: lambda g: rotate(g, kk))(k))
        for ax in ('h', 'v', 'main'):
            if _tup(flip(in0, ax)) == target:
                yield ('flip_' + ax, (lambda a: lambda g: flip(g, a))(ax))
        if h == w and _tup(flip(in0, 'anti')) == target:
            yield ('flip_anti', lambda g: flip(g, 'anti'))

    # translation (size-preserving, zero-fill); small window is the common case
    if (h, w) == (H, W):
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                if (dx, dy) == (0, 0):
                    continue
                if _tup(translate(in0, dx, dy)) == target:
                    yield ('translate(%d,%d)' % (dx, dy),
                           (lambda dx_, dy_: lambda g: translate(g, dx_, dy_))(dx, dy))

    # recolor (size-preserving): single color -> color, including background
    if (h, w) == (H, W):
        for c1 in cols:
            for c2 in range(10):
                if c2 == c1:
                    continue
                if _tup(recolor(in0, c1, c2)) == target:
                    yield ('recolor(%d->%d)' % (c1, c2),
                           (lambda a, b: lambda g: recolor(g, a, b))(c1, c2))

    # scale (each cell -> k x k block)
    if H % h == 0 and W % w == 0 and H // h == W // w:
        k = H // h
        if k >= 1 and _tup(scale(in0, k)) == target:
            yield ('scale(%d)' % k, (lambda kk: lambda g: scale(g, kk))(k))

    # tile (repeat whole grid n x m)
    if H % h == 0 and W % w == 0:
        n, m = H // h, W // w
        if (n, m) != (1, 1) and _tup(tile(in0, n, m)) == target:
            yield ('tile(%dx%d)' % (n, m), (lambda nn, mm: lambda g: tile(g, nn, mm))(n, m))

    # self-substitution tiling (each fg cell -> the grid)
    if H == h * h and W == w * w:
        if _tup(self_substitute(in0)) == target:
            yield ('self_substitute', lambda g: self_substitute(g))

    # crop to bounding box of non-background
    cr = crop_to_bbox(in0)
    if _tup(cr) == target:
        yield ('crop', lambda g: crop_to_bbox(g))

    # flood fills (size-preserving)
    if (h, w) == (H, W):
        for c in range(10):
            if _tup(fill_from_border(in0, c)) == target:
                yield ('fill_border(%d)' % c, (lambda cc: lambda g: fill_from_border(g, cc))(c))
            if _tup(fill_holes(in0, c)) == target:
                yield ('fill_holes(%d)' % c, (lambda cc: lambda g: fill_holes(g, cc))(c))


def _enumerate_depth2(in0, out0):
    # "extract the object, then transform it": crop to the non-background bbox,
    # then apply any depth-1 primitive to the cropped object.
    obj = crop_to_bbox(in0)
    for name, prog in _enumerate_depth1(obj, out0):
        yield ('crop->' + name, lambda g, p=prog: p(crop_to_bbox(g)))


def find_program(train):
    """Return a program function consistent with all training examples, or None."""
    in0, out0 = train[0]['input'], train[0]['output']
    seen = set()
    for name, prog in list(_enumerate_depth1(in0, out0)) + list(_enumerate_depth2(in0, out0)):
        if name in seen:
            continue
        seen.add(name)
        if all(_tup(prog(t['input'])) == _tup(t['output']) for t in train):
            return prog
    return None


def solve_task(task):
    prog = find_program(task['train'])
    if prog is None:
        return None
    return [prog(t['input']) for t in task['test']]
