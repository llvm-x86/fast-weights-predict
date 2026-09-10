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
#
# ---------------------------------------------------------------------------
# The ensemble of learned models
#
# The memory above keys on (position, colour) and so is blind to every rule that
# is *local* rather than positional.  Three further associative memories close
# most of that gap, all of them still plain counting with no LLM and no search:
#
#   ColorRemapModel   global colour co-occurrence C[in][out]  (global recolour)
#   LocalRuleModel    associative memory over local input patches -> output cell
#   (candidate list)  several LocalRuleModel geometries, see CANDIDATES below
#
# What changed relative to the first version of this file, and why:
#
#   * The 3x3 patch key used to be looked up exactly, and an unseen patch fell
#     back to the single nearest stored patch.  The key is now voted over the
#     k nearest stored patches with weight 1/(1+hamming), so an exact key still
#     dominates (distance 0) but near misses contribute smoothly.
#
#   * Border patches used to be padded with a sentinel (-1).  A sentinel makes
#     every border patch a unique key that shares nothing with the interior
#     statistics.  Reflecting the grid across the edge instead ("reflect" pad)
#     gives border patches real colours, which is the single largest gain here.
#
#   * A patch and its 8 dihedral transforms can share vote counts ("sym"), so a
#     rule learned in one orientation transfers to the others.  This is applied
#     only to patches fully inside the grid by default, because a padded patch's
#     transforms are not meaningful.
#
#   * Model selection used to minimise the summed held-out cell error alone.
#     A model that predicts "almost the input" scores well on that metric while
#     never being exactly right.  The score is now
#     (leave-one-example-out cell error) + (full-train cell error): the second
#     term is a training-fit check, in the same spirit as arc.py's "verify the
#     induced program on the training pairs before ranking it".
#
#   * Every model above is still *per-cell and same-size*: it can only say "the
#     output colour at (r,c) is a function of the input colours near (r,c)" on a
#     canvas of the input's dimensions, so 68% of ARC-AGI-1 and 74% of ARC-AGI-2
#     training tasks were simply skipped.  ObjectMemoryModel and ObjColorMapModel
#     add the other half of the substrate's own metaphor — the *items* in memory are whole objects, the
#     key is an object's canonical form (position- and optionally dihedral-
#     invariant), and the learned value is the object it becomes together with
#     its placement relative to the input object.  Those models can change the
#     canvas size; see OBJ_CANDIDATES and the section comment below.
#
# Selection remains entirely on the training pairs.  The test outputs are only
# ever compared with, never consulted.  A size-changing task is answered only
# when an object memory reproduces *every* training pair under leave-one-out,
# and otherwise abstains and falls through to the DSL.  On the two training sets
# the object memories add five/six tasks over the per-cell ensemble and two/three
# of those are tasks the DSL cannot do; on both evaluation sets (400 + 120 tasks)
# they change nothing at all.

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


class ColorRemapModel:
    """Hebbian color association: co-occurrence C[in_color][out_color], learned
    across every cell of every training pair, ignoring position.  This expresses
    global recolors (the most common single ARC transform) that the position-bound
    map cannot."""

    def __init__(self, H, W):
        self.H, self.W = H, W
        self.C = [[0] * 10 for _ in range(10)]

    def observe(self, x, y):
        for i in range(self.H):
            for j in range(self.W):
                self.C[x[i][j]][y[i][j]] += 1

    def predict(self, x):
        C = self.C
        return [[max(range(10), key=lambda c: C[x[i][j]][c])
                 for j in range(self.W)] for i in range(self.H)]


def _dihedral(g, n):
    """The distinct dihedral transforms of a square patch given row-major."""
    m = [g[i * n:(i + 1) * n] for i in range(n)]
    outs = []
    for k in range(4):
        r = m
        for _ in range(k):
            r = [[r[i][j] for i in range(n - 1, -1, -1)] for j in range(n)]
        outs.append(tuple(v for row in r for v in row))
        outs.append(tuple(v for row in [rw[::-1] for rw in r] for v in row))
    return list(dict.fromkeys(outs))


class LocalRuleModel:
    """Associative memory over local input neighbourhoods -> output cell colour.

    This is the patch-level generalization of the Hebbian map: instead of keying
    on (position, color), it keys on the local (2r+1)^2 pattern, so a rule learned
    at one location generalizes to every other location.  It expresses the *local*
    ARC family (cellular automata, hole/region fill, dilation, symmetry
    completion) that neither the position-bound map nor the global color map can.

    Parameters
      radius        r; the key is the (2r+1)^2 neighbourhood.  1 = the original 3x3.
      k             vote with the k nearest stored keys, weight 1/(1+hamming);
                    k=1 reproduces the original "nearest stored pattern" fallback.
      sym           also count/lookup the 8 dihedral transforms of the patch, so a
                    locally rotation/reflection-symmetric rule shares statistics.
      sym_interior  if True, only patches fully inside the grid are transformed
                    (a padded patch's transforms are not meaningful).
      pad           'sentinel' (-1 outside the grid — the original) or 'reflect'
                    (mirror the grid across the edge, so border patches hold real
                    colours and share statistics with the interior).
      prior         optional bias towards keeping the input colour (0 = off).

    Exact neighborhood matches dominate the vote; an unseen neighborhood falls
    back to its nearest stored patterns, then (with no memory at all) to identity.
    """

    def __init__(self, H, W, radius=1, k=1, sym=False, sym_interior=True,
                 pad='sentinel', prior=0.0):
        self.H, self.W = H, W
        self.radius, self.k = radius, k
        self.sym, self.sym_interior = sym, sym_interior
        self.pad, self.prior = pad, prior
        self.dict = {}  # patch key -> per-colour counts
        self._keys_arr = None
        self._counts = None

    def _patch(self, x, i, j):
        r, H, W = self.radius, self.H, self.W
        g, full = [], True
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    g.append(x[ni][nj])
                else:
                    full = False
                    if self.pad == 'reflect':
                        ri = -ni if ni < 0 else (2 * H - 2 - ni if ni >= H else ni)
                        rj = -nj if nj < 0 else (2 * W - 2 - nj if nj >= W else nj)
                        g.append(x[min(max(ri, 0), H - 1)][min(max(rj, 0), W - 1)])
                    else:
                        g.append(-1)
        return g, full

    def _keys(self, x, i, j):
        g, full = self._patch(x, i, j)
        if not self.sym:
            return [tuple(g)]
        if self.sym_interior and not full:
            return [tuple(g)]
        return _dihedral(g, 2 * self.radius + 1)

    def observe(self, x, y):
        for i in range(self.H):
            for j in range(self.W):
                for key in self._keys(x, i, j):
                    d = self.dict.get(key)
                    if d is None:
                        d = [0] * 10
                        self.dict[key] = d
                    d[y[i][j]] += 1
        self._keys_arr = None

    def _fit(self):
        if self._keys_arr is not None:
            return
        keys = sorted(self.dict)
        if keys:
            self._keys_arr = np.array(keys, dtype=np.int16)
            self._counts = np.array([self.dict[key] for key in keys], dtype=np.float64)
        else:
            self._keys_arr = np.zeros((0, 1), dtype=np.int16)
            self._counts = np.zeros((0, 10), dtype=np.float64)

    def predict(self, x):
        H, W = self.H, self.W
        self._fit()
        K, C = self._keys_arr, self._counts
        # one query per distinct patch (cheap, and the transform variants repeat a lot)
        index, queries, cells = {}, [], [[None] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                ks = self._keys(x, i, j)
                cells[i][j] = ks[0]
                for key in ks:
                    if key not in index:
                        index[key] = len(queries)
                        queries.append(key)
        votes = np.zeros((len(queries), 10), dtype=np.float64)
        if K.shape[0] and queries:
            Q = np.array(queries, dtype=np.int16)
            kk = min(self.k, K.shape[0])
            chunk = max(1, 300000 // max(1, K.shape[0]))
            for s in range(0, len(queries), chunk):
                q = Q[s:s + chunk]
                dist = (q[:, None, :] != K[None, :, :]).sum(axis=2)
                near = np.argsort(dist, axis=1)[:, :kk]
                rows = np.arange(q.shape[0])[:, None]
                w = 1.0 / (1.0 + dist[rows, near])
                for t in range(kk):
                    votes[s:s + q.shape[0]] += C[near[:, t]] * w[:, t:t + 1]
        out = [[0] * W for _ in range(H)]
        for i in range(H):
            for j in range(W):
                v = votes[index[cells[i][j]]].copy() if len(queries) else np.zeros(10)
                if self.prior:
                    v[x[i][j]] += self.prior
                if not v.any():                       # no memory at all: identity
                    out[i][j] = x[i][j]
                else:
                    out[i][j] = int(max(range(10), key=lambda c: (v[c], c == x[i][j], -c)))
        return out


# ---------------------------------------------------------------------------
# Object-level associative memory
#
# Every model above is *per-cell and same-size*: it can only say "the output
# colour at (r,c) is a function of the input colours near (r,c)" on a canvas of
# the input's dimensions.  Most of ARC is not like that; it is object-level and
# size-changing ("for each object produce this object", "the output is the odd
# object out", "stack the objects").  The memory below takes the other half of
# the substrate's own metaphor: the *items* stored are whole objects, the key is
# an object's canonical form, and what is learned is (a) which object an input
# object becomes and (b) where it lands relative to the input object.
#
# Exactly as before, nothing here is an LLM and nothing sees a test output.
# ---------------------------------------------------------------------------

_NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NB8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _mode_color(g):
    cnt = {}
    for row in g:
        for v in row:
            cnt[v] = cnt.get(v, 0) + 1
    return max(cnt, key=lambda c: (cnt[c], -c))


def _rot90(p):
    """Rotate a patch 90 degrees clockwise."""
    n, m = len(p), len(p[0])
    return tuple(tuple(p[n - 1 - j][i] for j in range(n)) for i in range(m))


def _flip(p):
    return tuple(tuple(row[::-1]) for row in p)


def _apply(p, tr):
    """Apply the dihedral element tr = (k, f): rotate k*90 CW, then flip."""
    k, f = tr
    for _ in range(k % 4):
        p = _rot90(p)
    if f:
        p = _flip(p)
    return p


def _rot_vec(v, k):
    """Rotate a displacement vector the same way _rot90 rotates a patch."""
    dr, dc = v
    for _ in range(k % 4):
        dr, dc = dc, -dr
    return (dr, dc)


def _vec_apply(v, tr):
    k, f = tr
    dr, dc = _rot_vec(v, k)
    return (dr, -dc) if f else (dr, dc)


def _vec_inv(v, tr):
    """Inverse of _vec_apply: invert flip, then invert the rotation."""
    k, f = tr
    dr, dc = v
    if f:
        dc = -dc
    return _rot_vec((dr, dc), -k)


def _canon(patch, sym):
    """Canonical form of a patch plus the transform that produced it.

    With sym=False the key is the exact tight subgrid, so the memory is
    translation-invariant only.  With sym=True the eight dihedral transforms
    share one key, so a rule learned on one orientation transfers to the others.
    """
    if not sym:
        return patch, (0, 0)
    best, btr = None, (0, 0)
    for k in range(4):
        r = patch
        for _ in range(k):
            r = _rot90(r)
        for f in (0, 1):
            q = _flip(r) if f else r
            if best is None or q < best:
                best, btr = q, (k, f)
    return best, btr


class _Obj:
    __slots__ = ('r0', 'c0', 'r1', 'c1', 'cells', 'patch', 'mask', 'colors')

    def __init__(self, r0, c0, r1, c1, cells, patch):
        self.r0, self.c0, self.r1, self.c1 = r0, c0, r1, c1
        self.cells, self.patch = cells, patch
        own = set(cells)
        self.mask = tuple(tuple(1 if (r0 + i, c0 + j) in own else 0
                                for j in range(c1 - c0 + 1))
                          for i in range(r1 - r0 + 1))
        self.colors = tuple(sorted(patch[r - r0][c - c0] for r, c in cells))


def _components(g, bg, conn):
    """4- or 8-connected components of the non-background cells, with their tight
    subgrids (position-invariant canonical forms of the objects themselves)."""
    H, W = len(g), len(g[0])
    nb = _NB4 if conn == 4 else _NB8
    seen = [[False] * W for _ in range(H)]
    objs = []
    for i in range(H):
        for j in range(W):
            if g[i][j] == bg or seen[i][j]:
                continue
            seen[i][j] = True
            stack, cells = [(i, j)], []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in nb:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and g[nr][nc] != bg:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            cells.sort()
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            patch = tuple(tuple(g[r][c0:c1 + 1]) for r in range(r0, r1 + 1))
            objs.append(_Obj(r0, c0, r1, c1, cells, patch))
    return objs


def _bbox_gap(a, b):
    """Chebyshev gap between two bounding boxes (0 when they touch/overlap)."""
    dr = max(a.r0 - b.r1, b.r0 - a.r1, 0)
    dc = max(a.c0 - b.c1, b.c0 - a.c1, 0)
    return max(dr, dc)


def _match_object(o, objs_out):
    """The output object an input object becomes, plus whether any real evidence
    supports the pairing.

    Evidence order: an identical patch (the object survives verbatim), then an
    identical shape mask (it is the same object recoloured or resized), then an
    identical colour multiset, then bounding-box proximity.  This is the honest
    "same position / same shape" relation the object memory learns on.  "No
    evidence" (every candidate differs in shape, colour and location) means the
    object does not appear in the output at all — acted on only when delete=True.
    """
    best, bkey = None, None
    for q in objs_out:
        ev = (q.patch == o.patch,
              q.mask == o.mask and len(q.patch) == len(o.patch)
              and len(q.patch[0]) == len(o.patch[0]),
              len(q.colors) == len(o.colors) and q.colors == o.colors,
              -_bbox_gap(o, q))
        if bkey is None or ev > bkey:
            best, bkey = q, ev
    if bkey is None:
        return None, False
    return best, bool(bkey[0] or bkey[1] or bkey[2] or bkey[3] == 0)


class ObjectMemoryModel:
    """Associative memory whose items are whole objects.

    observe(x, y)
      * decompose x into connected non-background objects;
      * key each object by its tight subgrid (canonical form, see _canon);
      * find the output object it becomes, by exact-patch / mask / overlap /
        proximity evidence (the same position-and-shape relation the sketch
        asks for), and remember (canonical output patch, relative placement).

    predict(x)
      * for every test object, look the key up and paint the remembered output
        patch at the remembered offset from the object; an unseen key is left
        exactly as it was (the model abstains on that object rather than
        inventing an answer).

    The canvas may be the input's own dimensions ('input'), a fixed shape
    learned from the training outputs ('output'), or the bounding box of what
    was painted ('content'), which is what lets the model change size.
    """

    def __init__(self, bg='zero', conn=4, sym=False, base='copy', canvas='input',
                 erase=False, keymode='patch', fallback=None, delete=False):
        self.bg_mode = bg
        self.conn = conn
        self.sym = sym
        self.base = base
        self.canvas = canvas
        self.erase = erase
        self.keymode = keymode
        self.fallback = fallback
        self.delete = delete
        self.table = {}          # key -> { (val_canon, delta_canon): count }
        self.out_shape = None
        self._arr = None         # lazily built (keys x pad) array for nearest lookup
        self._klist = None
        self._padw = 0
        self._padh = 0

    # -- helpers ---------------------------------------------------------
    def _bg(self, g):
        return 0 if self.bg_mode == 'zero' else _mode_color(g)

    def _key_of(self, o):
        if self.keymode == 'mask':
            return _canon(o.mask, self.sym)
        return _canon(o.patch, self.sym)

    # -- learning --------------------------------------------------------
    def observe(self, x, y):
        objs_in = _components(x, self._bg(x), self.conn)
        objs_out = _components(y, self._bg(y), self.conn)
        self.out_shape = (len(y), len(y[0]))
        if not objs_in:
            return
        for o in objs_in:
            key, tr = self._key_of(o)
            q, pos = (None, False) if not objs_out else _match_object(o, objs_out)
            if self.delete and not pos:
                d = self.table.setdefault(key, {})
                slot = (None, (0, 0))
                d[slot] = d.get(slot, 0) + 1
                self._arr = None
                continue
            if q is None:
                continue
            val, _ = _canon(q.patch, self.sym)
            delta = _vec_apply((q.r0 - o.r0, q.c0 - o.c0), tr)
            d = self.table.setdefault(key, {})
            slot = (val, delta)
            d[slot] = d.get(slot, 0) + 1
            self._arr = None

    def _fit_keys(self):
        """Pad every stored key into one array so an unseen object can fall back
        to the nearest remembered key (the same nearest-pattern rule the local
        patch memory uses)."""
        if self._arr is not None:
            return
        keys = sorted(self.table)
        if not keys:
            self._arr, self._klist = np.zeros((0, 0), dtype=np.int32), []
            self._padw = self._padh = 0
            return
        h = max(len(k) for k in keys)
        w = max(len(k[0]) for k in keys)
        a = np.full((len(keys), h * w), -1, dtype=np.int32)
        for i, k in enumerate(keys):
            for r, row in enumerate(k):
                for c, v in enumerate(row):
                    a[i, r * w + c] = v
        self._arr, self._klist, self._padw = a, keys, w
        self._padh = h

    def _lookup(self, key):
        d = self.table.get(key)
        if d or self.fallback != 'nearest':
            return d
        self._fit_keys()
        if self._arr.shape[0] == 0:
            return None
        w, h = self._padw, self._padh
        q = np.full(h * w, -1, dtype=np.int32)
        for r, row in enumerate(key):
            if r >= h:
                break
            for c, v in enumerate(row):
                if c < w:
                    q[r * w + c] = v
        dist = (self._arr != q[None, :]).sum(axis=1)
        return self.table[self._klist[int(dist.argmin())]]

    # -- prediction ------------------------------------------------------
    def predict(self, x):
        bg = self._bg(x)
        H, W = len(x), len(x[0])
        H2, W2 = H, W
        if self.canvas == 'output' and self.out_shape:
            H2, W2 = self.out_shape
        out = [row[:] for row in x] if self.base == 'copy' else [[bg] * W2 for _ in range(H2)]

        def put(r, c, v):
            if 0 <= r < H2 and 0 <= c < W2 and v != bg:
                out[r][c] = v

        def clear(o):
            for r, c in o.cells:
                if 0 <= r < H2 and 0 <= c < W2:
                    out[r][c] = bg

        painted = []
        for o in _components(x, bg, self.conn):
            key, tr = self._key_of(o)
            d = self._lookup(key)
            if not d:
                # unseen object and no fallback: leave it exactly as it was
                for r, c in o.cells:
                    if 0 <= r < H2 and 0 <= c < W2:
                        out[r][c] = x[r][c]
                continue
            (val, delta_canon), _ = max(d.items(), key=lambda kv: (kv[1], str(kv[0])))
            if val is None:                        # learned deletion
                clear(o)
                continue
            delta = _vec_inv(delta_canon, tr)
            patch = _apply(val, tr)
            pr, pc = o.r0 + delta[0], o.c0 + delta[1]
            if self.erase:                         # the object left its old cells behind
                clear(o)
            for i, row in enumerate(patch):
                for j, v in enumerate(row):
                    put(pr + i, pc + j, v)
            painted.append((pr, pc, pr + len(patch) - 1, pc + len(patch[0]) - 1))
        if self.canvas == 'content':
            if not painted:
                return [[bg]]
            r0 = min(max(0, min(p[0] for p in painted)), H2 - 1)
            c0 = min(max(0, min(p[1] for p in painted)), W2 - 1)
            r1 = min(max(r0, max(p[2] for p in painted)), H2 - 1)
            c1 = min(max(c0, max(p[3] for p in painted)), W2 - 1)
            out = [row[c0:c1 + 1] for row in out[r0:r1 + 1]]
        return out


class ObjColorMapModel:
    """Object-level memory whose key is the object's *shape* and whose value is
    the colour rewrite, not the recoloured pixels.

    The plain object memory keys on the exact tight subgrid, so it only
    transfers a rule to an object that reappears identically.  Half of the
    objects that fail that test nevertheless reappear with the same *shape* and
    a different palette, which is exactly ARC's "recolour every object of this
    shape" family.  This memory keeps the learned correspondence between the
    cells of the shape, so it can rewrite the colours of a shape it has never
    seen in that colour, and it remembers the relative placement alongside.
    """

    def __init__(self, bg='mode', conn=4, erase=False):
        self.bg_mode = bg
        self.conn = conn
        self.erase = erase
        self.table = {}          # (mask, delta) -> {in_colour: {out_colour: n}}

    def _bg(self, g):
        return 0 if self.bg_mode == 'zero' else _mode_color(g)

    def observe(self, x, y):
        bg = self._bg(x)
        objs_in = _components(x, bg, self.conn)
        objs_out = _components(y, self._bg(y), self.conn)
        if not objs_in or not objs_out:
            return
        for o in objs_in:
            q, pos = _match_object(o, objs_out)
            if q is None or not pos:
                continue
            if (len(q.patch), len(q.patch[0])) != (len(o.patch), len(o.patch[0])):
                continue
            if q.mask != o.mask:
                continue                      # a cell-wise colour map needs the shape
            delta = (q.r0 - o.r0, q.c0 - o.c0)
            d = self.table.setdefault((o.mask, delta), {})
            for r, c in o.cells:
                ic = x[r][c]
                oc = q.patch[r - o.r0][c - o.c0]
                g = d.setdefault(ic, {})
                g[oc] = g.get(oc, 0) + 1

    def predict(self, x):
        bg = self._bg(x)
        H, W = len(x), len(x[0])
        out = [row[:] for row in x]
        for o in _components(x, bg, self.conn):
            best, bkey = None, None
            for (mask, delta), d in self.table.items():
                if mask != o.mask:
                    continue
                n = sum(sum(g.values()) for g in d.values())
                k = (n, -abs(delta[0]) - abs(delta[1]))
                if bkey is None or k > bkey:
                    best, bkey = (delta, d), k
            if best is None:                  # unseen shape: leave the object alone
                continue
            delta, d = best
            if self.erase:
                for r, c in o.cells:
                    out[r][c] = bg
            for r, c in o.cells:
                g = d.get(x[r][c])
                if not g:
                    continue
                oc = max(g, key=lambda k: (g[k], -k))
                rr, cc = r + delta[0], c + delta[1]
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr][cc] = oc
        return out


# The candidate ensemble.  Each entry is a name and a factory.  They are distinct
# *rule families* — a position-bound memory, a global colour memory, and several
# geometries of the local patch memory — not a search over hyper-parameters, and
# nothing here is chosen per task; solve_task ranks them on the training pairs.
CANDIDATES = [
    ('hebb', lambda H, W: HebbianWorldModel(H, W)),
    ('color', lambda H, W: ColorRemapModel(H, W)),
    ('local3x3', lambda H, W: LocalRuleModel(H, W)),
    ('local3x3_sym', lambda H, W: LocalRuleModel(H, W, sym=True)),
    ('local3x3_sym_all', lambda H, W: LocalRuleModel(H, W, sym=True, sym_interior=False)),
    ('local3x3_reflect', lambda H, W: LocalRuleModel(H, W, pad='reflect')),
    ('local3x3_sym_reflect', lambda H, W: LocalRuleModel(H, W, sym=True, pad='reflect')),
]


# Object-level candidates.  They ignore the fixed (H, W) the same-size models
# need, because changing the size is the whole point; a size mismatch is charged
# as the area of the two grids put together, so a model that gets the shape
# wrong can never be selected over one that gets it right.
OBJ_CANDIDATES = [
    ('obj_rewrite', lambda H, W: ObjectMemoryModel(base='copy', canvas='input')),
    ('obj_rewrite_mode', lambda H, W: ObjectMemoryModel(bg='mode', base='copy',
                                                        canvas='input')),
    ('obj_rewrite8', lambda H, W: ObjectMemoryModel(bg='mode', conn=8, base='copy',
                                                    canvas='input')),
    ('obj_rewrite_sym', lambda H, W: ObjectMemoryModel(bg='mode', sym=True, base='copy',
                                                       canvas='input')),
    ('obj_rewrite_erase', lambda H, W: ObjectMemoryModel(bg='mode', base='copy',
                                                         canvas='input', erase=True)),
    ('obj_mask', lambda H, W: ObjectMemoryModel(bg='mode', base='copy', canvas='input',
                                                keymode='mask')),
    ('obj_near', lambda H, W: ObjectMemoryModel(bg='mode', base='copy', canvas='input',
                                                fallback='nearest')),
    ('obj_del', lambda H, W: ObjectMemoryModel(bg='mode', base='copy', canvas='input',
                                               delete=True)),
    ('obj_del_erase', lambda H, W: ObjectMemoryModel(bg='mode', base='copy', canvas='input',
                                                     delete=True, erase=True)),
    ('obj_blank', lambda H, W: ObjectMemoryModel(bg='mode', base='blank', canvas='input')),
    ('obj_content', lambda H, W: ObjectMemoryModel(bg='mode', base='blank', canvas='content')),
    ('obj_erase_content', lambda H, W: ObjectMemoryModel(bg='mode', base='copy',
                                                         canvas='content', erase=True)),
    ('obj_cmap', lambda H, W: ObjColorMapModel()),
    ('obj_cmap_erase', lambda H, W: ObjColorMapModel(erase=True)),
]


def _mismatch(model, e):
    """Mismatched output cells; a wrong canvas shape costs both areas."""
    p = model.predict(e['input'])
    o = e['output']
    if not p or not p[0] or len(p) != len(o) or len(p[0]) != len(o[0]):
        return len(o) * max(1, len(o[0])) + (len(p) * len(p[0]) if p and p[0] else 0)
    return sum(1 for i in range(len(o)) for j in range(len(o[0]))
               if p[i][j] != o[i][j])


def _score_parts(factory, train, H, W):
    """(held_out, fit) for one candidate, on the training pairs only.

    Held-out term:  for each training example, fit on the others and count the
    mismatched cells of its prediction.  This is the dreamer's honest check — a
    model that only memorizes is rejected even when it fits the training set.

    Fit term:  the mismatched cells when the model is fitted on all pairs.  The
    held-out term alone rewards a model that predicts "roughly the input" on
    every fold; requiring the model to also reproduce the pairs it saw is the
    same "verify before ranking" discipline arc.py applies to induced programs.
    """
    held_out = 0
    for k in range(len(train)):
        m = factory(H, W)
        for j, e in enumerate(train):
            if j != k:
                m.observe(e['input'], e['output'])
        held_out += _mismatch(m, train[k])
    full = factory(H, W)
    for e in train:
        full.observe(e['input'], e['output'])
    fit = sum(_mismatch(full, e) for e in train)
    return held_out, fit


def _selection_score(factory, train, H, W):
    ho, fit = _score_parts(factory, train, H, W)
    return ho + fit


def solve_task(task, eta=1.0, decay=0.0):
    """Learn each candidate world model, let the dreamer pick the one that
    generalizes best on the training pairs, then apply it to the test inputs.

    Two regimes:
      * every grid (train and test) is the same size — the original behaviour:
        the whole ensemble is ranked on the training pairs by held-out + fit
        cell error and the winner answers;
      * the sizes change — only the object-level memories are eligible, and one
        answers only when it reproduces *every* training pair exactly both when
        fitted on all pairs and when each pair is held out.  Otherwise the task
        is reported as unsolved (None) and falls through to the DSL rather than
        being answered by a map that has shown it does not generalize."""
    train = task['train']
    H, W = len(train[0]['output']), len(train[0]['output'][0])
    same_size = True
    for e in train + task['test']:
        if (len(e['input']), len(e['input'][0])) != (H, W):
            same_size = False
        if (len(e['output']), len(e['output'][0])) != (H, W):
            same_size = False

    if same_size:
        pool = CANDIDATES + OBJ_CANDIDATES
        best = min(pool, key=lambda c: (_selection_score(c[1], train, H, W),))[1]
    else:
        scored = [(c, _score_parts(c[1], train, H, W)) for c in OBJ_CANDIDATES]
        scored = [(c, ho, fit) for c, (ho, fit) in scored if ho == 0 and fit == 0]
        if not scored:
            return None
        best = scored[0][0][1]

    m = best(H, W)
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


def evaluate(data_dir, num_workers=2, eta=1.0, decay=0.0):
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
    print(f'{os.path.basename(data_dir)}: {len(solved)}/{sized} attempted tasks '
          f'({100.0 * len(solved) / max(1, sized):.1f}%); '
          f'{len(skipped)}/{n} abstained (size-changing, no memory generalized)')
    print('solved:', ' '.join(sorted(solved)) if solved else '(none)')
