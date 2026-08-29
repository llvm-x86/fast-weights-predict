#!/usr/bin/env python3
# arc.py — the discrete half of the world dreamer (v2).
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
#
# v2 adds the two ARC priors that matter most beyond raw geometry: *objectness*
# (decompose a grid into connected components and transform/crop/recolor whole
# objects) and *completion* (gravity, mirror completion, connecting points).
# v3 adds a program-*composition* search (depth-2/3 over primitive compositions,
# deduplicated by intermediate grid, with cheap color-set/dimension guards).
# Every primitive is verified against all training examples before it is used on
# the held-out test, so a wrong rule is filtered out rather than guessed.

from collections import Counter, defaultdict, deque

# ---------------------------------------------------------------- grid helpers

def _tup(g):
    return tuple(tuple(row) for row in g)


def _bg(g):
    """Background color.  ARC-AGI-1 uses black (0) as background whenever it is
    present; fully-colored grids have no 0, so fall back to the most common color
    (the object/numerosity tasks with no background cells)."""
    cnt = Counter()
    for row in g:
        cnt.update(row)
    if 0 in cnt:
        return 0
    return cnt.most_common(1)[0][0]


def _colors(g):
    s = set()
    for row in g:
        s.update(row)
    return s


def _bbox_of_nonzero(g, bg=0):
    h, w = len(g), len(g[0])
    r0 = c0 = None
    r1 = c1 = -1
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                if r0 is None:
                    r0, c0 = r, c
                r1 = max(r1, r)
                c1 = max(c1, c)
    if r0 is None:
        return 0, 0, h - 1, w - 1
    return r0, c0, r1, c1


def _components(g, bg):
    """4-connected same-color components of the non-background cells.

    Returns a list of (color, cells) with cells as a list of (r, c)."""
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg and not seen[r][c]:
                color = g[r][c]
                cells = []
                stack = [(r, c)]
                seen[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and not seen[nr][nc]
                                and g[nr][nc] == color):
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                comps.append((color, cells))
    return comps


def _bresenham(r0, c0, r1, c1):
    pts = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        pts.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return pts


# ---------------------------------------------------------------- primitives
# Each primitive takes (grid, ...) and returns a new grid.  Parameters are
# enumerated by the search; the search then verifies survivors on every example.

def rotate(g, k):
    out = [list(row) for row in g]
    for _ in range(k % 4):
        out = [list(r) for r in zip(*out[::-1])]
    return out


def flip(g, axis):
    h, w = len(g), len(g[0])
    if axis == 'h':
        return [row[::-1] for row in g]
    if axis == 'v':
        return g[::-1]
    if axis == 'main':
        return [list(r) for r in zip(*g)]
    if axis == 'anti':
        # Reflect across the anti-diagonal.  Square-safe; for a non-square grid
        # this is the (h<->w) transposed reflection and simply fails verification
        # rather than crashing.
        return [[g[h - 1 - c][w - 1 - r] for c in range(h)] for r in range(w)]
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
    h, w = len(g), len(g[0])
    return [[g[r // k][c // k] for c in range(w * k)] for r in range(h * k)]


def tile(g, n, m):
    return [[g[r % len(g)][c % len(g[0])] for c in range(m * len(g[0]))]
            for r in range(n * len(g))]


def self_substitute(g):
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
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    seen = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for cc in (0, w - 1):
            if out[r][cc] == 0 and not seen[r][cc]:
                seen[r][cc] = True
                q.append((r, cc))
    for cc in range(w):
        for r in (0, h - 1):
            if out[r][cc] == 0 and not seen[r][cc]:
                seen[r][cc] = True
                q.append((r, cc))
    while q:
        r, cc = q.popleft()
        out[r][cc] = c
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, c2 = r + dr, cc + dc
            if 0 <= rr < h and 0 <= c2 < w and not seen[rr][c2] and out[rr][c2] == 0:
                seen[rr][c2] = True
                q.append((rr, c2))
    return out


def fill_holes(g, c):
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    border_reached = [[False] * w for _ in range(h)]
    q = deque()
    for r in range(h):
        for cc in (0, w - 1):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                border_reached[r][cc] = True
                q.append((r, cc))
    for cc in range(w):
        for r in (0, h - 1):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                border_reached[r][cc] = True
                q.append((r, cc))
    while q:
        r, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, c2 = r + dr, cc + dc
            if 0 <= rr < h and 0 <= c2 < w and not border_reached[rr][c2] and out[rr][c2] == 0:
                border_reached[rr][c2] = True
                q.append((rr, c2))
    for r in range(h):
        for cc in range(w):
            if out[r][cc] == 0 and not border_reached[r][cc]:
                out[r][cc] = c
    return out


# ---- v2: objectness and completion ----------------------------------------

def gravity(g, direction):
    """All non-background cells fall to one side of the grid, preserving column
    (for down/up) or row (for left/right) order, like sand."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    if direction == 'down':
        for c in range(w):
            col = [g[r][c] for r in range(h) if g[r][c] != bg]
            for i, v in enumerate(col):
                out[h - len(col) + i][c] = v
    elif direction == 'up':
        for c in range(w):
            col = [g[r][c] for r in range(h) if g[r][c] != bg]
            for i, v in enumerate(col):
                out[i][c] = v
    elif direction == 'right':
        for r in range(h):
            row = [g[r][c] for c in range(w) if g[r][c] != bg]
            for i, v in enumerate(row):
                out[r][w - len(row) + i] = v
    elif direction == 'left':
        for r in range(h):
            row = [g[r][c] for c in range(w) if g[r][c] != bg]
            for i, v in enumerate(row):
                out[r][i] = v
    return out


def mirror_union(g, axis):
    """Complete a partial mirror image: reflect the non-background cells across
    the central axis and union the reflection into the background cells."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    if axis == 'v':  # horizontal axis: top <-> bottom
        for r in range(h):
            for c in range(w):
                if out[r][c] == bg:
                    out[r][c] = g[h - 1 - r][c]
    elif axis == 'h':  # vertical axis: left <-> right
        for r in range(h):
            for c in range(w):
                if out[r][c] == bg:
                    out[r][c] = g[r][w - 1 - c]
    return out


def connect_points(g):
    """Draw straight lines connecting same-colored points that share a row or
    column (fill the span between the leftmost/rightmost or top/bottom ones)."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    rows = defaultdict(list)
    cols = defaultdict(list)
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v != bg:
                rows[(v, r)].append(c)
                cols[(v, c)].append(r)
    for (v, r), cs in rows.items():
        if len(cs) >= 2:
            for c in range(min(cs) + 1, max(cs)):
                out[r][c] = v
    for (v, c), rs in cols.items():
        if len(rs) >= 2:
            for r in range(min(rs) + 1, max(rs)):
                out[r][c] = v
    return out


def connect_diag(g):
    """For a color with exactly two cells that are NOT aligned, draw the diagonal
    (Bresenham) line between them."""
    bg = _bg(g)
    out = [list(row) for row in g]
    byc = defaultdict(list)
    for r in range(len(g)):
        for c in range(len(g[0])):
            if g[r][c] != bg:
                byc[g[r][c]].append((r, c))
    for color, cells in byc.items():
        if len(cells) == 2:
            (r0, c0), (r1, c1) = cells
            if r0 != r1 and c0 != c1:
                for r, c in _bresenham(r0, c0, r1, c1):
                    if out[r][c] == bg:
                        out[r][c] = color
    return out


def dilate(g):
    """Grow every non-background cell one step in the 4 directions."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == bg:
                        out[nr][nc] = g[r][c]
    return out


def crop_component(g, key):
    """Crop to a single object: 'largest' or 'smallest' non-background component.
    The output is the object alone (background elsewhere), at its bounding box."""
    bg = _bg(g)
    comps = _components(g, bg)
    if not comps:
        return [list(row) for row in g]
    comps.sort(key=lambda cc: len(cc[1]), reverse=(key == 'largest'))
    color, cells = comps[0]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    out = [[bg] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
    for r, c in cells:
        out[r - r0][c - c0] = color
    return out


def _infer_size_colors(in0, out0, bg):
    """Learn a component-size -> output-color map from one (in, out) example.
    Returns None if the mapping is ambiguous (different colors for one size)."""
    mapping = {}
    for color, cells in _components(in0, bg):
        s = len(cells)
        outcolors = set(out0[r][c] for r, c in cells)
        if len(outcolors) == 1:
            nc = outcolors.pop()
            if s in mapping and mapping[s] != nc:
                return None
            mapping[s] = nc
    return mapping or None


def recolor_by_size(g, mapping):
    bg = _bg(g)
    out = [list(row) for row in g]
    for color, cells in _components(g, bg):
        s = len(cells)
        if s in mapping:
            for r, c in cells:
                out[r][c] = mapping[s]
    return out


def keep_color(g, c):
    """Keep only cells of one color; everything else becomes background
    (extract all objects of a given color)."""
    bg = _bg(g)
    return [[v if v == c else bg for v in row] for row in g]


def remove_component(g, key):
    """Remove the 'largest' or 'smallest' non-background component (denoising:
    drop the single big shape, or drop the scattered small object)."""
    bg = _bg(g)
    comps = _components(g, bg)
    if not comps:
        return [list(row) for row in g]
    comps.sort(key=lambda cc: len(cc[1]), reverse=(key == 'largest'))
    cells = comps[0][1]
    out = [list(row) for row in g]
    for r, c in cells:
        out[r][c] = bg
    return out


def map_flip(g, axis):
    """Flip each object in place (reflect its bounding box across axis 'h' or
    'v').  The object-wise analog of a global flip — the key 'map over objects'
    ARC-AGI-2 family."""
    bg = _bg(g)
    out = [list(row) for row in g]
    for color, cells in _components(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        sub = [[bg] * (c1 - c0 + 1) for _ in range(r1 - r0 + 1)]
        for r, c in cells:
            sub[r - r0][c - c0] = color
        t = flip(sub, axis)
        for r in range(r1 - r0 + 1):
            for c in range(c1 - c0 + 1):
                out[r0 + r][c0 + c] = t[r][c]
    return out


def map_rotate(g, k):
    """Rotate each object in place by k quarter-turns.  Requires every object's
    bounding box to be square (otherwise an in-place rotation would change shape);
    non-square-object grids are returned unchanged and simply fail to match."""
    bg = _bg(g)
    boxes = []
    for color, cells in _components(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        if (r1 - r0) != (c1 - c0):
            return [list(row) for row in g]
        boxes.append((color, cells, r0, c0, r1, c1))
    out = [list(row) for row in g]
    for color, cells, r0, c0, r1, c1 in boxes:
        n = r1 - r0 + 1
        sub = [[bg] * n for _ in range(n)]
        for r, c in cells:
            sub[r - r0][c - c0] = color
        t = rotate(sub, k)
        for r in range(n):
            for c in range(n):
                out[r0 + r][c0 + c] = t[r][c]
    return out


def _period_of(seq):
    n = len(seq)
    for p in range(1, n + 1):
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def extend_rows(g, H_target):
    """Continue the vertical repeating row pattern to H_target rows (the
    sequence-extrapolation family)."""
    rows = [tuple(r) for r in g]
    p = _period_of(rows)
    return [list(rows[i % p]) for i in range(H_target)]


def extend_cols(g, W_target):
    """Continue the horizontal repeating column pattern to W_target columns."""
    h = len(g)
    cols = [tuple(g[r][c] for r in range(h)) for c in range(len(g[0]))]
    p = _period_of(cols)
    return [[cols[c % p][r] for c in range(W_target)] for r in range(h)]


# ---------------------------------------------------------------- program search

# A "program" is a closed-over function grid -> grid.  Search enumerates programs
# from the primitive library (and two-step "extract object, then transform"
# compositions) that map the first example input to its output, then verifies the
# survivors on every example.  This is the "dreamer": imagine each candidate
# program's output and keep the one that predicts all observations.

def _anchor(g, h, w, bg=0):
    for r in range(h):
        for c in range(w):
            if g[r][c] != bg:
                return r, c
    return None


def _enumerate_depth1(in0, out0, fast=False):
    """Enumerate every single primitive mapping in0 -> out0.

    With fast=True the two expensive parameter sweeps (translation, recolor) use
    a single closed-form candidate instead of a full scan; this is exact except
    for the rare case where a translation clips the anchor cell, so it is used
    only for the final step of depth-3 search, never for depth-1."""
    h, w = len(in0), len(in0[0])
    H, W = len(out0), len(out0[0])
    target = _tup(out0)
    cols = _colors(in0)
    bg = _bg(in0)
    cdiff = len(cols ^ _colors(out0))  # color-set distance (for cheap guards)

    # identity
    if (h, w) == (H, W) and _tup(in0) == target:
        yield ('identity', lambda g: [list(r) for r in g])

    # rotations / reflections (size-preserving; 'anti' requires square).
    # These permute cells, so they can only match when the color sets are equal.
    if (h, w) == (H, W) and cdiff == 0:
        for k in (1, 2, 3):
            if _tup(rotate(in0, k)) == target:
                yield ('rotate%d' % (90 * k), (lambda kk: lambda g: rotate(g, kk))(k))
        for ax in ('h', 'v', 'main'):
            if _tup(flip(in0, ax)) == target:
                yield ('flip_' + ax, (lambda a: lambda g: flip(g, a))(ax))
        if h == w and _tup(flip(in0, 'anti')) == target:
            yield ('flip_anti', lambda g: flip(g, 'anti'))

    # translation (size-preserving, zero-fill); also color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if fast:
            ai = _anchor(in0, h, w)
            ao = _anchor(out0, h, w)
            if ai is not None and ao is not None:
                dx, dy = ao[1] - ai[1], ao[0] - ai[0]
                if (dx, dy) != (0, 0) and _tup(translate(in0, dx, dy)) == target:
                    yield ('translate(%d,%d)' % (dx, dy),
                           (lambda dx_, dy_: lambda g: translate(g, dx_, dy_))(dx, dy))
        else:
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if (dx, dy) == (0, 0):
                        continue
                    if _tup(translate(in0, dx, dy)) == target:
                        yield ('translate(%d,%d)' % (dx, dy),
                               (lambda dx_, dy_: lambda g: translate(g, dx_, dy_))(dx, dy))

    # recolor (size-preserving): single color -> color, including background.
    # One swap changes the color set by at most two (one removed, one added).
    if (h, w) == (H, W) and cdiff <= 2:
        if fast:
            for c1 in cols:
                c2 = None
                bad = False
                for r in range(h):
                    for c in range(w):
                        if in0[r][c] == c1:
                            if c2 is None:
                                c2 = out0[r][c]
                            elif out0[r][c] != c2:
                                bad = True
                                break
                    if bad:
                        break
                if not bad and c2 is not None and c2 != c1:
                    if _tup(recolor(in0, c1, c2)) == target:
                        yield ('recolor(%d->%d)' % (c1, c2),
                               (lambda a, b: lambda g: recolor(g, a, b))(c1, c2))
        else:
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

    # sequence extrapolation: continue the repeating row/column pattern
    if w == W and _tup(extend_rows(in0, H)) == target:
        yield ('extend_rows', lambda g: extend_rows(g, H))
    if h == H and _tup(extend_cols(in0, W)) == target:
        yield ('extend_cols', lambda g: extend_cols(g, W))

    # crop to bounding box of non-background
    cr = crop_to_bbox(in0)
    if _tup(cr) == target:
        yield ('crop', lambda g: crop_to_bbox(g))

    # flood fills (size-preserving); each adds at most one color
    if (h, w) == (H, W) and cdiff <= 1:
        for c in range(10):
            if _tup(fill_from_border(in0, c)) == target:
                yield ('fill_border(%d)' % c, (lambda cc: lambda g: fill_from_border(g, cc))(c))
            if _tup(fill_holes(in0, c)) == target:
                yield ('fill_holes(%d)' % c, (lambda cc: lambda g: fill_holes(g, cc))(c))

    # gravity (size-preserving), all four directions — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for d in ('down', 'up', 'left', 'right'):
            if _tup(gravity(in0, d)) == target:
                yield ('gravity_' + d, (lambda dd: lambda g: gravity(g, dd))(d))

    # mirror completion (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for ax in ('h', 'v'):
            if _tup(mirror_union(in0, ax)) == target:
                yield ('mirror_' + ax, (lambda a: lambda g: mirror_union(g, a))(ax))

    # connect points (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if _tup(connect_points(in0)) == target:
            yield ('connect', lambda g: connect_points(g))
        if _tup(connect_diag(in0)) == target:
            yield ('connect_diag', lambda g: connect_diag(g))

    # dilation (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if _tup(dilate(in0)) == target:
            yield ('dilate', lambda g: dilate(g))

    # per-object map transforms (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for ax in ('h', 'v'):
            if _tup(map_flip(in0, ax)) == target:
                yield ('map_flip_' + ax, (lambda a: lambda g: map_flip(g, a))(ax))
        for k in (1, 2, 3):
            if _tup(map_rotate(in0, k)) == target:
                yield ('map_rotate%d' % (90 * k), (lambda kk: lambda g: map_rotate(g, kk))(k))

    # crop to a single object (largest / smallest)
    for key in ('largest', 'smallest'):
        co = crop_component(in0, key)
        if _tup(co) == target:
            yield ('crop_' + key, (lambda k: lambda g: crop_component(g, k))(key))

    # keep only one color (size-preserving)
    if (h, w) == (H, W):
        for c in cols:
            if c == bg:
                continue
            if _tup(keep_color(in0, c)) == target:
                yield ('keep_color(%d)' % c, (lambda cc: lambda g: keep_color(g, cc))(c))

    # remove the largest / smallest component (size-preserving denoising)
    if (h, w) == (H, W):
        for key in ('largest', 'smallest'):
            if _tup(remove_component(in0, key)) == target:
                yield ('remove_' + key, (lambda k: lambda g: remove_component(g, k))(key))

    # recolor whole objects by their size (numerosity -> color)
    if (h, w) == (H, W):
        mapping = _infer_size_colors(in0, out0, bg)
        if mapping and _tup(recolor_by_size(in0, mapping)) == target:
            yield ('recolor_by_size', (lambda m: lambda g: recolor_by_size(g, m))(mapping))


def _unconditional_transitions(g):
    """Size-preserving or shrinking primitives, with bounded parameters, used as
    intermediate steps in composed programs.  Grow-only primitives (scale, tile,
    self-substitution) are excluded because they commute with these transforms and
    are always reachable as the final, target-matched step instead."""
    cols = _colors(g)
    bg = _bg(g)
    for k in (1, 2, 3):
        yield ('rotate%d' % (90 * k), (lambda kk: lambda x: rotate(x, kk))(k))
    for ax in ('h', 'v', 'main', 'anti'):
        yield ('flip_' + ax, (lambda a: lambda x: flip(x, a))(ax))
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            if (dx, dy) == (0, 0):
                continue
            yield ('translate(%d,%d)' % (dx, dy),
                   (lambda a, b: lambda x: translate(x, a, b))(dx, dy))
    for c1 in cols:
        for c2 in range(10):
            if c2 != c1:
                yield ('recolor(%d->%d)' % (c1, c2),
                       (lambda a, b: lambda x: recolor(x, a, b))(c1, c2))
    yield ('crop', lambda x: crop_to_bbox(x))
    for c in range(10):
        yield ('fill_border(%d)' % c, (lambda cc: lambda x: fill_from_border(x, cc))(c))
        yield ('fill_holes(%d)' % c, (lambda cc: lambda x: fill_holes(x, cc))(c))
    for d in ('down', 'up', 'left', 'right'):
        yield ('gravity_' + d, (lambda dd: lambda x: gravity(x, dd))(d))
    for ax in ('h', 'v'):
        yield ('mirror_' + ax, (lambda a: lambda x: mirror_union(x, a))(ax))
    yield ('connect', lambda x: connect_points(x))
    yield ('connect_diag', lambda x: connect_diag(x))
    yield ('dilate', lambda x: dilate(x))
    for ax in ('h', 'v'):
        yield ('map_flip_' + ax, (lambda a: lambda x: map_flip(x, a))(ax))
    for k in (1, 2, 3):
        yield ('map_rotate%d' % (90 * k), (lambda kk: lambda x: map_rotate(x, kk))(k))
    for key in ('largest', 'smallest'):
        yield ('crop_' + key, (lambda kk: lambda x: crop_component(x, kk))(key))
    for c in cols:
        if c != bg:
            yield ('keep_color(%d)' % c, (lambda cc: lambda x: keep_color(x, cc))(c))
    for key in ('largest', 'smallest'):
        yield ('remove_' + key, (lambda kk: lambda x: remove_component(x, kk))(key))


def _compose(outer, inner):
    return lambda g: outer(inner(g))


def _verify(prog, train):
    return all(_tup(prog(t['input'])) == _tup(t['output']) for t in train)


def _size_compatible(g, out0):
    """Can g reach out0's dimensions in one primitive?"""
    h, w = len(g), len(g[0])
    H, W = len(out0), len(out0[0])
    if (h, w) == (H, W):
        return True
    if H % h == 0 and W % w == 0:
        return True  # tile / scale / self-substitute imply divisibility
    if w == W:
        return True  # extend_rows: any height, width fixed
    if h == H:
        return True  # extend_cols: any width, height fixed
    return False


def _closeness(g, out0):
    """Hamming distance to the target when top-left aligned and padded, plus a
    penalty for area mismatch.  Lower is closer; used only to order the beam."""
    H, W = len(out0), len(out0[0])
    h, w = len(g), len(g[0])
    if h > H or w > W:
        return h * w + 1_000_000
    mism = 0
    for r in range(h):
        for c in range(w):
            if g[r][c] != out0[r][c]:
                mism += 1
    mism += (H - h) * W + H * (W - w)
    return mism


def find_program(train, max_depth=2, beam=16):
    """Program search over depth-1..3 compositions.

    Depth-1 and the *final* step of every composition are target-matched, so
    size-changing primitives (scale, tile, self-substitution) stay reachable.
    Intermediate steps are the size-preserving/shrinking primitives above, dedup'd
    by the intermediate grid they produce.  Depth-3 expands only the `beam`
    closest intermediate grids (a search heuristic — the winner is still verified
    against every training example, so pruning can only miss, never mis-answer)."""
    in0, out0 = train[0]['input'], train[0]['output']
    H, W = len(out0), len(out0[0])

    # depth 1
    for name, prog in _enumerate_depth1(in0, out0):
        if _verify(prog, train):
            return prog

    # depth 2: f1 (unconditional) then f2 (target-matched)
    g1_list = []  # (closeness, f1) for the beam that feeds depth 3
    if max_depth < 2:
        return None
    seen_g1 = set()
    for f1name, f1 in _unconditional_transitions(in0):
        g1 = f1(in0)
        k1 = _tup(g1)
        if k1 in seen_g1:
            continue
        seen_g1.add(k1)
        g1_list.append((_closeness(g1, out0), f1))
        if not _size_compatible(g1, out0):
            continue
        for f2name, f2 in _enumerate_depth1(g1, out0):
            prog = _compose(f2, f1)
            if _verify(prog, train):
                return prog

    # depth 3: f1 then f2 then f3, from the beam of closest g1
    if max_depth >= 3:
        g1_list.sort(key=lambda t: t[0])
        seen_g2 = set()  # global dedup: many (f1, f2) pairs reach the same grid
        for _, f1 in g1_list[:beam]:
            g1 = f1(in0)
            for f2name, f2 in _unconditional_transitions(g1):
                g2 = f2(g1)
                if len(g2) > H or len(g2[0]) > W:
                    continue
                k2 = _tup(g2)
                if k2 in seen_g2:
                    continue
                seen_g2.add(k2)
                # cheap pre-filters before the expensive target-matched enumeration
                if len(_colors(g2) ^ _colors(out0)) > 3:
                    continue
                if not _size_compatible(g2, out0):
                    continue
                for f3name, f3 in _enumerate_depth1(g2, out0, fast=True):
                    prog = _compose(f3, _compose(f2, f1))
                    if _verify(prog, train):
                        return prog
    return None


def solve_task(task, max_depth=2, beam=16):
    prog = find_program(task['train'], max_depth=max_depth, beam=beam)
    if prog is None:
        return None
    return [prog(t['input']) for t in task['test']]
