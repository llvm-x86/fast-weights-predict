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
from functools import lru_cache

# ---- W5_objectmap2 experiment toggles -------------------------------------
# Three independent widenings of the generic per-object operator, each measured
# on its own before being combined.  All three default to off, i.e. the shipped
# baseline behaviour.
_OMAP_CONN8 = True     # also sweep 8-connected objectness (corner-touching objects)
_OMAP_BUILD = True     # add construction inners: paint inside the object's own box
_OMAP_COMPOSE = False  # add length-2 inner compositions from a small pool

# ---------------------------------------------------------------- grid helpers

def _eq(g, target):
    """True when grid `g` equals the tuple-of-tuples `target`.

    `_eq(x, target)` builds a whole new nested tuple before comparing; for the
    hundreds of primitive candidates that do *not* match, that allocation is pure
    waste.  This compares row by row and stops at the first difference, which for
    a non-matching candidate is usually row 0."""
    if len(g) != len(target):
        return False
    for i, row in enumerate(g):
        if tuple(row) != target[i]:
            return False
    return True


def _tup(g):
    return tuple(tuple(row) for row in g)


def _bg(g):
    """Background color.  ARC-AGI-1 uses black (0) as background whenever it is
    present; fully-colored grids have no 0, so fall back to the most common color
    (the object/numerosity tasks with no background cells).

    The 0 fast path scans rows and returns on the first hit, which is what almost
    every ARC grid takes; the no-0 branch counts into a small dict while preserving
    first-seen order so the tie-break matches `Counter.most_common`."""
    for row in g:
        for v in row:
            if v == 0:
                return 0
    cnt = {}
    order = []
    for row in g:
        for v in row:
            n = cnt.get(v)
            if n is None:
                cnt[v] = 1
                order.append(v)
            else:
                cnt[v] = n + 1
    if not order:
        return 0
    best = order[0]
    bc = cnt[best]
    for v in order:
        if cnt[v] > bc:
            bc = cnt[v]
            best = v
    return best


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
                if r0 is None or r < r0:
                    r0 = r
                if c0 is None or c < c0:
                    c0 = c
                if r > r1:
                    r1 = r
                if c > c1:
                    c1 = c
    if r0 is None:
        return 0, 0, h - 1, w - 1
    return r0, c0, r1, c1


def _key(g):
    """Hashable snapshot of a grid (tuples hash by content, so this is the cache
    key for every derived quantity)."""
    return tuple(map(tuple, g))


@lru_cache(maxsize=4096)
def _components_cached(key, bg):
    """Cached worker for `_components`; `key` is a tuple-of-tuples grid."""
    g = key
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
                comps.append((color, tuple(cells)))
    return tuple(comps)


def _components(g, bg):
    """4-connected same-color components of the non-background cells.

    Returns a list of (color, cells) with cells as a list of (r, c).  The
    decomposition is memoised: one `_enumerate_depth1` sweep asks for the same
    grid's components many times over (numerosity, crop, denoise, recolour), and
    the cache is what makes that sweep affordable.  A fresh outer list is returned
    each call so callers may sort or filter it freely."""
    return [(color, cells) for color, cells in _components_cached(_key(g), bg)]


def _components8(g, bg):
    """8-connected same-color components (diagonal neighbours count as joined).
    Needed by object-bbox constructions whose shapes touch only at a corner."""
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
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
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


def _translate_eq(g, target, dx, dy):
    """Does translating `g` by (dx, dy) with zero fill reproduce `target`?

    Equivalent to `_eq(translate(g, dx, dy), target)` but compares cell by cell
    and returns at the first mismatch instead of materialising a whole grid.  This
    is the inner loop of the +/-6 translation sweep, where almost every candidate
    fails on its first few cells."""
    h, w = len(g), len(g[0])
    for r in range(h):
        trow = target[r]
        for c in range(w):
            sr, sc = r - dy, c - dx
            v = g[sr][sc] if (0 <= sr < h and 0 <= sc < w) else 0
            if trow[c] != v:
                return False
    return True


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


def gravity_color(g, color, direction):
    """Gravity applied to cells of ONE color only; every other color stays put.
    This expresses the 'sand/water' family where one color sinks through another
    (e.g. the 1s fall to the bottom of each column past the stationary 5s)."""
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            if out[r][c] == color:
                out[r][c] = 0
    if direction == 'down':
        for c in range(w):
            n = sum(1 for r in range(h) if g[r][c] == color)
            for i in range(n):
                out[h - n + i][c] = color
    elif direction == 'up':
        for c in range(w):
            n = sum(1 for r in range(h) if g[r][c] == color)
            for i in range(n):
                out[i][c] = color
    elif direction == 'right':
        for r in range(h):
            n = sum(1 for c in range(w) if g[r][c] == color)
            for i in range(n):
                out[r][w - n + i] = color
    elif direction == 'left':
        for r in range(h):
            n = sum(1 for c in range(w) if g[r][c] == color)
            for i in range(n):
                out[r][i] = color
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


def connect_newcolor(g, L):
    """Draw straight lines of color L between same-colored points that share a
    row or column (the 'draw with a *new* color' variant of connect_points)."""
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
                if out[r][c] == bg:
                    out[r][c] = L
    for (v, c), rs in cols.items():
        if len(rs) >= 2:
            for r in range(min(rs) + 1, max(rs)):
                if out[r][c] == bg:
                    out[r][c] = L
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
    comps = sorted(comps, key=lambda cc: len(cc[1]), reverse=(key == 'largest'))
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
    comps = sorted(comps, key=lambda cc: len(cc[1]), reverse=(key == 'largest'))
    cells = comps[0][1]
    out = [list(row) for row in g]
    for r, c in cells:
        out[r][c] = bg
    return out


def map_flip(g, axis):
    """Flip each object in place (reflect its bounding box across axis 'h', 'v',
    'main' or 'anti').  The object-wise analog of a global flip — the key 'map
    over objects' ARC-AGI-2 family.  'main'/'anti' transpose the box, so they
    require a square bounding box; a non-square object leaves the grid unchanged
    and simply fails verification rather than crashing."""
    bg = _bg(g)
    out = [list(row) for row in g]
    for color, cells in _components(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        if axis in ('main', 'anti') and (r1 - r0) != (c1 - c0):
            return [list(row) for row in g]
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


def map_objects(g, f, conn=4):
    """Apply a primitive `f` to every object's own bounding-box subgrid and write
    the transformed subgrid back in place — the *generic* form of ARC's largest
    family, 'apply this transformation to every shape'.  `map_flip`/`map_rotate`
    are the hand-written special cases of this operator; opening it up lets the
    search reach per-object gravity, per-object mirror/point completion, per-object
    cross/diagonal carving and dilation, none of which was expressible before.

    The inner primitive must be size-preserving on the subgrid, which is what makes
    the in-place write-back legal.  A subgrid whose transformed size differs — and
    any empty or 1x1 grid — leaves the whole grid unchanged and simply fails
    verification rather than raising, the same convention as `map_flip`/`map_rotate`
    (a non-square object simply cannot be transposed in place).  The subgrid is
    filled with the grid background colour and only this object's cells, exactly as
    `map_flip` builds it."""
    if not g or not g[0] or (len(g) == 1 and len(g[0]) == 1):
        return [list(row) for row in g]
    bg = _bg(g)
    comps = _components(g, bg) if conn == 4 else _components8(g, bg)
    if not comps:
        return [list(row) for row in g]
    out = [list(row) for row in g]
    for color, cells in comps:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        sub = [[bg] * bw for _ in range(bh)]
        for r, c in cells:
            sub[r - r0][c - c0] = color
        t = f(sub, color, bg)
        if not t or not t[0] or len(t) != bh or len(t[0]) != bw:
            return [list(row) for row in g]
        for r in range(bh):
            orow = out[r0 + r]
            trow = t[r]
            for c in range(bw):
                orow[c0 + c] = trow[c]
    return out


def _object_boxes(g, conn):
    """One shared extraction: background scan, component decomposition and one
    tight subgrid per object — the part every inner of one `map_objects` sweep
    reuses.  Returns `(bg, boxes)` with boxes a list of
    `(r0, c0, bh, bw, sub, color)`, or None when there is nothing to transform
    (an empty grid, a 1x1 grid, or no components at all).

    `sub` is the object's bounding box, filled with the grid background and only
    this object's cells, exactly as `map_flip` builds it; `color` is the
    component's colour (a component is monochromatic, so it is exact) and `bg` the
    grid background, both of which the construction inners need in order to paint
    inside the box without having to re-derive them from `sub`."""
    if not g or not g[0] or (len(g) == 1 and len(g[0]) == 1):
        return None
    bg = _bg(g)
    comps = _components(g, bg) if conn == 4 else _components8(g, bg)
    if not comps:
        return None
    boxes = []
    for color, cells in comps:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        sub = [[bg] * bw for _ in range(bh)]
        for r, c in cells:
            sub[r - r0][c - c0] = color
        boxes.append((r0, c0, bh, bw, sub, color))
    return bg, boxes


def _map_objects_batch(g, inners, conn=4, boxes=None):
    """Batched form of `map_objects`: run every inner in `inners` over ONE shared
    extraction (one background scan, one component decomposition, one subgrid per
    object) and return [(name, f, transformed_grid), ...].

    `_enumerate_depth1` is called once per intermediate grid of the composed
    search, so it is on the hot path; sharing the extraction there is what keeps
    the generic operator affordable.  The result is exactly what calling
    `map_objects(g, f)` per inner would give, including the 'a subgrid whose size
    changed leaves the whole grid unchanged' rule.  `boxes` may be a precomputed
    `_object_boxes` result for this grid and `conn`, which the caller has usually
    already computed to decide whether the sweep is worth running at all."""
    base = [list(row) for row in g]
    ext = _object_boxes(g, conn) if boxes is None else boxes
    if ext is None:
        return [(n, f, [list(r) for r in base]) for n, f in inners]
    bg, boxes = ext
    results = []
    for name, f in inners:
        out = None
        for r0, c0, bh, bw, sub, color in boxes:
            t = f(sub, color, bg)
            if not t or not t[0] or len(t) != bh or len(t[0]) != bw:
                out = None  # a box changed size: the whole grid is unchanged
                break
            if out is None:
                out = [list(row) for row in base]
            for r in range(bh):
                orow = out[r0 + r]
                trow = t[r]
                for c in range(bw):
                    orow[c0 + c] = trow[c]
        if out is None:
            out = [list(r) for r in base]
        results.append((name, f, out))
    return results


def _omap_conventions(g):
    """The objectness conventions `map_objects` should be swept under for `g`.

    Yields `(boxes, conn)` starting with the 4-connected extraction.  The
    8-connected extraction is added only when it actually partitions the grid
    differently from the 4-connected one — i.e. when some object touches another
    only at a corner.  When the two agree, every conn=8 sweep would reproduce the
    conn=4 results exactly, so running it would double the operator's cost for
    nothing; the comparison is one extra BFS, far cheaper than the per-inner
    work it saves."""
    ext4 = _object_boxes(g, 4)
    yield ext4, 4
    if not _OMAP_CONN8:
        return
    ext8 = _object_boxes(g, 8)
    if ext8 is not None and ext8 != ext4:
        yield ext8, 8


def _map_object_inners(with_build=True):
    """The bounded, deliberate set of inner primitives fed to `map_objects`.

    Every one is size-preserving on a subgrid, which is the precondition for the
    in-place write-back, and every one takes `(sub, color, bg)`: the object's own
    tight bounding box filled with the grid background, the colour of the
    component the box came from (components are monochromatic, so it is exact),
    and the grid background.  The colour and the background are passed in rather
    than re-derived from `sub` because a tight box can hold no background cell at
    all (a solid rectangle) and because a box that happens to be mostly object
    would otherwise fool any 'most common colour' guess.

    The set is grouped by the ARC family it expresses *per object*:
      * the cell permutations the DSL already hand-writes as `map_flip` /
        `map_rotate` — kept for completeness (they reproduce those grids exactly,
        so they dedup against them in the composed search); `main`/`anti` are new,
        `map_flip` was only wired for `h`/`v`;
      * gravity inside each object's own box ('each shape settles in its box');
      * mirror / point-symmetry completion of each object inside its box;
      * carving the cross / diagonals, or keeping the centre row/column, through
        each object's box centre;
      * dilation of each object;
      * (optional) *construction* inners that paint inside the box, and
        (optional) length-2 compositions of a small pool.
    Primitives that are identity on a tight bounding box (crop_to_bbox,
    remove_isolated, connect_points, ...) are deliberately excluded: they cannot
    change any object, so they would only cost search time."""
    for ax in ('h', 'v', 'main', 'anti'):
        yield ('flip_' + ax, (lambda a: lambda s, c, bg: flip(s, a))(ax))
    for k in (1, 2, 3):
        yield ('rotate%d' % (90 * k), (lambda kk: lambda s, c, bg: rotate(s, kk))(k))
    for d in ('down', 'up', 'left', 'right'):
        yield ('gravity_' + d, (lambda dd: lambda s, c, bg: gravity(s, dd))(d))
    for ax in ('h', 'v'):
        yield ('mirror_' + ax, (lambda a: lambda s, c, bg: mirror_union(s, a))(ax))
    yield ('mirror_hv', lambda s, c, bg: mirror_point(s))
    yield ('dilate', lambda s, c, bg: dilate(s))
    yield ('keep_cross', lambda s, c, bg: keep_cross(s))
    yield ('remove_cross', lambda s, c, bg: remove_cross(s))
    yield ('keep_mid_row', lambda s, c, bg: keep_mid_row(s))
    yield ('keep_mid_col', lambda s, c, bg: keep_mid_col(s))
    for which in ('main', 'anti', 'both'):
        yield ('keep_diag_' + which, (lambda q: lambda s, c, bg: keep_diag(s, q))(which))
        yield ('remove_diag_' + which, (lambda q: lambda s, c, bg: remove_diag(s, q))(which))
    if _OMAP_BUILD and with_build:
        for name, fn in _omap_build_inners():
            yield (name, fn)
    if _OMAP_COMPOSE:
        for name, fn in _omap_compositions():
            yield (name, fn)


def _omap_build_inners():
    """The construction inners: primitives that *build* inside the object's own
    box instead of erasing or permuting.  All four whole-grid forms
    (`fill_bbox_region`, `draw_bbox_outline`, `draw_object_cross`, `draw_border`)
    are size-preserving on the box they are handed, and on a tight bbox subgrid
    the box *is* the subgrid, so they act exactly 'inside the object's own box'.

    They are taken directly from the existing primitive set rather than rewritten
    per object because that set has already been exercised by the whole-grid
    search; the only difference is the background the object's colour is derived
    from, which here is the grid background passed in by the caller rather than
    `_bg(sub)` (for a box that is mostly ink the latter can be the ink colour)."""
    yield ('fill_box', lambda s, c, bg: s if c is None else _fill_box(s, c, bg))
    yield ('outline_box', lambda s, c, bg: s if c is None else _outline_box(s, c, bg))
    yield ('cross_box', lambda s, c, bg: s if c is None else _cross_box(s, c, bg))
    for which in ('main', 'anti', 'both'):
        yield ('draw_diag_' + which,
               (lambda q: lambda s, c, bg: s if c is None else _draw_box_diag(s, c, bg, q))(which))


def _fill_box(s, c, bg):
    """Flood the box's background cells with the object's colour — solidifies a
    sparse shape into its own bounding rectangle."""
    r0, c0, r1, c1 = _bbox_of_nonzero(s, bg)
    out = [list(row) for row in s]
    for r in range(r0, r1 + 1):
        row = out[r]
        for cc in range(c0, c1 + 1):
            if row[cc] == bg:
                row[cc] = c
    return out


def _outline_box(s, c, bg):
    """Draw the box outline in the object's colour (background cells only)."""
    h, w = len(s), len(s[0])
    out = [list(row) for row in s]
    for cc in range(w):
        if out[0][cc] == bg:
            out[0][cc] = c
        if out[h - 1][cc] == bg:
            out[h - 1][cc] = c
    for r in range(h):
        if out[r][0] == bg:
            out[r][0] = c
        if out[r][w - 1] == bg:
            out[r][w - 1] = c
    return out


def _cross_box(s, c, bg):
    """Draw the full row and column through the box centre in the object's colour
    (background cells only), i.e. complete each object into a cross."""
    h, w = len(s), len(s[0])
    cr, cc = h // 2, w // 2
    out = [list(row) for row in s]
    for j in range(w):
        if out[cr][j] == bg:
            out[cr][j] = c
    for i in range(h):
        if out[i][cc] == bg:
            out[i][cc] = c
    return out


def _draw_box_diag(s, c, bg, which):
    """Paint the box's own diagonal(s) in the object's colour (background only)."""
    h, w = len(s), len(s[0])
    out = [list(row) for row in s]
    for r in range(h):
        for cc in range(w):
            if out[r][cc] != bg:
                continue
            on_main = (r == cc)
            on_anti = (r + cc == h - 1) or (r + cc == w - 1)
            if (which == 'main' and on_main) or (which == 'anti' and on_anti) \
                    or (which == 'both' and (on_main or on_anti)):
                out[r][cc] = c
    return out


def _omap_compose_pool():
    """The small, deliberate pool from which the length-2 inner compositions are
    drawn (size 6, reported explicitly because it squares into the branch factor).
    It holds one of each *kind* the winning single inners came from — a mirror
    completion, a rotation, gravity, dilation, the cross keeper and the diagonal
    keeper — so a composition can express the families that need two steps inside
    the box (settle-then-complete, thicken-then-carve, complete-then-rotate)."""
    yield ('mirror_h', lambda s, c, bg: mirror_union(s, 'h'))
    yield ('rotate90', lambda s, c, bg: rotate(s, 1))
    yield ('gravity_down', lambda s, c, bg: gravity(s, 'down'))
    yield ('dilate', lambda s, c, bg: dilate(s))
    yield ('keep_cross', lambda s, c, bg: keep_cross(s))
    yield ('keep_diag_both', lambda s, c, bg: keep_diag(s, 'both'))


def _omap_compositions():
    """Length-2 inner compositions `f2 ∘ f1`, both drawn from
    `_omap_compose_pool` and applied inside the object's own box.  Only ordered
    pairs of *distinct* pool members are emitted: `f ∘ f` for these six is either
    identity or a single inner already in the set (a rotation, or an idempotent
    completion), so those 6 would be pure cost.  A composition that changes the
    box size on the way (a rotation of a non-square box) is caught by the
    size-preservation check in `_map_objects_batch` and leaves the grid
    unchanged; it is never allowed to write a wrongly-shaped subgrid back."""
    pool = list(_omap_compose_pool())
    for n2, f2 in pool:
        for n1, f1 in pool:
            if n1 == n2:
                continue
            yield ('%s_after_%s' % (n2, n1),
                   (lambda a, b: lambda s, c, bg: a(b(s, c, bg), c, bg))(f2, f1))


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


# ---- v4: palette permutation, painting, symmetry-axis completion, structure
# extraction.  These target the ARC families that plain single-color recolor and
# central mirroring cannot reach: whole-palette relabeling, border/dominant-color
# painting, completing a partially-drawn reflection about a *detected* (not
# necessarily central) axis, and extracting the cross/diagonal through the grid
# centre.

def recolor_map(g, mapping):
    """Relabel every cell by a global color->color map (a palette permutation).
    Unmapped colors pass through unchanged."""
    return [[mapping.get(v, v) for v in row] for row in g]


def _infer_recolor_map(in0, out0):
    """Learn a global input-color -> output-color map from one (in, out) pair.
    Returns None if any input color maps to two different output colors."""
    mapping = {}
    for r in range(len(in0)):
        for c in range(len(in0[0])):
            a, b = in0[r][c], out0[r][c]
            if a in mapping and mapping[a] != b:
                return None
            mapping[a] = b
    return mapping or None


def _infer_recolor_map_all(train):
    """Learn a global input-color -> output-color *permutation* consistent across
    all training pairs.  Returns None for size-changing tasks, identity maps,
    non-bijective maps, or a pair where one input color maps to two outputs.

    Bijectivity is the guard against the relational-relabel trap: a task where the
    mapping differs per example (e.g. 'shape color -> corner color, corner -> 0')
    can still produce a *single* consistent many-to-one map that fits every
    example yet is wrong on the held-out test.  A true palette permutation is
    injective, so two example-local maps can only agree if they really are one
    fixed permutation of the palette."""
    mapping = {}
    for t in train:
        inp, outp = t['input'], t['output']
        if len(inp) != len(outp) or len(inp[0]) != len(outp[0]):
            return None
        for r in range(len(inp)):
            for c in range(len(inp[0])):
                a, b = inp[r][c], outp[r][c]
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b
    if not mapping:
        return None
    if all(mapping.get(k) == k for k in mapping):
        return None  # identity — not a real recolor rule
    if 0 in mapping and mapping[0] != 0:
        return None  # a palette permutation leaves the background color alone
    if len(set(mapping.values())) != len(mapping):
        return None  # not a bijection — two colors collapse onto one
    return mapping


def draw_border(g, c):
    """Paint the outer ring of cells (the border) color c."""
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        out[r][0] = c
        out[r][w - 1] = c
    for cc in range(w):
        out[0][cc] = c
        out[h - 1][cc] = c
    return out


def fill_most_common(g):
    """Flood the whole grid with its most common color."""
    cnt = Counter()
    for row in g:
        cnt.update(row)
    mc = cnt.most_common(1)[0][0]
    h, w = len(g), len(g[0])
    return [[mc] * w for _ in range(h)]


def _reflect_eq(g, target, axis, pos, bg):
    """Does completing the reflection of `g` about the axis at `pos` reproduce
    `target`?  Same rule as `reflect_complete`, compared cell by cell with an early
    exit, because the axis scan tries ~2*size candidates per grid."""
    h, w = len(g), len(g[0])
    if axis == 'v':
        for r in range(h):
            trow = target[r]
            grow = g[r]
            for c in range(w):
                v = grow[c]
                if v == bg:
                    rr = int(round(2 * pos - r))
                    if 0 <= rr < h and g[rr][c] != bg:
                        v = g[rr][c]
                if trow[c] != v:
                    return False
        return True
    for r in range(h):
        trow = target[r]
        grow = g[r]
        for c in range(w):
            v = grow[c]
            if v == bg:
                cc = int(round(2 * pos - c))
                if 0 <= cc < w and grow[cc] != bg:
                    v = grow[cc]
            if trow[c] != v:
                return False
    return True


def reflect_complete(g, axis, pos):
    """Complete a partially-drawn reflection about a *detected* axis.

    axis 'v' mirrors across the horizontal line at row `pos` (top <-> bottom);
    axis 'h' mirrors across the vertical line at column `pos` (left <-> right).
    `pos` is a half-integer so the axis can lie between cells or through them.
    Each background cell is filled from its reflected cell when the reflection is
    non-background; the drawn side is never overwritten."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    if axis == 'v':
        for r in range(h):
            for c in range(w):
                if out[r][c] == bg:
                    rr = int(round(2 * pos - r))
                    if 0 <= rr < h and g[rr][c] != bg:
                        out[r][c] = g[rr][c]
    elif axis == 'h':
        for r in range(h):
            for c in range(w):
                if out[r][c] == bg:
                    cc = int(round(2 * pos - c))
                    if 0 <= cc < w and g[r][cc] != bg:
                        out[r][c] = g[r][cc]
    return out


# Structure-extraction primitives treat 0 as the ARC background and every other
# cell as 'ink'.  This is deliberate: these operations *erase* to 0, so they must
# not fall back to the most-common color when a grid happens to contain no 0
# (e.g. a fully-colored grid whose most-common color is an object, not the ground).

def keep_cross(g):
    """Keep only the non-zero cells in the central row or central column (the
    plus/cross through the grid centre); everything else becomes 0."""
    h, w = len(g), len(g[0])
    cr, cc = h // 2, w // 2
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if (r == cr or c == cc) and g[r][c] != 0:
                out[r][c] = g[r][c]
    return out


def keep_diag(g, which='both'):
    """Keep only the non-zero cells on the main ('main'), anti ('anti'), or both
    ('both') diagonals through the grid centre."""
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                continue
            on_main = (r == c)
            on_anti = (r + c == h - 1) or (r + c == w - 1)
            if (which == 'main' and on_main) or (which == 'anti' and on_anti) \
                    or (which == 'both' and (on_main or on_anti)):
                out[r][c] = g[r][c]
    return out


def keep_mid_row(g):
    """Keep only the central row (erase everything else to 0)."""
    h, w = len(g), len(g[0])
    cr = h // 2
    return [[0] * w if r != cr else list(g[r]) for r in range(h)]


def keep_mid_col(g):
    """Keep only the central column (erase everything else to 0)."""
    h, w = len(g), len(g[0])
    cc = w // 2
    return [[g[r][cc] if c == cc else 0 for c in range(w)] for r in range(h)]


def remove_cross(g):
    """Erase the central row and column (carve a plus of 0 through the grid)."""
    h, w = len(g), len(g[0])
    cr, cc = h // 2, w // 2
    out = [list(row) for row in g]
    for r in range(h):
        out[r][cc] = 0
    for c in range(w):
        out[cr][c] = 0
    return out


def remove_diag(g, which='both'):
    """Erase the main ('main'), anti ('anti'), or both ('both') diagonals (carve
    an X of 0 through the grid)."""
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            on_main = (r == c)
            on_anti = (r + c == h - 1) or (r + c == w - 1)
            if (which == 'main' and on_main) or (which == 'anti' and on_anti) \
                    or (which == 'both' and (on_main or on_anti)):
                out[r][c] = 0
    return out


def checkerboard(g):
    """Checkerboard by Manhattan-distance parity from the 0 'hole'.  The anchor is
    the centroid of the 0 cells; cells at odd Manhattan distance from it become
    the dominant ink color, even distance stay 0."""
    h, w = len(g), len(g[0])
    holes = []
    cnt = Counter()
    for r in range(h):
        for c in range(w):
            if g[r][c] == 0:
                holes.append((r, c))
            else:
                cnt[g[r][c]] += 1
    if not holes or not cnt:
        return [list(row) for row in g]
    color = cnt.most_common(1)[0][0]
    ar = sum(r for r, _ in holes) // len(holes)
    ac = sum(c for _, c in holes) // len(holes)
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if (abs(r - ar) + abs(c - ac)) % 2 == 1:
                out[r][c] = color
    return out


def fill_uniform_rows(g, c):
    """Paint each monochromatic (single-color) row with color c; every other row
    becomes 0 (the 'highlight uniform lines' family)."""
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for r in range(h):
        if len(set(g[r])) == 1:
            out[r] = [c] * w
    return out


def fill_uniform_cols(g, c):
    """Paint each monochromatic column with color c; every other column becomes
    0."""
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for cc in range(w):
        if len(set(g[r][cc] for r in range(h))) == 1:
            for r in range(h):
                out[r][cc] = c
    return out


# ---- v6: rectangle construction (draw / fill a bounding box, per-object) ----

def draw_bbox_outline(g, color):
    """Draw the rectangle outline of the bounding box of all non-background
    cells (only background cells are painted)."""
    bg = _bg(g)
    r0, c0, r1, c1 = _bbox_of_nonzero(g, bg)
    out = [list(row) for row in g]
    for c in range(c0, c1 + 1):
        if out[r0][c] == bg:
            out[r0][c] = color
        if out[r1][c] == bg:
            out[r1][c] = color
    for r in range(r0, r1 + 1):
        if out[r][c0] == bg:
            out[r][c0] = color
        if out[r][c1] == bg:
            out[r][c1] = color
    return out


def map_bbox_outline(g, color):
    """Draw a rectangle outline around *each* object (only background cells are
    painted, so object pixels are preserved)."""
    bg = _bg(g)
    out = [list(row) for row in g]
    for _col, cells in _components(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        for c in range(c0, c1 + 1):
            if out[r0][c] == bg:
                out[r0][c] = color
            if out[r1][c] == bg:
                out[r1][c] = color
        for r in range(r0, r1 + 1):
            if out[r][c0] == bg:
                out[r][c0] = color
            if out[r][c1] == bg:
                out[r][c1] = color
    return out


def fill_bbox_region(g, color):
    """Fill the bounding box of all non-background cells with `color` (only
    background cells are painted)."""
    bg = _bg(g)
    r0, c0, r1, c1 = _bbox_of_nonzero(g, bg)
    out = [list(row) for row in g]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r][c] == bg:
                out[r][c] = color
    return out


def map_bbox_fill(g, color):
    """Fill the bounding box of *each* object with `color` (only background cells
    are painted) — the per-object counterpart of fill_bbox_region.  Objects are
    8-connected here: the ARC shapes that need this rule touch at a corner, so
    4-connectivity would split one object into two and fill the wrong box."""
    bg = _bg(g)
    out = [list(row) for row in g]
    for _col, cells in _components8(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r][c] == bg:
                    out[r][c] = color
    return out


def draw_object_cross(g, color):
    """Draw a full horizontal + vertical line through the centre of each object's
    bounding box (only background cells are painted)."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for _col, cells in _components(g, bg):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        cr = (min(rs) + max(rs)) // 2
        cc = (min(cs) + max(cs)) // 2
        for c in range(w):
            if out[cr][c] == bg:
                out[cr][c] = color
        for r in range(h):
            if out[r][cc] == bg:
                out[r][cc] = color
    return out


def crop_topleft(g, H, W):
    """Crop to the top-left H x W corner (the 'extract the fundamental unit in the
    top-left corner' family)."""
    return [row[:W] for row in g[:H]]


def tile_2d(g, H, W):
    """Detect the smallest 2-D repeating unit (row period x column period) and
    tile it to H x W — the 2-D analog of extend_rows / extend_cols."""
    h, w = len(g), len(g[0])
    ph = _period_of([tuple(r) for r in g])
    cols = [tuple(g[r][c] for r in range(h)) for c in range(w)]
    pw = _period_of(cols)
    return [[g[r % ph][c % pw] for c in range(W)] for r in range(H)]


def _count_components(g):
    return len(_components(g, _bg(g)))


def count_row(g, color):
    """Count connected components -> a 1 x N row of `color` cells (numerosity)."""
    return [[color] * _count_components(g)]


def count_col(g, color):
    """Count connected components -> an N x 1 column of `color` cells."""
    return [[color] for _ in range(_count_components(g))]


def count_diag(g, color):
    """Count connected components -> an N x N grid with `color` on the main
    diagonal."""
    n = _count_components(g)
    return [[color if i == j else 0 for j in range(n)] for i in range(n)]


def _component_color_counts(g):
    """Number of 4-connected components per color."""
    cnt = Counter()
    for color, cells in _components(g, _bg(g)):
        cnt[color] += 1
    return cnt


def noise_color(g):
    """1x1 grid of the 'noise' color in a two-color grid: the least frequent
    non-background color, with ties broken toward the color that forms *more*
    separate connected components (the scattered one rather than the solid
    block)."""
    bg = _bg(g)
    cnt = Counter()
    for row in g:
        cnt.update(row)
    if bg in cnt:
        del cnt[bg]
    if not cnt:
        return [[0]]
    comps = _component_color_counts(g)
    def key(color):
        return (cnt[color], -comps.get(color, 0))
    return [[min(cnt, key=key)]]


def least_common_color(g):
    """1x1 grid of the least frequent non-background color."""
    bg = _bg(g)
    cnt = Counter()
    for row in g:
        cnt.update(row)
    if bg in cnt:
        del cnt[bg]
    if not cnt:
        return [[0]]
    return [[min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]]]


def most_common_color(g):
    """1x1 grid of the most frequent non-background color."""
    bg = _bg(g)
    cnt = Counter()
    for row in g:
        cnt.update(row)
    if bg in cnt:
        del cnt[bg]
    if not cnt:
        return [[0]]
    return [[max(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]]]


# ------------------------------------------------- v8: layout, rank, tiling

def _size_rank_palette(in0, out0, bg, order):
    """Learn rank -> color from one example, where rank orders the *distinct
    component sizes* (not the sizes themselves).  Learned from example 1 and
    verified on all, exactly like `recolor_map`.  Returns None if a component's
    cells do not all receive one output color."""
    comps = _components(in0, bg)
    if not comps:
        return None
    sizes = sorted(set(len(cells) for _, cells in comps), reverse=(order == 'desc'))
    rank = {s: i for i, s in enumerate(sizes)}
    pal = {}
    for _, cells in comps:
        outs = set(out0[r][c] for r, c in cells)
        if len(outs) != 1:
            return None
        r = rank[len(cells)]
        v = outs.pop()
        if r in pal and pal[r] != v:
            return None
        pal[r] = v
    return [pal[i] for i in sorted(pal)] or None


def recolor_by_size_rank(g, palette, order):
    """Recolor every object by the rank of its size (largest first for 'desc'),
    taking the color from the induced rank palette.  Ranks past the learned
    palette keep their original color, so the rule degrades rather than guesses."""
    bg = _bg(g)
    comps = _components(g, bg)
    if not comps:
        return [list(row) for row in g]
    sizes = sorted(set(len(cells) for _, cells in comps), reverse=(order == 'desc'))
    rank = {s: i for i, s in enumerate(sizes)}
    out = [list(row) for row in g]
    for _, cells in comps:
        i = rank[len(cells)]
        if i < len(palette):
            for r, c in cells:
                out[r][c] = palette[i]
    return out


def _uniform_lines(g, color):
    """Row/column indices that are entirely `color` (grid separators)."""
    h, w = len(g), len(g[0])
    rows = [r for r in range(h) if all(v == color for v in g[r])]
    cols = [c for c in range(w) if all(g[r][c] == color for r in range(h))]
    return rows, cols


def remove_separator(g, color):
    """Delete the uniform separator rows/columns of `color` (the 'strip the grid
    lines and keep the panels' rule)."""
    h, w = len(g), len(g[0])
    rows, cols = _uniform_lines(g, color)
    if not rows and not cols:
        return [list(row) for row in g]
    keep_r = [r for r in range(h) if r not in rows]
    keep_c = [c for c in range(w) if c not in cols]
    if not keep_r or not keep_c:
        return [list(row) for row in g]
    return [[g[r][c] for c in keep_c] for r in keep_r]


def extract_panels(g, color):
    """Sub-grids delimited by uniform `color` separator rows/columns.  Returns the
    whole grid as a single panel when there is no separator, so callers always get
    a non-empty list."""
    h, w = len(g), len(g[0])
    rows, cols = _uniform_lines(g, color)
    rs = [-1] + rows + [h]
    cs = [-1] + cols + [w]
    out = []
    for i in range(len(rs) - 1):
        for j in range(len(cs) - 1):
            r0, r1 = rs[i] + 1, rs[i + 1]
            c0, c1 = cs[j] + 1, cs[j + 1]
            if r0 < r1 and c0 < c1:
                out.append([[g[r][c] for c in range(c0, c1)] for r in range(r0, r1)])
    return out


def _panel_slots(g, color):
    """Separator-delimited panel rectangles of `g`, in row-major order, together
    with their contents.  Unlike `extract_panels` this also returns *where* each
    panel sits, which is what lets a caller write a reordered panel back into the
    slot it came from and keep the grid size (and the separators) intact."""
    if not g or not g[0]:
        return [], []
    h, w = len(g), len(g[0])
    rows, cols = _uniform_lines(g, color)
    rs = [-1] + rows + [h]
    cs = [-1] + cols + [w]
    slots, panels = [], []
    for i in range(len(rs) - 1):
        for j in range(len(cs) - 1):
            r0, r1 = rs[i] + 1, rs[i + 1]
            c0, c1 = cs[j] + 1, cs[j + 1]
            if r0 < r1 and c0 < c1:
                slots.append((r0, r1, c0, c1))
                panels.append([[g[r][c] for c in range(c0, c1)] for r in range(r0, r1)])
    return slots, panels


def reorder_panels(g, color, key='count', desc=True):
    """Sort the separator-delimited panels of `g` by `key` and write them back
    into the same slots in sorted order.

    This is the "reorder the panels by their content" rule: the layout carries no
    information and only the *order* of the panels is the answer (42a15761, where
    each horizontal band sorts its four strips by cell count; 85b81ff1 likewise).
    Only the panel interiors move, so the separators and the grid size are
    untouched.  Panels of differing shape cannot be swapped, so they leave the
    grid unchanged and simply fail verification rather than crashing."""
    if not g or not g[0]:
        return [list(row) for row in g]
    slots, panels = _panel_slots(g, color)
    if len(panels) < 2:
        return [list(row) for row in g]
    if len({(len(p), len(p[0])) for p in panels}) != 1:
        return [list(row) for row in g]
    bg = _bg(g)

    def rank(p):
        if key == 'count':
            return sum(1 for row in p for v in row if v != bg)
        if key == 'sum':
            return sum(v for row in p for v in row)
        if key == 'lex':
            return tuple(tuple(row) for row in p)
        return len(set(v for row in p for v in row))

    order = sorted(range(len(panels)), key=lambda i: rank(panels[i]), reverse=desc)
    out = [list(row) for row in g]
    for pos, idx in enumerate(order):
        r0, r1, c0, c1 = slots[pos]
        panel = panels[idx]
        for rr in range(r1 - r0):
            row = out[r0 + rr]
            prow = panel[rr]
            for cc in range(c1 - c0):
                row[c0 + cc] = prow[cc]
    return out


def _same_shape(grids):
    if len(grids) < 2:
        return False
    if any(not g or not g[0] for g in grids):
        return False
    h, w = len(grids[0]), len(grids[0][0])
    return all(len(p) == h and len(p[0]) == w for p in grids)


def _combine_grids(grids, op, bg):
    """Cellwise logical combination of equal-shaped grids.  A cell is 'on' when it
    differs from the background; 'or' takes the first on-cell, 'and' requires all,
    'xor' exactly one, 'majority' more than half."""
    n = len(grids)
    out = []
    for r in range(len(grids[0])):
        row = []
        for c in range(len(grids[0][0])):
            nz = [p[r][c] for p in grids if p[r][c] != bg]
            if op == 'or':
                v = nz[0] if nz else bg
            elif op == 'and':
                v = nz[0] if len(nz) == n else bg
            elif op == 'xor':
                v = nz[0] if len(nz) == 1 else bg
            else:
                v = nz[0] if len(nz) * 2 > n else bg
            row.append(v)
        out.append(row)
    return out


def panel_combine(g, op, color=0):
    """Combine the separator-delimited panels of a grid by a cellwise logical op."""
    panels = extract_panels(g, color)
    if not _same_shape(panels):
        return [list(row) for row in g]
    return _combine_grids(panels, op, _bg(g))


def split_grid4(g):
    """The four quadrants of a 2x2 layout, or None.  Handles a plain even split and
    a separator-delimited layout (odd size with a uniform middle row and column)."""
    h, w = len(g), len(g[0])
    if h % 2 == 1 and w % 2 == 1:
        mr, mc = h // 2, w // 2
        if len(set(g[mr])) == 1 and len(set(g[r][mc] for r in range(h))) == 1:
            return [[[g[r][c] for c in range(mc)] for r in range(mr)],
                    [[g[r][c] for c in range(mc + 1, w)] for r in range(mr)],
                    [[g[r][c] for c in range(mc)] for r in range(mr + 1, h)],
                    [[g[r][c] for c in range(mc + 1, w)] for r in range(mr + 1, h)]]
    if h % 2 == 0 and w % 2 == 0:
        mr, mc = h // 2, w // 2
        return [[[g[r][c] for c in range(mc)] for r in range(mr)],
                [[g[r][c] for c in range(mc, w)] for r in range(mr)],
                [[g[r][c] for c in range(mc)] for r in range(mr, h)],
                [[g[r][c] for c in range(mc, w)] for r in range(mr, h)]]
    return None


def quad_combine(g, op):
    """Combine the four quadrants of a 2x2 layout cellwise ('or'/'and'/'xor'/majority)."""
    q = split_grid4(g)
    if q is None or not _same_shape(q):
        return [list(row) for row in g]
    return _combine_grids(q, op, _bg(g))


def quad_overlay(g, order):
    """Overlay the quadrants of a 2x2 layout in order, later ones winning on
    non-background cells (the 'apply one panel as a mask/stamp over another')."""
    q = split_grid4(g)
    if q is None or not _same_shape(q):
        return [list(row) for row in g]
    bg = _bg(g)
    out = [list(r) for r in q[0]]
    for i in order:
        p = q[i]
        for r in range(len(p)):
            for c in range(len(p[0])):
                if p[r][c] != bg:
                    out[r][c] = p[r][c]
    return out


def mirror_tile(g, mode):
    """Tile the grid 2x2, optionally mirroring: 'replicate' (plain), 'mirror_h'
    (left-right), 'mirror_v' (top-bottom) or 'both' (fourfold symmetry)."""
    A = [list(r) for r in g]

    def fh(x):
        return [list(reversed(r)) for r in x]

    def fv(x):
        return [list(r) for r in reversed(x)]

    B = fh(A) if mode in ('mirror_h', 'both') else A
    C = fv(A) if mode in ('mirror_v', 'both') else A
    D = fv(B) if mode in ('mirror_h', 'mirror_v', 'both') else A
    top = [A[r] + B[r] for r in range(len(A))]
    bot = [C[r] + D[r] for r in range(len(C))]
    return top + bot


def mirror_point(g):
    """Point-symmetry completion: fill background cells from the 180-degree
    rotation (the diagonal-mirror analog of `mirror_union`)."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            if out[r][c] == bg:
                out[r][c] = g[h - 1 - r][w - 1 - c]
    return out


def remove_isolated(g):
    """Erase cells with no non-background neighbour (denoising the scattered
    singleton that does not belong to any object)."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    out = [list(row) for row in g]
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg:
                continue
            n = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != bg:
                        n += 1
            if n == 0:
                out[r][c] = bg
    return out


def object_outline(g, color, conn=4, keep=False):
    """Paint the outline of every object in `color`: background-neighbouring (or
    grid-edge) object cells.  With keep=False only the outlines survive."""
    bg = _bg(g)
    h, w = len(g), len(g[0])
    nbrs = ((-1, 0), (1, 0), (0, -1), (0, 1)) if conn == 4 else \
           ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    out = [list(row) for row in g] if keep else [[bg] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg:
                continue
            edge = False
            for dr, dc in nbrs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or g[nr][nc] == bg:
                    edge = True
                    break
            if edge:
                out[r][c] = color
    return out


def crop_then_scale(g, k):
    """Crop to the bounding box of the content, then scale by k."""
    sub = crop_to_bbox(g)
    if not sub or not sub[0]:
        return [list(row) for row in g]
    out = []
    for row in sub:
        big = []
        for v in row:
            big.extend([v] * k)
        for _ in range(k):
            out.append(list(big))
    return out


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


def _enumerate_learned_maps(in0, out0):
    """The primitives that *induce a colour map from this (in0, out0) pair*.

    They are split out of `_enumerate_depth1` for two reasons.  Semantically, a
    map fitted to an intermediate grid reproduces the first example by
    construction, so its agreement is weak evidence and the search prefers
    explanations that do not re-learn a palette; keeping them in a separate
    generator lets the search take a cheap second pass over them instead of
    re-running the whole primitive sweep.  Practically, that second pass is then
    nearly free, which is what keeps the composed search affordable."""
    h, w = len(in0), len(in0[0])
    H, W = len(out0), len(out0[0])
    if (h, w) != (H, W):
        return
    target = _tup(out0)
    bg = _bg(in0)
    # recolor whole objects by their size (numerosity -> color)
    mapping = _infer_size_colors(in0, out0, bg)
    if mapping and _eq(recolor_by_size(in0, mapping), target):
        yield ('recolor_by_size', (lambda m: lambda g: recolor_by_size(g, m))(mapping))
    # recolor whole objects by the *rank* of their size.  The palette is induced
    # from this example and verified on every example, so a rank rule survives
    # examples whose size sets differ (which a size->color map cannot).
    for order in ('desc', 'asc'):
        pal = _size_rank_palette(in0, out0, bg, order)
        if pal and _eq(recolor_by_size_rank(in0, pal, order), target):
            yield ('recolor_by_rank_' + order,
                   (lambda p, o: lambda g: recolor_by_size_rank(g, p, o))(pal, order))
    # global palette permutation (learned from in0 -> out0)
    mapping = _infer_recolor_map(in0, out0)
    if mapping and len(mapping) > 1 and _eq(recolor_map(in0, mapping), target):
        yield ('recolor_map', (lambda m: lambda g: recolor_map(g, m))(mapping))


def _enumerate_depth1(in0, out0, fast=False, allow_learned=True):
    """Enumerate every single primitive mapping in0 -> out0.

    With fast=True the two expensive parameter sweeps (translation, recolor) use
    a single closed-form candidate instead of a full scan; this is exact except
    for the rare case where a translation clips the anchor cell, so it is used
    only for the final step of depth-3 search, never for depth-1.

    With allow_learned=False the primitives that *learn a colour map from this
    (in0, out0) pair* are skipped.  That matters for composed programs: a learned
    map applied to an arbitrary intermediate reproduces the first example by
    construction, so its agreement carries no evidence, and it opens a large space
    of degenerate two-step explanations (`recolor(0->2)` then a re-learned rank
    palette, say).  Those primitives stay available at depth 1, where the pair is
    a real training example."""
    h, w = len(in0), len(in0[0])
    H, W = len(out0), len(out0[0])
    target = _tup(out0)
    cols = _colors(in0)
    bg = _bg(in0)
    cdiff = len(cols ^ _colors(out0))  # color-set distance (for cheap guards)

    # identity
    if (h, w) == (H, W) and _eq(in0, target):
        yield ('identity', lambda g: [list(r) for r in g])

    # rotations / reflections (size-preserving; 'anti' requires square).
    # These permute cells, so they can only match when the color sets are equal.
    if (h, w) == (H, W) and cdiff == 0:
        for k in (1, 2, 3):
            if _eq(rotate(in0, k), target):
                yield ('rotate%d' % (90 * k), (lambda kk: lambda g: rotate(g, kk))(k))
        for ax in ('h', 'v', 'main'):
            if _eq(flip(in0, ax), target):
                yield ('flip_' + ax, (lambda a: lambda g: flip(g, a))(ax))
        if h == w and _eq(flip(in0, 'anti'), target):
            yield ('flip_anti', lambda g: flip(g, 'anti'))

    # translation (size-preserving, zero-fill); also color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if fast:
            ai = _anchor(in0, h, w)
            ao = _anchor(out0, h, w)
            if ai is not None and ao is not None:
                dx, dy = ao[1] - ai[1], ao[0] - ai[0]
                if (dx, dy) != (0, 0) and _translate_eq(in0, target, dx, dy):
                    yield ('translate(%d,%d)' % (dx, dy),
                           (lambda dx_, dy_: lambda g: translate(g, dx_, dy_))(dx, dy))
        else:
            for dy in range(-6, 7):
                for dx in range(-6, 7):
                    if (dx, dy) == (0, 0):
                        continue
                    if _translate_eq(in0, target, dx, dy):
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
                    if _eq(recolor(in0, c1, c2), target):
                        yield ('recolor(%d->%d)' % (c1, c2),
                               (lambda a, b: lambda g: recolor(g, a, b))(c1, c2))
        else:
            for c1 in cols:
                for c2 in range(10):
                    if c2 == c1:
                        continue
                    if _eq(recolor(in0, c1, c2), target):
                        yield ('recolor(%d->%d)' % (c1, c2),
                               (lambda a, b: lambda g: recolor(g, a, b))(c1, c2))

    # scale (each cell -> k x k block)
    if H % h == 0 and W % w == 0 and H // h == W // w:
        k = H // h
        if k >= 1 and _eq(scale(in0, k), target):
            yield ('scale(%d)' % k, (lambda kk: lambda g: scale(g, kk))(k))

    # tile (repeat whole grid n x m)
    if H % h == 0 and W % w == 0:
        n, m = H // h, W // w
        if (n, m) != (1, 1) and _eq(tile(in0, n, m), target):
            yield ('tile(%dx%d)' % (n, m), (lambda nn, mm: lambda g: tile(g, nn, mm))(n, m))

    # self-substitution tiling (each fg cell -> the grid)
    if H == h * h and W == w * w:
        if _eq(self_substitute(in0), target):
            yield ('self_substitute', lambda g: self_substitute(g))

    # sequence extrapolation: continue the repeating row/column pattern
    if w == W and _eq(extend_rows(in0, H), target):
        yield ('extend_rows', lambda g: extend_rows(g, H))
    if h == H and _eq(extend_cols(in0, W), target):
        yield ('extend_cols', lambda g: extend_cols(g, W))

    # crop to bounding box of non-background
    cr = crop_to_bbox(in0)
    if _eq(cr, target):
        yield ('crop', lambda g: crop_to_bbox(g))

    # flood fills (size-preserving); each adds at most one color
    if (h, w) == (H, W) and cdiff <= 1:
        # a fill paints with `c`, so `c` must occur in the output; if nothing is
        # painted the grid was already the answer and identity would have matched
        for c in _colors(out0):
            if _eq(fill_from_border(in0, c), target):
                yield ('fill_border(%d)' % c, (lambda cc: lambda g: fill_from_border(g, cc))(c))
            if _eq(fill_holes(in0, c), target):
                yield ('fill_holes(%d)' % c, (lambda cc: lambda g: fill_holes(g, cc))(c))

    # gravity (size-preserving), all four directions — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for d in ('down', 'up', 'left', 'right'):
            if _eq(gravity(in0, d), target):
                yield ('gravity_' + d, (lambda dd: lambda g: gravity(g, dd))(d))

    # gravity of a single color (others stay put) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for d in ('down', 'up', 'left', 'right'):
            for c in cols:
                if c == bg:
                    continue
                if _eq(gravity_color(in0, c, d), target):
                    yield ('gravity_color(%d,%s)' % (c, d),
                           (lambda cc, dd: lambda g: gravity_color(g, cc, dd))(c, d))

    # mirror completion (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for ax in ('h', 'v'):
            if _eq(mirror_union(in0, ax), target):
                yield ('mirror_' + ax, (lambda a: lambda g: mirror_union(g, a))(ax))

    # connect points (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if _eq(connect_points(in0), target):
            yield ('connect', lambda g: connect_points(g))
        if _eq(connect_diag(in0), target):
            yield ('connect_diag', lambda g: connect_diag(g))

    # connect same-colored points with a *new* color (adds a color: cdiff <= 1)
    if (h, w) == (H, W) and cdiff <= 1:
        for L in _colors(out0) - cols:
            if _eq(connect_newcolor(in0, L), target):
                yield ('connect_newcolor(%d)' % L,
                       (lambda cc: lambda g: connect_newcolor(g, cc))(L))

    # dilation (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        if _eq(dilate(in0), target):
            yield ('dilate', lambda g: dilate(g))

    # per-object map transforms (size-preserving) — color-preserving
    if (h, w) == (H, W) and cdiff == 0:
        for ax in ('h', 'v'):
            if _eq(map_flip(in0, ax), target):
                yield ('map_flip_' + ax, (lambda a: lambda g: map_flip(g, a))(ax))
        for k in (1, 2, 3):
            if _eq(map_rotate(in0, k), target):
                yield ('map_rotate%d' % (90 * k), (lambda kk: lambda g: map_rotate(g, kk))(k))

    # single-cell numerosity: output the least/most frequent color, or the
    # scattered 'noise' color (least frequent, ties -> more components).  Tried
    # before the crop rules because, for a 1x1 output, 'count/identify a color' is
    # the honest rule and 'crop to a single cell' is a degenerate one.
    if (H, W) == (1, 1):
        if _eq(noise_color(in0), target):
            yield ('noise_color', lambda g: noise_color(g))
        if _eq(least_common_color(in0), target):
            yield ('least_common_color', lambda g: least_common_color(g))
        if _eq(most_common_color(in0), target):
            yield ('most_common_color', lambda g: most_common_color(g))

    # crop to a single object (largest / smallest)
    for key in ('largest', 'smallest'):
        co = crop_component(in0, key)
        if _eq(co, target):
            yield ('crop_' + key, (lambda k: lambda g: crop_component(g, k))(key))

    # keep only one color (size-preserving)
    if (h, w) == (H, W):
        for c in cols:
            if c == bg:
                continue
            if _eq(keep_color(in0, c), target):
                yield ('keep_color(%d)' % c, (lambda cc: lambda g: keep_color(g, cc))(c))

    # remove the largest / smallest component (size-preserving denoising)
    if (h, w) == (H, W):
        for key in ('largest', 'smallest'):
            if _eq(remove_component(in0, key), target):
                yield ('remove_' + key, (lambda k: lambda g: remove_component(g, k))(key))

    # border painting (size-preserving)
    if (h, w) == (H, W):
        for c in _colors(out0):
            if _eq(draw_border(in0, c), target):
                yield ('draw_border(%d)' % c, (lambda cc: lambda g: draw_border(g, cc))(c))

    # dominant-color flood (size-preserving)
    if (h, w) == (H, W) and _eq(fill_most_common(in0), target):
        yield ('fill_most_common', lambda g: fill_most_common(g))

    # symmetry-axis completion: reflect about a detected (half-integer) axis.
    if (h, w) == (H, W) and cdiff == 0:
        for ax, extent in (('v', h), ('h', w)):
            for i in range(2 * extent + 1):
                pos = i / 2.0
                if _reflect_eq(in0, target, ax, pos, bg):
                    yield ('reflect_%s(%.1f)' % (ax, pos),
                           (lambda a, p: lambda g: reflect_complete(g, a, p))(ax, pos))

    # structure extraction (cross / X through the centre, keep or erase); these
    # erase to 0, which can add/remove the 0 color, so they are gated on size only.
    if (h, w) == (H, W):
        if _eq(keep_cross(in0), target):
            yield ('keep_cross', lambda g: keep_cross(g))
        for which in ('main', 'anti', 'both'):
            if _eq(keep_diag(in0, which), target):
                yield ('keep_diag_' + which,
                       (lambda q: lambda g: keep_diag(g, q))(which))
        if _eq(keep_mid_row(in0), target):
            yield ('keep_mid_row', lambda g: keep_mid_row(g))
        if _eq(keep_mid_col(in0), target):
            yield ('keep_mid_col', lambda g: keep_mid_col(g))
        if _eq(remove_cross(in0), target):
            yield ('remove_cross', lambda g: remove_cross(g))
        for which in ('main', 'anti', 'both'):
            if _eq(remove_diag(in0, which), target):
                yield ('remove_diag_' + which,
                       (lambda q: lambda g: remove_diag(g, q))(which))
        if _eq(checkerboard(in0), target):
            yield ('checkerboard', lambda g: checkerboard(g))

    # highlight uniform rows / columns (size-preserving)
    if (h, w) == (H, W):
        for c in _colors(out0):
            if _eq(fill_uniform_rows(in0, c), target):
                yield ('fill_uniform_rows(%d)' % c,
                       (lambda cc: lambda g: fill_uniform_rows(g, cc))(c))
            if _eq(fill_uniform_cols(in0, c), target):
                yield ('fill_uniform_cols(%d)' % c,
                       (lambda cc: lambda g: fill_uniform_cols(g, cc))(c))

    # rectangle construction: draw/fill a bounding box (whole-grid or per-object).
    # These paint background cells only, so they add at most one color.
    if (h, w) == (H, W) and cdiff <= 1:
        for c in _colors(out0) - cols:
            if _eq(draw_bbox_outline(in0, c), target):
                yield ('draw_bbox_outline(%d)' % c,
                       (lambda cc: lambda g: draw_bbox_outline(g, cc))(c))
            if _eq(map_bbox_outline(in0, c), target):
                yield ('map_bbox_outline(%d)' % c,
                       (lambda cc: lambda g: map_bbox_outline(g, cc))(c))
            if _eq(fill_bbox_region(in0, c), target):
                yield ('fill_bbox_region(%d)' % c,
                       (lambda cc: lambda g: fill_bbox_region(g, cc))(c))
            if _eq(map_bbox_fill(in0, c), target):
                yield ('map_bbox_fill(%d)' % c,
                       (lambda cc: lambda g: map_bbox_fill(g, cc))(c))
            if _eq(draw_object_cross(in0, c), target):
                yield ('draw_object_cross(%d)' % c,
                       (lambda cc: lambda g: draw_object_cross(g, cc))(c))

    # ---- v8: layout (panels / quadrants), symmetry tiling, denoise, outlines

    # 2x2 layout: mirror tiling grows the grid to 2H x 2W
    if H == 2 * h and W == 2 * w:
        for mode in ('both', 'mirror_h', 'mirror_v', 'replicate'):
            if _eq(mirror_tile(in0, mode), target):
                yield ('mirror_tile_' + mode, (lambda m: lambda g: mirror_tile(g, m))(mode))

    # 2x2 layout as a logical combination or overlay of the four quadrants
    if (h, w) == (H, W) and cdiff == 0:
        for op in ('or', 'xor', 'and', 'majority'):
            if _eq(quad_combine(in0, op), target):
                yield ('quad_' + op, (lambda o: lambda g: quad_combine(g, o))(op))
        for order in ((3, 2, 1), (1, 2, 3), (2, 3, 1), (0, 1, 2, 3)):
            if _eq(quad_overlay(in0, order), target):
                yield ('quad_overlay_%d%d%d%d' % (order if len(order) == 4 else order + (0,)),
                       (lambda o: lambda g: quad_overlay(g, o))(order))

    # separator-delimited panels: combine equal-shaped panels cellwise.  The
    # separator colour is swept, because the delimiter is as often a distinct
    # colour as it is the background.
    if (h, w) == (H, W) and cdiff == 0:
        for sep in _colors(in0):
            for op in ('or', 'xor', 'majority', 'and'):
                if _eq(panel_combine(in0, op, sep), target):
                    yield ('panels_%s(%d)' % (op, sep),
                           (lambda o, s: lambda g: panel_combine(g, o, s))(op, sep))

    # reorder the panels by their content, keeping the layout.  Size-preserving
    # and separator-preserving, so it is exact in both directions; only the panel
    # interiors are permuted.  `cdiff == 0` because a permutation moves cells but
    # never introduces or removes a colour.
    if (h, w) == (H, W) and cdiff == 0:
        for sep in _colors(in0):
            slots, panels = _panel_slots(in0, sep)
            if len(panels) < 2 or len({(len(p), len(p[0])) for p in panels}) != 1:
                continue
            for key in ('count', 'sum', 'lex', 'distinct'):
                for desc in (True, False):
                    if _eq(reorder_panels(in0, sep, key, desc), target):
                        yield ('reorder_panels(%d,%s,%s)' % (sep, key, 'desc' if desc else 'asc'),
                               (lambda s, k, d: lambda g: reorder_panels(g, s, k, d))(sep, key, desc))

    # strip the uniform separator rows/columns (shrinks)
    for sep in _colors(in0):
        rs = remove_separator(in0, sep)
        if _eq(rs, target) and (len(rs), len(rs[0])) != (h, w):
            yield ('remove_separator(%d)' % sep, (lambda s: lambda g: remove_separator(g, s))(sep))

    # point-symmetry completion
    if (h, w) == (H, W) and cdiff == 0 and _eq(mirror_point(in0), target):
        yield ('mirror_hv', lambda g: mirror_point(g))

    # denoise: erase cells with no non-background neighbour
    if (h, w) == (H, W) and _eq(remove_isolated(in0), target):
        yield ('remove_isolated', lambda g: remove_isolated(g))

    # per-object outlines (size-preserving, adds at most one color)
    if (h, w) == (H, W) and cdiff <= 1:
        for c in _colors(out0):
            for conn in (4, 8):
                if _eq(object_outline(in0, c, conn), target):
                    yield ('object_outline_%d(%d)' % (conn, c),
                           (lambda cc, cn: lambda g: object_outline(g, cc, cn))(c, conn))
                if _eq(object_outline(in0, c, conn, True), target):
                    yield ('object_outline_%d_keep(%d)' % (conn, c),
                           (lambda cc, cn: lambda g: object_outline(g, cc, cn, True))(c, conn))

    # crop to the content bounding box, then scale
    for k in (2, 3):
        if _eq(crop_then_scale(in0, k), target):
            yield ('crop_then_scale(%d)' % k, (lambda kk: lambda g: crop_then_scale(g, kk))(k))

    # crop to the top-left corner at output size.  A 1x1 crop is just 'return the
    # corner cell', a degenerate rule that spuriously fits single-cell-output
    # tasks, so it is excluded; the single-cell numerosity primitives below are the
    # honest 1x1 family.
    if H <= h and W <= w and (H, W) != (h, w) and (H, W) != (1, 1) \
            and _eq(crop_topleft(in0, H, W), target):
        yield ('crop_topleft', (lambda hh, ww: lambda g: crop_topleft(g, hh, ww))(H, W))

    # 2-D period tiling to output size (grows or shrinks, but not a 1x1 crop)
    if (H, W) != (1, 1) and _eq(tile_2d(in0, H, W), target):
        yield ('tile_2d', (lambda hh, ww: lambda g: tile_2d(g, hh, ww))(H, W))

    # numerosity: count components -> a row / column / diagonal of cells.
    # A count primitive has exactly one possible shape, so the output size alone
    # decides which of the three can even apply -- checking that first avoids
    # materialising an N x N grid for every colour of every candidate pair.
    ncomp = len(_components(in0, bg))
    for color in _colors(out0):
        if color == bg and len(_colors(out0)) > 1:
            continue
        if (H, W) == (1, ncomp) and _eq(count_row(in0, color), target):
            yield ('count_row(%d)' % color, (lambda cc: lambda g: count_row(g, cc))(color))
        if (H, W) == (ncomp, 1) and _eq(count_col(in0, color), target):
            yield ('count_col(%d)' % color, (lambda cc: lambda g: count_col(g, cc))(color))
        if (H, W) == (ncomp, ncomp) and _eq(count_diag(in0, color), target):
            yield ('count_diag(%d)' % color, (lambda cc: lambda g: count_diag(g, cc))(color))

    # (single-cell numerosity is tried earlier, before the degenerate crop rules)

    # generic per-object operator (v9): apply a size-preserving primitive to each
    # object's own bounding-box subgrid.  This is the generic form of the
    # map_flip / map_rotate family that `_map_object_inners` enumerates, so the
    # whole block is deliberately placed *after* every hand-written primitive: a
    # task that already had a depth-1 solution keeps the identical program, and the
    # operator can only add solves.  The dimension guard is exact — `map_objects`
    # is size-preserving on the whole grid — and is a pure speed filter.
    if (h, w) == (H, W):
        inners = list(_map_object_inners())
        for boxes, conn in _omap_conventions(in0):
            suffix = '' if conn == 4 else ':c8'
            for iname, fin, cand in _map_objects_batch(in0, inners, conn, boxes):
                if _eq(cand, target):
                    yield ('map_objects:' + iname + suffix,
                           (lambda ff, cc: lambda g: map_objects(g, ff, cc))(fin, conn))

    if allow_learned:
        for name, prog in _enumerate_learned_maps(in0, out0):
            yield (name, prog)


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
    for d in ('down', 'up', 'left', 'right'):
        for c in cols:
            if c != bg:
                yield ('gravity_color(%d,%s)' % (c, d),
                       (lambda cc, dd: lambda x: gravity_color(x, cc, dd))(c, d))
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
    for c in range(10):
        yield ('draw_border(%d)' % c, (lambda cc: lambda x: draw_border(x, cc))(c))
    yield ('fill_most_common', lambda x: fill_most_common(x))
    yield ('keep_cross', lambda x: keep_cross(x))
    for which in ('main', 'anti', 'both'):
        yield ('keep_diag_' + which, (lambda q: lambda x: keep_diag(x, q))(which))
    yield ('keep_mid_row', lambda x: keep_mid_row(x))
    yield ('keep_mid_col', lambda x: keep_mid_col(x))
    yield ('remove_cross', lambda x: remove_cross(x))
    for which in ('main', 'anti', 'both'):
        yield ('remove_diag_' + which, (lambda q: lambda x: remove_diag(x, q))(which))
    yield ('checkerboard', lambda x: checkerboard(x))
    # v8 layout / symmetry / denoise primitives (all size-preserving here)
    yield ('mirror_hv', lambda x: mirror_point(x))
    yield ('remove_isolated', lambda x: remove_isolated(x))
    for op in ('or', 'xor', 'and', 'majority'):
        yield ('quad_' + op, (lambda o: lambda x: quad_combine(x, o))(op))
        for sep in cols:
            yield ('panels_%s(%d)' % (op, sep),
                   (lambda o, s: lambda x: panel_combine(x, o, s))(op, sep))
    for order in ((3, 2, 1), (1, 2, 3), (2, 3, 1)):
        yield ('quad_overlay_%d%d%d' % order,
               (lambda o: lambda x: quad_overlay(x, o))(order))
    # reorder the separator-delimited panels by content (size-preserving).  Only
    # separator colours that actually split the grid into equal-shaped panels are
    # offered, which keeps the branch factor at ~4 per real separator rather than
    # 80.
    for sep in cols:
        slots, panels = _panel_slots(g, sep)
        if len(panels) < 2 or len({(len(p), len(p[0])) for p in panels}) != 1:
            continue
        for key in ('count', 'lex'):
            for desc in (True, False):
                yield ('reorder_panels(%d,%s,%s)' % (sep, key, 'desc' if desc else 'asc'),
                       (lambda s, k, d: lambda x: reorder_panels(x, s, k, d))(sep, key, desc))
    yield ('remove_separator(0)', lambda x: remove_separator(x, 0))
    for c in range(10):
        yield ('fill_uniform_rows(%d)' % c, (lambda cc: lambda x: fill_uniform_rows(x, cc))(c))
        yield ('fill_uniform_cols(%d)' % c, (lambda cc: lambda x: fill_uniform_cols(x, cc))(c))
    # reflect_complete is deliberately NOT an intermediate step: its half-integer
    # axis scan is high-cardinality (~2*size candidates) and, being a *completion*
    # operation, it belongs as the final, target-matched step rather than a
    # size-preserving intermediate.  Keeping it out of the intermediate set is what
    # keeps depth-3 search feasible.

    # v9: the generic per-object operator.  Every inner is size-preserving, so the
    # operator is a legal intermediate step.  Appended last so the intermediate
    # grids it introduces never displace an existing f1 in enumeration order — an
    # already-solved task still finds the same program first, which is what keeps
    # this addition regression-free.  Redundant grids (the flip/rotate inners
    # reproduce map_flip/map_rotate) are removed by the caller's dedup.
    # v_cb2: the widened inners and conn=8 are kept out of the *intermediate*
    # step set.  Every extra f1 candidate that reaches a new grid costs a whole
    # target-matched depth-1 sweep, so widening there is what makes the operator
    # expensive; the measured gain comes from the widened inners as the final,
    # target-matched step, so that is where they are wired.
    inners = list(_map_object_inners(with_build=False))
    for _boxes, conn in ((_object_boxes(g, 4), 4),):
        suffix = ''
        for iname, fin in inners:
            yield ('map_objects:' + iname + suffix,
                   (lambda ff, cc: lambda x: map_objects(x, ff, cc))(fin, conn))


def _compose(outer, inner):
    return lambda g: outer(inner(g))


def _verify(prog, train):
    return all(_tup(prog(t['input'])) == _tup(t['output']) for t in train)


def _size_compatible(g, out0):
    """Can g reach out0's dimensions in one primitive?

    Always yes.  The old guard admitted only equal size, divisible dims,
    w == W (extend_rows), h == H (extend_cols) and shrink (H <= h, W <= w),
    but that set is NOT complete with respect to `_enumerate_depth1`:

      * `tile_2d` re-tiles the detected row/column period to *exactly* H x W
        from any input, so it realises every (h, w) -> (H, W) relation (the
        only exclusion, a 1x1 output, is admitted by the shrink branch);
      * `crop_then_scale(k)` crops to the content bbox (h' x w', h' <= h,
        w' <= w) and scales by k in {2, 3}, so it realises H = k*h',
        W = k*w' — e.g. (3, 3) -> (4, 4), which the old guard rejected;
      * `count_row` / `count_col` / `count_diag` output `ncomp` (1 x n,
        n x 1, n x n) where `ncomp` is the number of 4-connected components
        and can exceed h or w, so e.g. (5, 5) -> (1, 13) was rejected.

    Because `tile_2d` makes the reachable relation set universal, the only
    sound and complete filter is "no filter"; a synthetic task built as
    `recolor(tile_2d(in0, 3, 3))` is solved ungated and missed gated.  The
    gate only ever rejected 5702 of 256551 distinct g1 over the two training
    sets and a paired re-run (80 unsolved tasks) measured the ungated search
    at 0.98x its gated cost, so dropping it is free."""
    return True


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

    # global palette permutation, learned across *all* examples (a single example
    # cannot fix the whole map when different colors appear in different pairs).
    mapping = _infer_recolor_map_all(train)
    if mapping is not None:
        return lambda g: recolor_map(g, mapping)

    # depth 2: f1 (unconditional) then f2 (target-matched)
    if max_depth < 2:
        return None
    g1_entries = []  # (name, f1, g1, size-compatible) — built once, reused below
    seen_g1 = set()
    for f1name, f1 in _unconditional_transitions(in0):
        g1 = f1(in0)
        k1 = _tup(g1)
        if k1 in seen_g1:
            continue
        seen_g1.add(k1)
        g1_entries.append((f1name, f1, g1, _size_compatible(g1, out0)))

    # Two passes.  A *learned* colour map as the second step (recolor_by_size /
    # recolor_by_rank / recolor_map) is fitted to the intermediate grid, so it
    # reproduces the first example by construction and its agreement is weak
    # evidence; after a pure colour change it is outright redundant (the
    # composition is just that map on the original).  Pass 1 therefore searches
    # only the explanations that do not re-learn a palette, and pass 2 admits them
    # as a fallback.  Both passes verify against every training example, so this
    # changes only which verified program is preferred, never whether one is.
    outc = _colors(out0)
    for learned_pass in (False, True):
        # In the learned pass the first step is ordered by how close its colour set
        # already is to the target's, so an intermediate that carries the target
        # palette is preferred over one that merely reaches it by re-colouring.
        # The learned second step reproduces example 1 by construction, so this
        # ordering is what separates a real hypothesis from a coincidental fit.
        entries = g1_entries if not learned_pass else sorted(
            g1_entries, key=lambda e: len(_colors(e[2]) ^ outc))
        for f1name, f1, g1, size_ok in entries:
            if not size_ok:
                continue
            f2s = _enumerate_learned_maps(g1, out0) if learned_pass \
                else _enumerate_depth1(g1, out0, allow_learned=False)
            for f2name, f2 in f2s:
                prog = _compose(f2, f1)
                if _verify(prog, train):
                    return prog

    # depth 3: f1 then f2 then f3, from the beam of closest g1
    if max_depth >= 3:
        g1_list = sorted(((_closeness(g1, out0), f1)
                          for _, f1, g1, _ in g1_entries), key=lambda t: t[0])
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
                for f3name, f3 in _enumerate_depth1(g2, out0, fast=True, allow_learned=True):
                    prog = _compose(f3, _compose(f2, f1))
                    if _verify(prog, train):
                        return prog
    return None


def solve_task(task, max_depth=2, beam=16):
    prog = find_program(task['train'], max_depth=max_depth, beam=beam)
    if prog is None:
        return None
    return [prog(t['input']) for t in task['test']]
