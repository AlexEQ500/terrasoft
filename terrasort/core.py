"""Terrasort core: rule-based point cloud classification for survey data.

Pipeline (all on a voxel-averaged "working set", labels are mapped back to every
original point afterwards):

    noise -> ground (progressive lowest-point TIN) -> height above ground
          -> kerbs (2.5D step detection) -> local PCA features
          -> walls (vertical planar components) -> objects (tree / shrub / person / other)

Every threshold lives in a Profile, so a sensor type (static scan, drone
photogrammetry, ...) is just a different set of numbers.

IMPORTANT: the default thresholds are starting points. They were only exercised
on synthetic data, so expect to tune them on real sites.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- classes
UNCLASSIFIED, GROUND, LOW_VEG, MID_VEG, TREE, BUILDING, NOISE = 1, 2, 3, 4, 5, 6, 7  # 6 = LAS 'Building'
KERB, PERSON, POLE, ROOF, WALL, CAR, POLE_LOW, FURNITURE, SIGN, LOWFLAT = 64, 65, 66, 67, 68, 69, 70, 71, 72, 73  # user-definable LAS codes (need point format 6+)

CLASS_NAMES = {
    UNCLASSIFIED: "Unclassified / other",
    GROUND: "Ground",
    LOW_VEG: "Low vegetation",
    MID_VEG: "Shrub / medium vegetation",
    TREE: "Tree",
    BUILDING: "Building",
    WALL: "Wall / fence (low)",
    CAR: "Car / vehicle",
    NOISE: "Noise",
    KERB: "Kerb",
    PERSON: "Person (heuristic)",
    POLE: "High pole (3 m and up)",
    POLE_LOW: "Low pole / post / rail (under 3 m)",
    FURNITURE: "Street furniture (bin, bench, cabinet)",
    SIGN: "Sign / signal",
    LOWFLAT: "Low flat surface (step, plinth, bed)",
    ROOF: "Roof / canopy (low)",
}


# ---------------------------------------------------------------- profiles
@dataclass(frozen=True)
class Profile:
    name: str
    voxel: float                      # working-set voxel size, m
    max_voxels: int = 4_000_000       # voxel is enlarged until the working set fits
    # noise
    noise_k: int = 6
    noise_mult: float = 8.0           # isolated if 6th neighbour is > mult*voxel away
    # ground
    ground_cells: tuple = (8.0, 4.0, 2.0, 1.0, 0.5)
    ground_seed_th: tuple = (1.5, 0.8, 0.5, 0.3, 0.2)
    ground_th: float = 0.08           # max height above surface to still be ground
    ground_below: float = 0.15        # max depth below surface
    # kerb
    kerb_cell: float = 0.05
    kerb_edge_radius: float = 0.05    # window used to see the step
    kerb_min_step: float = 0.06
    kerb_max_step: float = 0.30
    kerb_clear_radius: float = 0.30   # nothing taller than the kerb top within this
    kerb_min_len: float = 1.5
    # shape features (measured at ~0.4 m scale on a coarse subsample, robust to sensor noise)
    coarse_cell: float = 0.15
    coarse_k: int = 24
    # wall / fence (vertical planar)
    wall_min_hag: float = 0.30
    wall_planar_min: float = 0.45     # planarity >= this
    wall_nz_max: float = 0.30         # |normal z| below this = vertical surface
    wall_min_height: float = 0.6
    wall_min_len: float = 1.0
    wall_min_pts: int = 150
    wall_box_len: float = 8.0         # shorter components must be thin (rejects boxes: cars, cabinets, bins)
    # roof / elevated flat surface
    roof_min_hag: float = 2.0
    roof_planar_min: float = 0.5
    roof_nz_min: float = 0.45
    roof_min_area: float = 3.0        # m2
    # buildings (tall wall / roof structures) and cars
    low_flat: str = "class"           # low horizontal leftovers just above the ground: "off" | "ground" | "class"
    low_flat_max: float = 0.8         # ... up to this height above ground (m)
    low_flat_min_area: float = 0.4    # ... in connected patches of at least this area (m2)
    building_min_h: float = 2.5       # wall/roof structure reaching this height is a building
    car_min_hag: float = 0.30
    car_max_hag: float = 2.40
    car_len: tuple = (2.5, 7.5)
    car_width: tuple = (1.3, 2.6)
    car_height: tuple = (1.0, 2.4)
    car_min_pts: int = 120
    car_max_foliage: float = 0.30
    car_planar_share: float = 0.25
    car_roof_extend: float = 0.5      # roof-slab pass: grow the roof footprint this far along the car (m)
    car_roof_flat: float = 0.5        # roof-slab pass: max height spread of the flat roof (m)
    # objects
    obj_cell_mult: float = 3.0        # component cell = mult * voxel
    pole_lin_min: float = 0.6         # linearity >= this = line-like point
    pole_split_h: float = 3.0         # pole taller than this = high pole, else low pole
    pole_min_h: float = 0.3           # shortest thing still called a pole
    thin_cell: float = 0.15           # fine-scale check for thin posts / rails
    thin_lin_min: float = 0.75
    thin_max_width: float = 0.5       # plan-view width of a thin object (m)
    pole_frac: float = 0.6            # component is pole/rail if >= this share is line-like
    fol_sph_min: float = 0.08         # sphericity >= this (and not line/plane) = foliage-like
    fol_frac_min: float = 0.45        # component is vegetation if >= this share is foliage-like
    tree_min_height: float = 2.5
    crown_hag: float = 2.0            # foliage above this height in a tall cluster is tree crown
    trunk_near_xy: float = 0.6        # line-like cluster this close (plan view) to a crown is a trunk
    trunk_catch_xy: float = 1.0       # leftovers (thick trunks, branches) this close to a crown, below it -> tree
    tree_near_xy: float = 1.5         # lower foliage / trunks within this (m, plan view) of a crown are tree
    bare_tree_width: float = 2.0      # leafless tree: thin-branch cluster at least this wide and 2.5 m tall
    struct_excl: float = 0.35         # foliage test ignores points this close to a wall / roof
    low_veg_max: float = 0.5
    detect_people: bool = True
    person_min_pts: int = 150
    min_obj_pts: int = 8


PROFILES = {
    "static_scan": Profile("static_scan", voxel=0.05),
    "drone_photogrammetry": Profile(
        "drone_photogrammetry", voxel=0.08, ground_th=0.12, kerb_cell=0.08,
        kerb_edge_radius=0.08, coarse_cell=0.2, coarse_k=20, wall_min_pts=80,
        detect_people=False,  # people are not reliably resolved in nadir photogrammetry
    ),
}


def get_profile(name: str, **overrides) -> Profile:
    p = PROFILES[name]
    return replace(p, **overrides) if overrides else p


# ---------------------------------------------------------------- helpers
def _pack(ijk, dims):
    return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]


def voxelize(xyz: np.ndarray, voxel: float, max_voxels: int):
    """Average points per voxel. Returns (voxel_xyz, inverse, voxel_used)."""
    mn = xyz.min(0)
    v = voxel
    while True:
        ijk = np.floor((xyz - mn) / v).astype(np.int64)
        dims = ijk.max(0) + 1
        key = _pack(ijk, dims)
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        if len(uk) <= max_voxels:
            break
        v *= 1.25
    inv = inv.reshape(-1)
    sums = np.stack(
        [np.bincount(inv, weights=xyz[:, d] - mn[d], minlength=len(uk)) for d in range(3)], 1
    )
    return (sums / cnt[:, None] + mn), inv, v


def within(src: np.ndarray, dst: np.ndarray, r: float) -> np.ndarray:
    """bool per dst point: is some src point within distance r? Same answer as a nearest-neighbour query but
    prunes dst to the cells next to src first, so a small src set no longer costs a query for every dst point."""
    out = np.zeros(len(dst), bool)
    if len(src) == 0 or len(dst) == 0:
        return out
    mn = np.minimum(src.min(0), dst.min(0))
    cs = np.floor((src - mn) / r).astype(np.int64)
    cd = np.floor((dst - mn) / r).astype(np.int64)
    dims = np.maximum(cs.max(0), cd.max(0)) + 3
    off = np.array([(a, b, c) for a in (0, 1, 2) for b in (0, 1, 2) for c in (0, 1, 2)], np.int64)
    sk = np.unique(_pack(cs + 1, dims))
    # neighbours of every occupied src cell (cells are shifted by +1 so offsets never go negative)
    nk = np.unique((sk[:, None] + (_pack(off, dims) - _pack(np.array([[1, 1, 1]]), dims))).ravel())
    dk = _pack(cd + 1, dims)
    pos = np.searchsorted(nk, dk)
    pos[pos >= len(nk)] = len(nk) - 1
    cand = np.flatnonzero(nk[pos] == dk)
    if len(cand):
        d, _ = cKDTree(src).query(dst[cand], k=1, distance_upper_bound=r * 1.000001, workers=-1)
        out[cand] = d <= r
    return out


def components(pts: np.ndarray, cell: float, conn: int = 26):
    """Connected components of occupied cells (26- or 6-connectivity)."""
    mn = pts.min(0)
    ijk = np.floor((pts - mn) / cell).astype(np.int64) + 1
    dims = ijk.max(0) + 2
    key = _pack(ijk, dims)
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    iu = ijk[first]
    rows, cols = [], []
    for o in itertools.product((-1, 0, 1), repeat=3):
        if o <= (0, 0, 0):
            continue
        if conn == 6 and sum(map(abs, o)) != 1:
            continue
        nk = _pack(iu + np.array(o), dims)
        pos = np.minimum(np.searchsorted(uk, nk), len(uk) - 1)
        ok = uk[pos] == nk
        rows.append(np.flatnonzero(ok))
        cols.append(pos[ok])
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    g = csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(len(uk), len(uk)))
    n, lab = connected_components(g, directed=False)
    return lab[inv], n


def _groups(lab: np.ndarray):
    """Yield (label, index array) for every label value."""
    order = np.argsort(lab, kind="stable")
    sl = lab[order]
    cuts = np.flatnonzero(np.r_[True, sl[1:] != sl[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        yield sl[a], order[a:b]


def _xy_extent(xy: np.ndarray):
    """(major, minor) extent of a point set along its principal XY axes."""
    if len(xy) < 3:
        return 0.0, 0.0
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    p = c @ v
    e = np.ptp(p, axis=0)
    return float(e[1]), float(e[0])


class Surface:
    """Piecewise-linear ground surface through seed points (nearest-neighbour outside hull)."""

    def __init__(self, seeds: np.ndarray):
        self.z = seeds[:, 2]
        self.nn = cKDTree(seeds[:, :2])
        self.lin = None
        if len(seeds) >= 4:
            try:
                self.lin = LinearNDInterpolator(seeds[:, :2], seeds[:, 2])
            except Exception:
                self.lin = None

    def __call__(self, xy):
        z = self.lin(xy) if self.lin is not None else np.full(len(xy), np.nan)
        bad = np.isnan(z)
        if bad.any():
            _, j = self.nn.query(xy[bad])
            z[bad] = self.z[j]
        return z


def _lowest_per_cell(xy, z, idx, cell, origin):
    ij = np.floor((xy[idx] - origin) / cell).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    order = np.lexsort((z[idx], key))
    ks = key[order]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    return idx[order[first]]


# ---------------------------------------------------------------- stages
def find_noise(pts, prof):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=prof.noise_k + 1, workers=-1)
    return d[:, -1] > prof.noise_mult * prof.voxel, tree


def find_ground(pts, noise, prof, prog=None):
    xy, z = pts[:, :2], pts[:, 2]
    origin = xy.min(0)
    ok = np.flatnonzero(~noise)
    surf = None
    for i, (cell, th) in enumerate(zip(prof.ground_cells, prof.ground_seed_th)):
        if surf is None:
            pool = ok
        else:
            dz = z[ok] - surf(xy[ok])
            pool = ok[(dz < th) & (dz > -prof.ground_below * 3)]
        seeds = _lowest_per_cell(xy, z, pool, cell, origin)
        surf = Surface(pts[seeds])
        if prog:
            prog(0.10 + 0.10 * (i + 1) / len(prof.ground_cells), f"ground pass {i + 1}/{len(prof.ground_cells)}")
    dz = np.full(len(pts), np.inf)
    dz[ok] = z[ok] - surf(xy[ok])
    ground = (~noise) & (dz <= prof.ground_th) & (dz >= -prof.ground_below)
    # re-fit the surface through the *middle* of the ground points (0.5 m cell means)
    g = np.flatnonzero(ground)
    ij = np.floor((xy[g] - origin) / 0.5).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv)
    seeds = np.stack([np.bincount(inv, weights=pts[g, d]) / cnt for d in range(3)], 1)
    surf2 = Surface(seeds)
    hag = z - surf2(xy)
    return ground, hag


def find_kerbs(pts, hag, noise, prof, log=print):
    """Detect kerbs as short, low, elongated height steps on a 2.5D raster."""
    xy = pts[:, :2]
    z = (pts[:, 2] - pts[:, 2].min()).astype(np.float32)
    mn = xy.min(0)
    cell = prof.kerb_cell
    ext = xy.max(0) - mn
    while (ext[0] / cell + 2) * (ext[1] / cell + 2) > 25e6:
        cell *= 1.5
        log(f"kerb raster too large, coarsening cell to {cell:.2f} m")
    ij = np.floor((xy - mn) / cell).astype(np.int64)
    shape = ij.max(0) + 1
    flat = ij[:, 0] * shape[1] + ij[:, 1]
    BIG = np.float32(1e6)

    low = (~noise) & (hag <= 0.5) & (hag >= -0.2)
    zlo = np.full(shape.prod(), BIG, np.float32)
    np.minimum.at(zlo, flat[low], z[low])
    zlo = zlo.reshape(shape)
    near = (~noise) & (hag <= 1.0) & (hag >= -0.2)
    zhi = np.full(shape.prod(), -BIG, np.float32)
    np.maximum.at(zhi, flat[near], z[near])
    zhi = zhi.reshape(shape)

    empty = zlo >= BIG / 2
    s1 = 2 * max(1, int(round(prof.kerb_edge_radius / cell))) + 1
    s2 = 2 * max(1, int(round(prof.kerb_clear_radius / cell))) + 1
    wlo = ndimage.minimum_filter(zlo, size=s1)
    whi = ndimage.maximum_filter(np.where(empty, -BIG, zlo), size=s1)
    whi2 = ndimage.maximum_filter(np.where(empty, -BIG, zlo), size=s2)
    cap = ndimage.maximum_filter(zhi, size=s2)
    step = whi - wlo
    valid = (wlo < BIG / 2) & (whi > -BIG / 2)
    cand = valid & (step >= prof.kerb_min_step) & (step <= prof.kerb_max_step) & ((cap - whi2) <= 0.06)

    # keep only elongated components of at least kerb_min_len
    lab, n = ndimage.label(cand, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    if n:
        rr, cc = np.nonzero(lab)
        cl = lab[rr, cc]
        for L, idx in _groups(cl):
            major, minor = _xy_extent(np.stack([rr[idx], cc[idx]], 1) * cell)
            if major >= prof.kerb_min_len and minor <= 0.4 * major:
                keep[L] = True
    cellmask = keep[lab]
    fl = flat
    in_kerb = cellmask.reshape(-1)[fl] & low
    upper = (z - wlo.reshape(-1)[fl]) >= np.maximum(0.02, 0.15 * step.reshape(-1)[fl])
    return in_kerb & upper


def coarse_features(pts, idx, cell, k, chunk=100_000):
    """Shape features at roughly 0.4 m scale, computed on a coarse subsample of pts[idx].

    Returns dict of float32 arrays (len(pts)): lin (line-like), pla (plane-like),
    sph (volumetric), nz (|normal z|), vz (|z of principal axis|).  Zero outside idx.
    """
    names = ("lin", "pla", "sph", "nz", "vz")
    out = {n: np.zeros(len(pts), np.float32) for n in names}
    if len(idx) < 8:
        return out
    sub, inv, _ = voxelize(pts[idx], cell, 10**9)
    if len(sub) < 8:
        return out
    kk = min(k, len(sub))
    tree = cKDTree(sub)
    f = np.zeros((len(sub), 5), np.float32)
    for s in range(0, len(sub), chunk):
        q = np.arange(s, min(s + chunk, len(sub)))
        _, nb = tree.query(sub[q], k=kk, workers=-1)
        P = sub[nb]
        P = P - P.mean(1, keepdims=True)
        C = np.einsum("mki,mkj->mij", P, P) / kk
        w, v = np.linalg.eigh(C)
        sg = np.sqrt(np.clip(w, 0, None)) + 1e-9
        s0, s1, s2 = sg[:, 0], sg[:, 1], sg[:, 2]
        f[q, 0] = (s2 - s1) / s2
        f[q, 1] = (s1 - s0) / s2
        f[q, 2] = s0 / s2
        f[q, 3] = np.abs(v[:, 2, 0])
        f[q, 4] = np.abs(v[:, 2, 2])
    for j, n in enumerate(names):
        out[n][idx] = f[inv, j]
    return out


def _planar_components(pts, hag, cand, prof, min_h, min_len, min_pts, min_area=0.0, thin_below=0.0):
    """Connected planar candidates that are large enough. Returns bool mask."""
    keep = np.zeros(len(pts), bool)
    ci = np.flatnonzero(cand)
    if len(ci) < min_pts:
        return keep
    lab, _ = components(pts[ci], max(2 * prof.voxel, 0.12))
    for L, idx in _groups(lab):
        if len(idx) < min_pts or len(idx) * prof.voxel**2 < min_area:
            continue
        sel = ci[idx]
        major, minor = _xy_extent(pts[sel, :2])
        h = np.ptp(hag[sel])
        if h >= min_h and major >= min_len and (major >= thin_below or minor <= 0.25 * major + 0.15):
            keep[sel] = True
    return keep


def find_walls(pts, hag, F, free, prof):
    """Vertical planar surfaces (walls, fences, panels)."""
    cand = free & (hag >= prof.wall_min_hag) & (F["pla"] >= prof.wall_planar_min) & (F["nz"] <= prof.wall_nz_max)
    wall = _planar_components(pts, hag, cand, prof, prof.wall_min_height, prof.wall_min_len, prof.wall_min_pts,
                              thin_below=prof.wall_box_len)
    # one dilation step: pick up wall edges / window frames / the foot of the wall
    wi = np.flatnonzero(wall)
    if len(wi):
        rest = np.flatnonzero(free & ~wall & (hag >= 0.1))
        if len(rest):
            wall[rest[within(pts[wi], pts[rest], 2.0 * prof.voxel)]] = True
    return wall


def find_roofs(pts, hag, F, free, prof):
    """Large planar surfaces well above the ground that are not vertical."""
    cand = free & (hag >= prof.roof_min_hag) & (F["pla"] >= prof.roof_planar_min) & (F["nz"] >= prof.roof_nz_min)
    roof = _planar_components(pts, hag, cand, prof, 0.0, 1.0, prof.wall_min_pts, prof.roof_min_area)
    ri = np.flatnonzero(roof)
    if len(ri):  # roof edges, ridge line, fascia
        rest = np.flatnonzero(free & ~roof & (hag >= prof.roof_min_hag - 0.3))
        if len(rest):
            roof[rest[within(pts[ri], pts[rest], 2.5 * prof.voxel)]] = True
    return roof


def find_cars(pts, hag, F, free, prof):
    """Box-like clusters of car size that are not foliage (needs the roof or both sides to be seen)."""
    car = np.zeros(len(pts), bool)
    cand = free & (hag >= prof.car_min_hag) & (hag <= prof.car_max_hag)
    ci = np.flatnonzero(cand)
    if len(ci) < prof.car_min_pts:
        return car
    foliage = (F["sph"] >= prof.fol_sph_min) & (F["lin"] < prof.pole_lin_min) & (F["pla"] < prof.roof_planar_min)
    lab, _ = components(pts[ci], max(2 * prof.voxel, 0.2))
    for L, idx in _groups(lab):
        if len(idx) < prof.car_min_pts:
            continue
        sel = ci[idx]
        major, minor = _xy_extent(pts[sel, :2])
        maxh = float(hag[sel].max())
        if not (prof.car_len[0] <= major <= prof.car_len[1] and prof.car_width[0] <= minor <= prof.car_width[1]
                and prof.car_height[0] <= maxh <= prof.car_height[1]):
            continue
        if foliage[sel].mean() > prof.car_max_foliage or (F["pla"][sel] >= 0.4).mean() < prof.car_planar_share:
            continue
        car[sel] = True
    # second pass: cars touching each other / hedges / fences never form one car-sized cluster, but each
    # roof is still a separate flat patch ~1.2-2.0 m above ground; grow the car from its roof footprint
    rc = np.flatnonzero(free & ~car & (hag >= 0.9) & (hag <= prof.car_height[1]) & (F["pla"] >= 0.4) & (F["nz"] >= 0.75))
    if len(rc) >= prof.car_min_pts:
        lab, _ = components(pts[rc], max(2 * prof.voxel, 0.2))
        for L, idx in _groups(lab):
            if len(idx) < prof.car_min_pts:
                continue
            sel = rc[idx]
            xy = pts[sel, :2] - pts[sel, :2].mean(0)
            w, v = np.linalg.eigh(xy.T @ xy / len(sel))
            u, n_ = v[:, 1], v[:, 0]
            a_all = xy @ u
            # roofs of bumper-to-bumper cars join into one long patch: cut it at gaps along its length
            o = np.argsort(a_all)
            piece = np.zeros(len(sel), np.int64)
            piece[o] = np.concatenate([[0], np.cumsum(np.diff(a_all[o]) > 1.3 * prof.voxel)])
            for pid in np.unique(piece):
                ps = sel[piece == pid]
                if len(ps) < prof.car_min_pts or np.ptp(hag[ps]) > prof.car_roof_flat:
                    continue
                c0 = pts[ps, :2].mean(0)
                rel_ = pts[ps, :2] - c0
                a, b = rel_ @ u, rel_ @ n_
                major, minor = float(np.ptp(a)), float(np.ptp(b))
                if not (1.2 <= major <= prof.car_len[1] - 1.0 and 1.1 <= minor <= prof.car_width[1]
                        and len(ps) * prof.voxel**2 >= 1.2):
                    continue
                a0, a1 = a.min() - prof.car_roof_extend, a.max() + prof.car_roof_extend
                b0, b1 = b.min() - 0.2, b.max() + 0.2
                top = float(hag[ps].max()) + 0.25
                ri_ = np.flatnonzero(free & ~car & (hag >= 0.1) & (hag <= top))
                rel = pts[ri_, :2] - c0
                ra, rb = rel @ u, rel @ n_
                car[ri_[(ra >= a0) & (ra <= a1) & (rb >= b0) & (rb <= b1)]] = True
                car[ps] = True
    ki = np.flatnonzero(car)
    if len(ki):  # lower body, sills, wheels
        rest = np.flatnonzero(free & ~car & (hag >= 0.05))
        if len(rest):
            car[rest[within(pts[ki], pts[rest], 3.0 * prof.voxel)]] = True
    return car


def split_buildings(pts, hag, wall, roof, prof):
    """Tall connected wall/roof structures become buildings; low ones stay wall/fence or canopy."""
    building = np.zeros(len(pts), bool)
    si = np.flatnonzero(wall | roof)
    if len(si) == 0:
        return building
    lab, _ = components(pts[si], max(2 * prof.voxel, 0.2))
    for L, idx in _groups(lab):
        sel = si[idx]
        if float(hag[sel].max()) >= prof.building_min_h:
            building[sel] = True
    return building


def _median_curv(pts, tree, sel, k=20):
    _, nb = tree.query(pts[sel], k=k, workers=-1)
    P = pts[nb]
    P = P - P.mean(1, keepdims=True)
    w = np.linalg.eigvalsh(np.einsum("mki,mkj->mij", P, P) / k)
    return float(np.median(w[:, 0] / (w.sum(1) + 1e-12)))


def classify_objects(pts, hag, F, free, prof, tree, near_struct):
    """Objects left after ground / kerb / wall / roof.

    A: tiny clusters -> noise, person candidates (all shapes together).
    B: foliage-like points -> tree crowns / shrubs / low vegetation (per point height inside a cluster,
       so a hedge or lawn touching a tree does not become 'tree').
    C: line-like points -> bare tree, trunk under a crown, or pole / post / rail.
    Everything else (flat leftovers, boxes) stays unclassified.
    """
    out = np.zeros(len(pts), np.uint8)
    ri = np.flatnonzero(free)
    if len(ri) == 0:
        return out

    # ---- A: noise + people (+ cluster-level foliage share)
    linear = F["lin"] >= prof.pole_lin_min
    foliage = (F["sph"] >= prof.fol_sph_min) & ~linear & (F["pla"] < prof.roof_planar_min) & ~near_struct
    lab, _ = components(pts[ri], prof.obj_cell_mult * prof.voxel)
    for L, idx in _groups(lab):
        sel = ri[idx]
        n = len(sel)
        if n < prof.min_obj_pts:
            out[sel] = NOISE
            continue
        # a cluster whose points are mostly NOT foliage-like (box edges, wires, flat leftovers)
        # does not count as vegetation, even if its edge points look scattered
        fol_f = float(foliage[sel].mean())
        if fol_f < prof.fol_frac_min:
            foliage[sel] = False
        if not prof.detect_people or n < prof.person_min_pts:
            continue
        maxh, minh = float(hag[sel].max()), float(hag[sel].min())
        if not (1.2 <= maxh <= 2.2 and minh <= 0.4):
            continue
        major, minor = _xy_extent(pts[sel, :2])
        if (0.25 <= major <= 1.1 and 0.15 <= minor <= 0.9 and maxh / max(major, 0.05) >= 1.6
                and _median_curv(pts, tree, sel[:: max(1, n // 400)]) < 0.08):
            out[sel] = PERSON

    rem = free & (out == 0)

    # ---- B: foliage
    fi = np.flatnonzero(rem & foliage)
    low_in_tall = []
    if len(fi):
        lab, _ = components(pts[fi], max(prof.obj_cell_mult * prof.voxel, 0.25))
        for L, idx in _groups(lab):
            sel = fi[idx]
            if len(sel) < prof.min_obj_pts:
                continue
            if hag[sel].max() >= prof.tree_min_height:
                crown = sel[hag[sel] >= prof.crown_hag]
                out[crown] = TREE
                low_in_tall.append(sel[hag[sel] < prof.crown_hag])
            else:
                out[sel] = MID_VEG if hag[sel].max() >= prof.low_veg_max else LOW_VEG
    crown_xy = cKDTree(pts[out == TREE][:, :2]) if (out == TREE).any() else None
    for sel in low_in_tall:
        if crown_xy is not None:
            d, _ = crown_xy.query(pts[sel, :2], k=1, workers=-1)
            near = (d <= prof.tree_near_xy) & (hag[sel] >= prof.low_veg_max)
        else:
            near = np.zeros(len(sel), bool)
        out[sel[near]] = TREE
        far = sel[~near]
        out[far] = np.where(hag[far] >= prof.low_veg_max, MID_VEG, LOW_VEG)

    # ---- C: line-like things
    li = np.flatnonzero(rem & linear & (out == 0))
    if len(li):
        lab, _ = components(pts[li], max(prof.obj_cell_mult * prof.voxel, 0.25))
        for L, idx in _groups(lab):
            sel = li[idx]
            if len(sel) < prof.min_obj_pts:
                continue
            maxh = float(hag[sel].max())
            major, minor = _xy_extent(pts[sel, :2])
            near_crown = False
            if crown_xy is not None:
                d, _ = crown_xy.query(pts[sel, :2], k=1, workers=-1)
                near_crown = float((d <= prof.trunk_near_xy).mean()) >= 0.5 and maxh >= prof.low_veg_max
            if near_crown or (maxh >= prof.tree_min_height and minor >= prof.bare_tree_width * 0.5
                              and major >= prof.bare_tree_width):
                out[sel] = TREE
            elif maxh >= prof.pole_min_h:
                out[sel] = POLE if maxh >= prof.pole_split_h else POLE_LOW

    # ---- D: trunks and branches that are not thin/line-like at the coarse scale: whatever is left
    # directly under / beside a crown and is not a flat surface belongs to that tree
    if crown_xy is not None:
        left = np.flatnonzero(free & (out == 0) & (hag >= prof.low_veg_max) & ~near_struct & (F["pla"] < 0.6))
        if len(left):
            d, _ = crown_xy.query(pts[left, :2], k=1, workers=-1)
            out[left[d <= prof.trunk_catch_xy]] = TREE

    # ---- E: thin posts / bollards / rails the coarse scale could not see: fine-scale line-like leftovers
    rest = np.flatnonzero(free & (out == 0) & (hag >= prof.pole_min_h) & ~near_struct)
    if len(rest) > 20:
        Ff = coarse_features(pts, rest, max(prof.thin_cell, 2 * prof.voxel), 12)
        ti = rest[(Ff["lin"][rest] >= prof.thin_lin_min) & (Ff["pla"][rest] < 0.3)]
        if len(ti):
            lab, _ = components(pts[ti], max(prof.obj_cell_mult * prof.voxel, 0.25))
            for L, idx in _groups(lab):
                sel = ti[idx]
                if len(sel) < prof.min_obj_pts:
                    continue
                major, minor = _xy_extent(pts[sel, :2])
                maxh = float(hag[sel].max())
                vert = float(Ff["vz"][sel].mean()) > 0.7
                if minor <= prof.thin_max_width and (vert or major >= 0.5):
                    out[sel] = POLE if maxh >= prof.pole_split_h else POLE_LOW

    out[free & (out == 0)] = UNCLASSIFIED
    return out


# ---------------------------------------------------------------- driver
def classify_points(pts: np.ndarray, prof: Profile, prog=None, log=print, store: dict | None = None):
    """Classify a (voxel-averaged) point set. Returns (labels uint8, hag float32).

    If a dict is passed as `store`, the shape features are put into store["F"] (used by click-to-pick).
    """
    P = (lambda f, m: prog(f, m)) if prog else (lambda f, m: None)
    n = len(pts)
    labels = np.full(n, UNCLASSIFIED, np.uint8)

    P(0.05, "noise")
    noise, tree = find_noise(pts, prof)
    P(0.10, "ground")
    ground, hag = find_ground(pts, noise, prof, prog)
    labels[ground] = GROUND

    P(0.25, "kerbs")
    kerb = find_kerbs(pts, hag, noise, prof, log) & ~noise
    labels[kerb] = KERB
    ground &= ~kerb

    P(0.40, "shape features")
    free = (~ground) & (~noise) & (~kerb)
    F = coarse_features(pts, np.flatnonzero(free), prof.coarse_cell, prof.coarse_k)
    if store is not None:
        store["F"] = F

    P(0.55, "cars")
    car = find_cars(pts, hag, F, free, prof)
    labels[car] = CAR
    free &= ~car

    P(0.62, "walls and fences")
    wall = find_walls(pts, hag, F, free, prof)
    free &= ~wall

    P(0.70, "roofs")
    roof = find_roofs(pts, hag, F, free, prof)
    free &= ~roof

    building = split_buildings(pts, hag, wall, roof, prof)
    labels[roof] = ROOF
    labels[wall] = WALL
    labels[building] = BUILDING

    P(0.80, "objects")
    near_struct = np.zeros(n, bool)
    si = np.flatnonzero(wall | roof)
    if len(si):
        fi_ = np.flatnonzero(free)
        near_struct[fi_[within(pts[si], pts[fi_], prof.struct_excl)]] = True
    obj = classify_objects(pts, hag, F, free, prof, tree, near_struct)
    labels[free] = obj[free]
    labels[noise] = NOISE

    if prof.low_flat in ("ground", "class"):
        # flat, horizontal, low surfaces (steps, plinths, planters, paving edges, ground the surface fit sits below)
        # that no other rule claimed
        cand = np.flatnonzero((labels == UNCLASSIFIED) & (hag >= 0.0) & (hag <= prof.low_flat_max)
                              & (F["pla"] >= 0.4) & (F["nz"] >= 0.8))
        if len(cand):
            lab, _ = components(pts[cand], max(2 * prof.voxel, 0.2))
            for L, idx in _groups(lab):
                if len(idx) * prof.voxel**2 >= prof.low_flat_min_area:
                    labels[cand[idx]] = GROUND if prof.low_flat == "ground" else LOWFLAT
        log(f"low flat surfaces: {int((labels == (GROUND if prof.low_flat == 'ground' else LOWFLAT)).sum()):,} voxels")
    P(1.0, "done")
    return labels, hag.astype(np.float32)
