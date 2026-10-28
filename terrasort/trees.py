"""Tree inventory (stems, diameter at breast height, height, ground level) on a classified cloud.

find_stems()    automatic: slice the cloud at breast height (1.0-1.6 m above ground), cluster the slice in plan,
                fit a circle to each cluster, keep round, tree-like, vertically continuous ones
                (walls and fences are long and straight, so they fail the circle test).
fit_at()        manual: fit a circle around one clicked spot.
assign()        give every tree-like point to its nearest stem; height / crown radius per tree.
to_csv(), to_dxf()  export the list for CAD / spreadsheets.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import core as C

SLICE_LO, SLICE_HI = 1.0, 1.6          # DBH is measured in this height band above local ground
NOT_STEM = {C.GROUND, C.BUILDING, C.WALL, C.ROOF, C.CAR, C.KERB, C.NOISE, C.PERSON, C.POLE, C.POLE_LOW,
            C.FURNITURE, C.SIGN}


def fit_circle(xy: np.ndarray, iters: int = 4):
    """Robust algebraic circle fit. Returns (cx, cy, r, rms, arc_deg, n_used) or None."""
    p = xy
    sol = None
    for _ in range(iters):
        if len(p) < 6:
            return None
        c0 = p.mean(0)
        q = p - c0
        A = np.c_[2 * q, np.ones(len(q))]
        b = (q ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[:2]
        r2 = sol[2] + cx * cx + cy * cy
        if r2 <= 1e-8:
            return None
        r = float(np.sqrt(r2))
        res = np.abs(np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r)
        keep = res <= max(2.5 * float(np.median(res)), 0.012)
        if keep.all() or keep.sum() < 6:
            p, q = p, q
            break
        p = p[keep]
    c0 = p.mean(0)
    q = p - c0
    A = np.c_[2 * q, np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
    cx, cy = sol[:2]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 1e-8:
        return None
    r = float(np.sqrt(r2))
    res = np.hypot(q[:, 0] - cx, q[:, 1] - cy) - r
    rms = float(np.sqrt(np.mean(res ** 2)))
    ang = np.sort(np.arctan2(q[:, 1] - cy, q[:, 0] - cx))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    arc = float(360.0 - np.degrees(gaps.max()))
    return float(c0[0] + cx), float(c0[1] + cy), r, rms, arc, len(p)


def _ground_z(pts, hag, sel):
    return float(np.median(pts[sel, 2] - hag[sel]))


def inside_any(xy, polys):
    """Boolean mask: which xy points lie inside at least one polygon (even-odd rule)."""
    out = np.zeros(len(xy), bool)
    for poly in polys:
        P = np.asarray(poly, float)
        if len(P) < 3:
            continue
        x, y = xy[:, 0], xy[:, 1]
        inside = np.zeros(len(xy), bool)
        j = len(P) - 1
        for i in range(len(P)):
            xi, yi, xj, yj = P[i, 0], P[i, 1], P[j, 0], P[j, 1]
            cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi)
            inside ^= cond
            j = i
        out |= inside
    return out


def find_stems(pts, hag, labels, voxel, dbh_min=0.08, dbh_max=1.2, log=print, areas=None):
    sl = np.flatnonzero((hag >= SLICE_LO) & (hag <= SLICE_HI) & ~np.isin(labels, list(NOT_STEM)))
    if areas:
        sl = sl[inside_any(pts[sl, :2], areas)]
    if len(sl) < 20:
        return []
    cell = max(2.5 * voxel, 0.12)
    flat = np.c_[pts[sl, 0], pts[sl, 1], np.zeros(len(sl))]
    lab, _ = C.components(flat, cell)
    tree_pts = np.flatnonzero(np.isin(labels, [C.TREE, C.MID_VEG]) & (hag >= 1.5))
    tree_kd = cKDTree(pts[tree_pts, :2]) if len(tree_pts) else None
    below = np.flatnonzero((hag >= 0.35) & (hag < SLICE_LO) & ~np.isin(labels, list(NOT_STEM - {C.POLE, C.POLE_LOW})))
    above = np.flatnonzero((hag > SLICE_HI + 0.1) & (hag <= 2.8) & ~np.isin(labels, list(NOT_STEM - {C.POLE, C.POLE_LOW})))
    kd_b = cKDTree(pts[below, :2]) if len(below) else None
    kd_a = cKDTree(pts[above, :2]) if len(above) else None
    # what surrounds a candidate within 1 m at stem height: must look like a tree, not a building / wall / car
    ring = np.flatnonzero((hag >= 0.3) & (hag <= 2.6))
    ring_kd = cKDTree(pts[ring, :2]) if len(ring) else None
    TREE_LIKE = [C.TREE, C.MID_VEG, C.LOW_VEG]
    STRUCT = [C.BUILDING, C.WALL, C.ROOF, C.CAR, C.KERB, C.FURNITURE, C.SIGN]
    out = []
    for L, idx in C._groups(lab):
        if len(idx) < 8:
            continue
        sel = sl[idx]
        major, _minor = C._xy_extent(pts[sel, :2])
        if major > dbh_max * 2.6 + 0.3:      # too long: wall, fence, hedge, bench
            continue
        f = fit_circle(pts[sel, :2])
        if f is None:
            continue
        cx, cy, r, rms, arc, nu = f
        if not (dbh_min / 2 <= r <= dbh_max / 2):
            continue
        if arc < 100 or rms > max(0.22 * r, 1.5 * voxel):
            continue
        rr = r + 0.15
        nb = len(kd_b.query_ball_point([cx, cy], rr)) if kd_b is not None else 0
        na = len(kd_a.query_ball_point([cx, cy], rr)) if kd_a is not None else 0
        if not ((nb >= 3 and na >= 1) or (na >= 3 and nb >= 1)):
            continue                               # not a vertical stem
        if tree_kd is None or len(tree_kd.query_ball_point([cx, cy], 3.5)) < 40:
            continue                               # no crown right above: pole, post, sign, wall detail
        if ring_kd is not None:
            nb_i = ring[np.array(ring_kd.query_ball_point([cx, cy], 1.0), dtype=np.int64)]
            if len(nb_i):
                lb = labels[nb_i]
                if np.isin(lb, STRUCT).mean() > 0.25 or np.isin(lb, TREE_LIKE).mean() < 0.4:
                    continue                       # surrounded by building / wall / unclassified clutter
        out.append(dict(x=cx, y=cy, z=_ground_z(pts, hag, sel), r=r, arc=arc, manual=False, crown_r=0.0))
    # drop duplicates (two clusters of one stem)
    out.sort(key=lambda t: -t["arc"])
    keep = []
    for t in out:
        if all(np.hypot(t["x"] - k["x"], t["y"] - k["y"]) > max(0.3, k["r"] + t["r"]) for k in keep):
            keep.append(t)
    log(f"{len(keep)} stems found")
    return keep


def fit_at(pts, hag, labels, i, search=0.7, voxel=0.05):
    """Manual circle at the clicked point: DBH from the 1.0-1.6 m band around it. Returns (tree dict | None, message)."""
    p0 = pts[i, :2]
    kd = cKDTree(pts[:, :2])
    near = np.array(kd.query_ball_point(p0, search), dtype=np.int64)
    near = near[(hag[near] >= SLICE_LO) & (hag[near] <= SLICE_HI) & (labels[near] != C.GROUND) & (labels[near] != C.NOISE)]
    if len(near) < 6:
        return None, "not enough points at breast height (1.0-1.6 m above ground) around the click; click the trunk or raise the search radius"
    f = fit_circle(pts[near, :2])
    if f is None:
        return None, "could not fit a circle here"
    cx, cy, r, rms, arc, _ = f
    if r > 1.5 or r < 0.02:
        return None, f"fitted diameter {2 * r:.2f} m is not a trunk; click closer to the trunk"
    msg = f"diameter {2 * r:.2f} m"
    if arc < 90 or rms > max(0.3 * r, 2 * voxel):
        msg += " (weak fit: trunk only partly visible)"
    return dict(x=cx, y=cy, z=_ground_z(pts, hag, near), r=r, arc=arc, manual=True, crown_r=0.0), msg


def assign(pts, hag, labels, trees, max_reach=10.0):
    """Per-point tree id (1-based index into trees; 0 = none). Fills h, crown_r, n_pts in each tree dict."""
    tid = np.zeros(len(pts), np.int32)
    if not trees:
        return tid
    cand = np.flatnonzero(np.isin(labels, [C.TREE, C.MID_VEG]) & (hag >= 0.3))
    if len(cand) == 0:
        for t in trees:
            t.update(h=0.0, n_pts=0)
        return tid
    xy = np.array([[t["x"], t["y"]] for t in trees])
    d, j = cKDTree(xy).query(pts[cand, :2], k=1, workers=-1)
    reach = np.array([t.get("crown_user", 0.0) or max_reach for t in trees])
    ok = d <= reach[j]
    cand, j, d = cand[ok], j[ok], d[ok]
    tid[cand] = j + 1
    for k, t in enumerate(trees):
        m = j == k
        if not m.any():
            t.update(h=0.0, n_pts=0)
            t["crown_r"] = t.get("crown_user", 0.0) or 0.0
            continue
        h = hag[cand[m]]
        t["h"] = float(np.percentile(h, 99.8)) if len(h) > 20 else float(h.max())
        t["n_pts"] = int(m.sum())
        t["crown_r"] = float(t.get("crown_user", 0.0) or np.percentile(d[m], 92))
    return tid


def to_csv(trees, hide=()):
    rows = ["id,x,y,ground_z,dbh_m,height_m,crown_radius_m,manual"]
    for t in trees:
        if t["id"] in hide:
            continue
        rows.append(f'{t["id"]},{t["x"]:.3f},{t["y"]:.3f},{t["z"]:.3f},{2 * t["r"]:.3f},{t.get("h", 0):.2f},{t.get("crown_r", 0):.2f},{int(t["manual"])}')
    return "\n".join(rows) + "\n"


def to_dxf(trees, hide=()) -> bytes:
    """AutoCAD-safe DXF (via ezdxf): trunk circle at ground level + crown circle + label text + ground point per tree."""
    from . import dtm

    doc = dtm.new_doc([("TREE_STEM", 1), ("TREE_CROWN", 3), ("TREE_GROUND", 2), ("TREE_LABEL", 7)])
    msp = doc.modelspace()
    for t in trees:
        if t["id"] in hide:
            continue
        x, y, z = float(t["x"]), float(t["y"]), float(t["z"])
        msp.add_circle((x, y, z), float(t["r"]), dxfattribs={"layer": "TREE_STEM"})
        if t.get("crown_r", 0) > 0:
            msp.add_circle((x, y, z), float(t["crown_r"]), dxfattribs={"layer": "TREE_CROWN"})
        msp.add_point((x, y, z), dxfattribs={"layer": "TREE_GROUND"})
        msp.add_text(f"{t['id']}  D{2 * t['r']:.2f}  H{t.get('h', 0):.1f}", height=0.35,
                     dxfattribs={"layer": "TREE_LABEL", "insert": (x + float(t["r"]) + 0.2, y, z)})
    return dtm.doc_bytes(doc)