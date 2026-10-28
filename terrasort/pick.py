"""Click-to-classify: each class has its own region-growing rule.

pick_object() takes the index of the clicked voxel and returns the indices of the whole object
the user means, chosen by the *tool* (= target class):

  tree      connected foliage-like points above ground (trunk to crown), not flat surfaces
  shrub     same, but small (low) vegetation
  building  connected planar wall + roof surfaces, including edges
  roof      connected planar surfaces that are not vertical and well above ground
  wall      connected vertical planar surfaces
  ground    connected points on the ground surface
  kerb      the low step band along the kerb line through the click
  car       connected non-foliage points of car height around the click
  pole      connected line-like points (posts, rails, lamp posts)
  person    small connected object up to head height around the click
  noise     everything within a small sphere around the click
  object    any connected object above the ground (class is chosen by the caller)

It needs the height above ground (hag) and the shape features F from the last classification.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import core as C


def _foliage(F, prof_like=None):
    return (F["sph"] >= 0.08) & (F["lin"] < 0.6) & (F["pla"] < 0.45)


# tool -> (target class, default reach in m, connection cell in voxels (min metres), candidate mask fn)
def _spec(tool, hag, F):
    pla, nz, lin = F["pla"], F["nz"], F["lin"]
    if tool == "tree":
        return C.TREE, 12.0, 3.0, 0.30, (hag >= 0.15) & (pla < 0.45)
    if tool == "shrub":
        return C.MID_VEG, 6.0, 3.0, 0.25, (hag >= 0.10) & (hag <= 3.0) & (pla < 0.45)
    if tool == "building":
        return C.BUILDING, 40.0, 2.0, 0.20, (hag >= 0.30) & (pla >= 0.40)
    if tool == "roof":
        return C.ROOF, 30.0, 2.0, 0.20, (hag >= 1.0) & (pla >= 0.40) & (nz >= 0.40)
    if tool == "wall":
        return C.WALL, 30.0, 2.0, 0.20, (hag >= 0.30) & (pla >= 0.40) & (nz <= 0.35)
    if tool == "ground":
        return C.GROUND, 25.0, 2.0, 0.15, (hag >= -0.30) & (hag <= 0.20)
    if tool == "kerb":
        return C.KERB, 15.0, 2.0, 0.12, (hag >= 0.02) & (hag <= 0.40)
    if tool == "car":
        return C.CAR, 6.0, 2.0, 0.20, (hag >= 0.05) & (hag <= 2.4) & ~_foliage(F)
    if tool == "pole":
        return C.POLE, 10.0, 3.0, 0.25, (hag >= 0.30) & (lin >= 0.4)
    if tool == "lowpole":
        return C.POLE_LOW, 6.0, 3.0, 0.20, (hag >= 0.20) & (hag <= 3.0) & (lin >= 0.4)
    if tool == "person":
        return C.PERSON, 3.0, 2.0, 0.15, (hag >= 0.10) & (hag <= 2.4)
    if tool == "furniture":
        return C.FURNITURE, 4.0, 2.0, 0.20, (hag >= 0.10) & (hag <= 2.5) & (pla < 0.9)
    if tool == "lowflat":
        return C.LOWFLAT, 6.0, 2.0, 0.20, (hag >= 0.0) & (hag <= 0.8) & (pla >= 0.4) & (nz >= 0.8)
    if tool == "sign":
        return C.SIGN, 4.0, 2.0, 0.20, hag >= 0.8
    if tool == "object":
        return None, 6.0, 3.0, 0.25, hag >= 0.15
    raise KeyError(tool)


def _pick_kerb(i, roi, pts, hag, voxel, prof):
    """Run the kerb step detector (a bit more sensitive than the automatic pass) around the click."""
    from dataclasses import replace

    prof = replace(prof or C.get_profile("static_scan"), kerb_min_step=0.04, kerb_min_len=0.8)
    sub = pts[roi]
    mask = C.find_kerbs(sub, hag[roi], np.zeros(len(roi), bool), prof, log=lambda *_: None)
    ki = np.flatnonzero(mask)
    if len(ki) == 0:
        return roi[:0], C.KERB, "no kerb step found near the click"
    d, j = cKDTree(sub[ki]).query(pts[i])
    if d > 0.5:
        return roi[:0], C.KERB, f"the clicked spot is not on a kerb step (nearest is {d:.1f} m away)"
    lab, _ = C.components(sub[ki], 0.2)
    sel = roi[ki[lab == lab[j]]]
    return np.unique(sel), C.KERB, f"{len(sel):,} voxels"


TOOLS = ["tree", "shrub", "building", "roof", "wall", "ground", "kerb", "car", "pole", "lowpole", "furniture", "lowflat", "sign", "person", "noise", "object"]


def pick_object(tool, i, pts, hag, F, tree, voxel, reach=0.0, prof=None):
    """Returns (indices, target_class, message). indices is empty if nothing suitable was found."""
    p0 = pts[i]
    if tool == "noise":
        r = reach or 0.4
        sel = np.array(tree.query_ball_point(p0, r=r), dtype=np.int64)
        return sel, C.NOISE, f"{len(sel):,} voxels within {r:.1f} m"

    target, R, cmult, cmin, cand = _spec(tool, hag, F)
    R = reach or R
    roi = np.array(tree.query_ball_point(p0, r=R), dtype=np.int64)
    if len(roi) == 0:
        return roi, target, "nothing nearby"
    if tool == "kerb":
        return _pick_kerb(i, roi, pts, hag, voxel, prof)
    m = cand[roi]
    ci = roi[m]
    if len(ci) == 0:
        return ci, target, f"no {tool}-like surface within {R:.0f} m of the click"
    # seed = nearest candidate to the click
    d, j = cKDTree(pts[ci]).query(p0)
    if d > max(0.4, 8 * voxel):
        return ci[:0], target, f"the clicked spot does not look like a {tool} (nearest {tool}-like point is {d:.1f} m away)"
    lab, _ = C.components(pts[ci], max(cmult * voxel, cmin))
    sel = ci[lab == lab[j]]

    # ---- per-class refinements
    if tool in ("building", "roof", "wall", "car", "pole", "lowpole", "furniture", "sign"):
        dil = {"building": (2.5, 0.25), "roof": (2.5, 0.8), "wall": (3.0, 0.05), "car": (3.0, 0.03),
               "pole": (1.6, 0.2), "lowpole": (1.6, 0.2), "furniture": (2.0, 0.08), "sign": (2.0, 0.5)}[tool]
        rest = roi[~np.isin(roi, sel)]
        rest = rest[hag[rest] >= dil[1]]
        if len(rest):
            dd, _ = cKDTree(pts[sel]).query(pts[rest], k=1, workers=-1)
            sel = np.concatenate([sel, rest[dd <= dil[0] * voxel]])
    elif tool == "kerb":
        near = sel[np.linalg.norm(pts[sel, :2] - p0[:2], axis=1) <= 4.0]
        if len(near) >= 5:
            xy = pts[near, :2] - pts[near, :2].mean(0)
            w, v = np.linalg.eigh(xy.T @ xy)
            u = v[:, 1]                       # direction of the kerb line
            rel = pts[sel, :2] - p0[:2]
            perp = np.abs(rel[:, 0] * -u[1] + rel[:, 1] * u[0])
            sel = sel[perp <= 0.35]
    elif tool == "person":
        sel = sel[np.linalg.norm(pts[sel, :2] - p0[:2], axis=1) <= 1.2]
    sel = np.unique(sel)
    return sel, target, f"{len(sel):,} voxels"
