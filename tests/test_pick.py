"""Click on one point of each synthetic object, check what the matching pick tool selects."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrasort import core as C
from terrasort.pick import pick_object
from terrasort.synth import make_scene

xyz, truth = make_scene(noise_sd=0.012)
prof = C.get_profile("static_scan")
pts, inv, v = C.voxelize(xyz, prof.voxel, prof.max_voxels)
store = {}
lab, hag = C.classify_points(pts, prof, store=store)
F = store["F"]
tree = cKDTree(pts)
# truth per voxel = majority of its points (use the first point's label is fine here)
vt = np.zeros(len(pts), np.uint8)
vt[inv] = truth

def click_at(xy, cls, z_hint=None):
    """voxel of truth class `cls` closest to xy (and z if given)"""
    idx = np.flatnonzero(vt == cls)
    d = np.linalg.norm(pts[idx, :2] - np.array(xy), axis=1)
    if z_hint is not None:
        d = d + np.abs(pts[idx, 2] - z_hint)
    return idx[np.argmin(d)]

z0 = lambda x: 100 + 0.02 * x
cases = [  # tool, truth class expected, click xy, z hint
    ("tree", C.TREE, (32, 16), z0(32) + 5.2),
    ("tree", C.TREE, (3, 21), z0(3) + 3.0),      # leafless tree, click a branch
    ("shrub", C.MID_VEG, (15, 22), None),
    ("building", C.BUILDING, (15, 26), z0(15) + 2.0),
    ("roof", C.BUILDING, (15, 29), z0(15) + 4.5),
    ("wall", C.WALL, (32, 29), None),
    ("ground", C.GROUND, (20, 3), None),
    ("kerb", C.KERB, (25, 12), None),
    ("car", C.CAR, (28, 5), z0(28) + 1.0),
    ("pole", C.POLE, (10, 13.5), z0(10) + 4.0),
    ("pole", C.POLE, (18, 14.5), z0(18) + 1.0),  # handrail
    ("person", C.PERSON, (20, 6), z0(20) + 1.0),
]
print(f"{'tool':9s} {'click on':22s} {'selected':>9s} {'precision':>10s} {'recall':>8s}  note")
for tool, cls, xy, zh in cases:
    i = click_at(xy, cls, zh)
    sel, target, msg = pick_object(tool, i, pts, hag, F, tree, v, prof=prof)
    t = vt == cls
    # recall measured against the truth object nearest the click (same-class objects elsewhere don't count)
    if tool == "roof":
        t = t & (pts[:, 1] > 26.3)                # roof part of the building
    near = np.linalg.norm(pts[:, :2] - pts[i, :2], axis=1)
    obj = t & (near < {"ground": 25, "building": 40, "roof": 30, "tree": 3.2, "kerb": 15}.get(tool, 6))
    s = np.zeros(len(pts), bool); s[sel] = True
    prec = (s & t).sum() / max(s.sum(), 1)
    rec = (s & obj).sum() / max(obj.sum(), 1)
    print(f"{tool:9s} {C.CLASS_NAMES[cls][:22]:22s} {s.sum():9d} {prec:10.3f} {rec:8.3f}  {msg if not s.any() else ''}")

# wrong-tool message: tree tool on a wall
i = click_at((15, 26), C.BUILDING, z0(15) + 2.0)
sel, _, msg = pick_object("tree", i, pts, hag, F, tree, v)
print("tree tool on a wall ->", len(sel), "|", msg)
