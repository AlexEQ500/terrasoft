"""Run the pipeline on the synthetic scene and print per-class precision / recall."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrasort import core as C
from terrasort.synth import make_scene

noise = float(sys.argv[2]) if len(sys.argv) > 2 else 0.012
xyz, truth = make_scene(noise_sd=noise)
print(f"{len(xyz):,} points")
prof = C.get_profile(sys.argv[1] if len(sys.argv) > 1 else "static_scan")
t = time.time()
pts, inv, v = C.voxelize(xyz, prof.voxel, prof.max_voxels)
print(f"voxels: {len(pts):,} (voxel {v:.3f} m)  [{time.time() - t:.1f}s]")
lab, hag = C.classify_points(pts, prof, prog=lambda f, m: print(f"  {f:4.0%} {m}"))
print(f"classified in {time.time() - t:.1f}s")
pred = lab[inv]

print(f"\n{'class':28s} {'truth':>8s} {'pred':>8s} {'precision':>10s} {'recall':>8s}")
for c, name in C.CLASS_NAMES.items():
    t_ = truth == c
    p_ = pred == c
    tp = (t_ & p_).sum()
    prec = tp / p_.sum() if p_.sum() else float("nan")
    rec = tp / t_.sum() if t_.sum() else float("nan")
    print(f"{name:28s} {t_.sum():8d} {p_.sum():8d} {prec:10.3f} {rec:8.3f}")

print("\nWhere each true class went (rows=truth, % of its points):")
cls = list(C.CLASS_NAMES)
print(f"{'':12s}" + "".join(f"{c:>7d}" for c in cls))
for c in cls:
    m = truth == c
    if not m.any():
        continue
    print(f"{C.CLASS_NAMES[c][:11]:12s}" + "".join(f"{(pred[m] == k).mean() * 100:7.1f}" for k in cls))
