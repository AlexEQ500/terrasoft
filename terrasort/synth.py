"""Synthetic street scene with ground truth, for testing without real data.

Not a substitute for real scans: surfaces are cleaner than any real sensor produces.
"""
from __future__ import annotations

import numpy as np

from . import core as C


def make_scene(seed: int = 1, noise_sd: float = 0.012):
    rng = np.random.default_rng(seed)
    P, T = [], []  # points, truth labels

    def add(xyz, lab):
        P.append(xyz)
        T.append(np.full(len(xyz), lab, np.uint8))

    def gz(x):  # terrain height: 2 % fall along x
        return 100.0 + 0.02 * x

    # road (y 0..12) + pavement/verge (y 12..30), pavement 0.12 m above road
    n = 480_000
    x = rng.uniform(0, 40, n)
    y = rng.uniform(0, 30, n)
    z = gz(x) + np.where(y >= 12, 0.12, 0.0)
    add(np.stack([x, y, z], 1), C.GROUND)

    # kerb face at y = 12
    n = 3000 * 0 + int(40 * 0.12 * 3000)
    x = rng.uniform(0, 40, n)
    z = gz(x) + rng.uniform(0, 0.12, n)
    add(np.stack([x, np.full(n, 12.0), z], 1), C.KERB)

    # wall (building face) x 5..25 at y = 26, height 3.5 m above verge
    n = int(20 * 3.5 * 900)
    x = rng.uniform(5, 25, n)
    zz = rng.uniform(0, 3.5, n)
    add(np.stack([x, np.full(n, 26.0), gz(x) + 0.12 + zz], 1), C.BUILDING)

    # trees: trunk (surface) + canopy (volumetric)
    for tx, ty in [(32, 16), (35, 22), (8, 17), (38, 5.0)]:
        base = gz(tx) + (0.12 if ty >= 12 else 0.0)
        n = 3000
        a = rng.uniform(0, 2 * np.pi, n)
        h = rng.uniform(0, 3.2, n)
        add(np.stack([tx + 0.15 * np.cos(a), ty + 0.15 * np.sin(a), base + h], 1), C.TREE)
        n = 25000
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= rng.uniform(0, 1, (n, 1)) ** (1 / 3)
        c = np.array([tx, ty, base + 5.2])
        add(c + v * np.array([2.2, 2.2, 2.0]), C.TREE)

    # shrub (volumetric)
    n = 6000
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0, 1, (n, 1)) ** (1 / 3)
    add(np.array([15, 22, gz(15) + 0.12 + 0.55]) + v * np.array([0.7, 0.7, 0.5]), C.MID_VEG)

    # person on the road: cylinder body + head
    n = 7000
    a = rng.uniform(0, 2 * np.pi, n)
    h = rng.uniform(0, 1.55, n)
    r = 0.19
    px, py = 20.0, 6.0
    add(np.stack([px + r * np.cos(a), py + r * 0.7 * np.sin(a), gz(px) + h], 1), C.PERSON)
    n = 900
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    add(np.array([px, py, gz(px) + 1.65]) + v * 0.1, C.PERSON)

    # parked car (box) on the road
    for cx, cy, L, W, H in ((28.0, 5.0, 4.2, 1.8, 1.4), (23.7, 5.0, 4.2, 1.8, 1.5), (32.3, 5.0, 4.2, 1.75, 1.35)):  # bumper-to-bumper row
        nside = int(L * H * 1500)
        x = rng.uniform(cx - L / 2, cx + L / 2, nside)
        z = rng.uniform(0.2, H, nside)
        for yy in (cy - W / 2, cy + W / 2):
            add(np.stack([x, np.full(nside, yy), gz(x) + z], 1), C.CAR)
        nend = int(W * H * 1500)
        y = rng.uniform(cy - W / 2, cy + W / 2, nend)
        z = rng.uniform(0.2, H, nend)
        for xx in (cx - L / 2, cx + L / 2):
            add(np.stack([np.full(nend, xx), y, gz(xx) + z], 1), C.CAR)
        ntop = int(L * W * 1500)
        x = rng.uniform(cx - L / 2, cx + L / 2, ntop)
        y = rng.uniform(cy - W / 2, cy + W / 2, ntop)
        add(np.stack([x, y, gz(x) + H], 1), C.CAR)

    def tube(p0, p1, r, dens, lab):
        """Surface samples on a thin cylinder from p0 to p1."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        n = max(30, int(2 * np.pi * r * L * dens))
        t = rng.uniform(0, 1, n)
        ax = (p1 - p0) / L
        u = np.cross(ax, [0, 0, 1.0] if abs(ax[2]) < 0.9 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        w = np.cross(ax, u)
        a = rng.uniform(0, 2 * np.pi, n)
        pts = p0 + t[:, None] * (p1 - p0) + r * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * w)
        add(pts, lab)

    pav = lambda x: gz(x) + 0.12  # pavement level

    # solid fence panel: x 28..36 at y = 29, 1.8 m high; a shrub grows against it
    n = int(8 * 1.8 * 900)
    x = rng.uniform(28, 36, n)
    add(np.stack([x, np.full(n, 29.0), pav(x) + rng.uniform(0, 1.8, n)], 1), C.WALL)
    n = 5000
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0, 1, (n, 1)) ** (1 / 3)
    add(np.array([32.0, 28.2, pav(32) + 0.7]) + v * np.array([0.9, 0.7, 0.6]), C.MID_VEG)

    # lamp post: 6 m pole + small head
    lx, ly = 10.0, 13.5
    tube([lx, ly, pav(lx)], [lx, ly, pav(lx) + 6.0], 0.06, 2500, C.POLE)
    tube([lx, ly, pav(lx) + 6.0], [lx + 0.8, ly, pav(lx) + 6.1], 0.04, 2500, C.POLE)

    # handrail: 4 posts + horizontal rail at 1 m
    for px_ in (14.0, 16.7, 19.3, 22.0):
        tube([px_, 14.5, pav(px_)], [px_, 14.5, pav(px_) + 1.0], 0.025, 3000, C.POLE)
    # low flat slab (step / plinth): 3 x 1.4 m top at +0.35 m on the pavement, with its front face
    sx0, sy0, sw, sl, sh = 17.0, 18.5, 3.0, 1.4, 0.35
    n = int(sw * sl * 1500)
    add(np.stack([rng.uniform(sx0, sx0 + sw, n), rng.uniform(sy0, sy0 + sl, n), np.full(n, pav(sx0) + sh)], 1), C.LOWFLAT)
    n = int(sw * sh * 1500)
    add(np.stack([rng.uniform(sx0, sx0 + sw, n), np.full(n, sy0), pav(sx0) + rng.uniform(0, sh, n)], 1), C.LOWFLAT)

    tube([14.0, 14.5, pav(14) + 1.0], [22.0, 14.5, pav(22) + 1.0], 0.025, 3000, C.POLE)

    # leafless tree: trunk + branches + twigs (thin, line-like, but a tree)
    bx, by = 3.0, 21.0
    base = pav(bx)
    tube([bx, by, base], [bx, by, base + 3.5], 0.14, 1500, C.TREE)
    for _ in range(14):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) * 1.2 + 0.4
        d /= np.linalg.norm(d)
        s = np.array([bx, by, base + rng.uniform(2.5, 3.8)])
        e = s + d * rng.uniform(1.5, 2.5)
        tube(s, e, 0.03, 1500, C.TREE)
        for _ in range(6):
            t = rng.uniform(0.4, 1.0)
            s2 = s + (e - s) * t
            d2 = rng.normal(size=3)
            d2[2] = abs(d2[2])
            d2 /= np.linalg.norm(d2)
            tube(s2, s2 + d2 * rng.uniform(0.5, 1.0), 0.012, 1500, C.TREE)

    # pitched roof above the wall: from (y=26, z=3.5) rising to (y=31, z=5.5)
    n = int(20 * 5.4 * 500)
    x = rng.uniform(5, 25, n)
    s_ = rng.uniform(0, 1, n)
    add(np.stack([x, 26.0 + 5 * s_, gz(x) + 0.12 + 3.5 + 2.0 * s_], 1), C.BUILDING)

    xyz = np.concatenate(P)
    truth = np.concatenate(T)
    xyz += rng.normal(scale=noise_sd, size=xyz.shape)

    # sparse outliers
    n = 150
    out = np.stack([rng.uniform(0, 40, n), rng.uniform(0, 30, n), rng.uniform(95, 112, n)], 1)
    xyz = np.concatenate([xyz, out])
    truth = np.concatenate([truth, np.full(n, C.NOISE, np.uint8)])
    return xyz, truth


def write_las(path, xyz, truth=None):
    import laspy

    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.scales = [0.001, 0.001, 0.001]
    hdr.offsets = np.floor(xyz.min(0))
    las = laspy.LasData(hdr)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.write(path)
