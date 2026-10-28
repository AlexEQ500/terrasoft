"""Local web server: loads a LAS/LAZ, runs the classifier, serves arrays to the viewer."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import traceback
from dataclasses import fields
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from scipy.spatial import cKDTree

from . import core as C
from .pick import pick_object
from . import trees as T

VERSION = "2026-10-28 csv-edit"
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Terrasort")


@app.middleware("http")
async def _no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-store"
    return resp


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = "idle"  # idle | loading | classifying | ready | error
        self.progress = 0.0
        self.message = ""
        self.name = ""
        self.path: Path | None = None
        self.las = None
        self.xyz = None
        self.pts = None
        self.inv = None
        self.labels = None
        self.extra = None
        self.hag = None
        self.feat = None
        self.tree = None
        self.trees: list = []
        self.tid = None
        self.tree_next = 1
        self.rgb = None
        self.intensity = None
        self.center = np.zeros(3)
        self.voxel = 0.0
        self.profile = "static_scan"
        self.undo: list = []
        self.log: list[str] = []
        self.seconds = 0.0


S = Session()
WORK = Path(tempfile.gettempdir()) / "terrasort"
WORK.mkdir(exist_ok=True)


def _say(msg):
    S.log.append(msg)
    S.message = msg


def _build_working(profile_name: str, overrides: dict | None = None):
    prof = C.get_profile(profile_name, **(overrides or {}))
    xyz = S.xyz
    pts, inv, v = C.voxelize(xyz, prof.voxel, prof.max_voxels)
    n = len(pts)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    if S.rgb is not None:
        S.rgb_w = np.stack(
            [np.bincount(inv, weights=S.rgb[:, d], minlength=n) / cnt for d in range(3)], 1
        ).astype(np.uint8)
    else:
        S.rgb_w = None
    if S.intensity is not None:
        S.int_w = (np.bincount(inv, weights=S.intensity, minlength=n) / cnt).astype(np.uint8)
    else:
        S.int_w = None
    S.pts, S.inv, S.voxel = pts, inv, v
    S.center = pts.mean(0)
    S.labels = np.zeros(n, np.uint8)
    S.hag = np.zeros(n, np.float32)
    S.feat = None
    S.tree = None
    S.trees, S.tid, S.tree_next = [], None, 1
    S.undo.clear()
    return prof


def _load(path: Path):
    import laspy

    try:
        S.state, S.progress = "loading", 0.0
        _say(f"reading {path.name}")
        las = laspy.read(path)
        S.las = las
        S.name = path.name
        S.path = path
        S.xyz = np.stack([las.x, las.y, las.z], 1).astype(np.float64)
        dims = set(las.point_format.dimension_names)
        S.rgb = None
        if {"red", "green", "blue"} <= dims:
            r = np.stack([las.red, las.green, las.blue], 1).astype(np.float64)
            if r.max() > 0:
                S.rgb = np.clip(r / (256.0 if r.max() > 255 else 1.0), 0, 255)
        S.intensity = None
        if "intensity" in dims:
            it = np.asarray(las.intensity, np.float64)
            if it.max() > 0:
                hi = max(np.percentile(it[::max(1, len(it) // 2_000_000)], 99.5), 1.0)
                S.intensity = np.clip(it / hi, 0, 1) * 255.0
        _say("building preview")
        _build_working(S.profile)
        S.progress = 1.0
        S.state = "ready"
        _say(f"{len(S.xyz):,} points loaded, {len(S.pts):,} shown" + ("" if S.rgb is not None else " — no RGB colour in this file (all zero or missing)") + ("" if S.intensity is not None else " — no intensity"))
    except Exception as e:  # noqa: BLE001
        S.state = "error"
        _say(f"load failed: {e}")
        S.log.append(traceback.format_exc())


def _classify(profile: str, overrides: dict):
    try:
        t0 = time.time()
        S.state, S.progress = "classifying", 0.0
        prof = _build_working(profile, overrides)
        S.profile = profile

        def prog(f, m):
            S.progress = f
            S.message = m

        store: dict = {}
        S.labels, S.hag = C.classify_points(S.pts, prof, prog, log=_say, store=store)
        S.feat = store["F"]
        _say("indexing for click-to-pick")
        S.tree = cKDTree(S.pts)
        S.seconds = time.time() - t0
        S.state = "ready"
        _say(f"classified {len(S.pts):,} voxels in {S.seconds:.1f}s")
    except Exception as e:  # noqa: BLE001
        S.state = "error"
        _say(f"classification failed: {e}")
        S.log.append(traceback.format_exc())


def load_file(path: str | Path):
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(404, f"file not found: {p}")
    threading.Thread(target=_load, args=(p,), daemon=True).start()


# ------------------------------------------------------------------ API
@app.get("/api/status")
def status():
    return {
        "state": S.state,
        "progress": S.progress,
        "message": S.message,
        "name": S.name,
        "n_points": 0 if S.xyz is None else int(len(S.xyz)),
        "n_work": 0 if S.pts is None else int(len(S.pts)),
        "voxel": S.voxel,
        "profile": S.profile,
        "version": VERSION,
        "center": S.center.tolist(),
        "has_rgb": S.rgb is not None,
        "has_intensity": S.intensity is not None,
        "seconds": S.seconds,
        "can_undo": len(S.undo) > 0,
        "log": S.log[-8:],
    }


@app.get("/api/profiles")
def profiles():
    return {
        k: {f.name: getattr(v, f.name) for f in fields(v) if f.name != "name"} for k, v in C.PROFILES.items()
    }


@app.get("/api/classes")
def classes():
    return {str(k): v for k, v in C.CLASS_NAMES.items()}


@app.post("/api/upload")
async def upload(request: Request, name: str = "upload.las"):
    safe = Path(name).name
    dest = WORK / safe
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            f.write(chunk)
    load_file(dest)
    return {"ok": True}


@app.post("/api/open")
async def open_path(request: Request):
    body = await request.json()
    load_file(body["path"])
    return {"ok": True}


@app.post("/api/classify")
async def classify(request: Request):
    if S.xyz is None:
        raise HTTPException(400, "load a file first")
    if S.state in ("loading", "classifying"):
        raise HTTPException(409, "busy")
    body = await request.json()
    profile = body.get("profile", "static_scan")
    overrides = body.get("overrides", {})
    threading.Thread(target=_classify, args=(profile, overrides), daemon=True).start()
    return {"ok": True}


def _bytes(a: np.ndarray):
    return Response(content=np.ascontiguousarray(a).tobytes(), media_type="application/octet-stream")


@app.get("/api/pos")
def pos():
    return _bytes((S.pts - S.center).astype(np.float32))


@app.get("/api/labels")
def labels():
    return _bytes(S.labels)


@app.get("/api/hag")
def hag():
    return _bytes(S.hag)


@app.get("/api/rgb")
def rgb():
    if S.rgb_w is None:
        raise HTTPException(404)
    return _bytes(S.rgb_w)


@app.get("/api/intensity")
def intensity():
    if S.int_w is None:
        raise HTTPException(404)
    return _bytes(S.int_w)


@app.post("/api/relabel")
async def relabel(request: Request):
    body = await request.json()
    idx = np.asarray(body["indices"], np.int64)
    cls = int(body["cls"])
    if len(idx) == 0:
        return {"changed": 0}
    S.undo.append((idx, S.labels[idx].copy()))
    if len(S.undo) > 50:
        S.undo.pop(0)
    S.labels[idx] = cls
    return {"changed": int(len(idx))}


@app.post("/api/pick")
async def pick(request: Request):
    """Click-to-classify: grow the object under the click with the rule of the chosen tool.
    Returns the selection only (preview); the viewer applies it with /api/relabel once confirmed."""
    body = await request.json()
    if S.feat is None or S.tree is None:
        raise HTTPException(400, "classify first - picking needs the height and shape information from a classification run")
    sel, target, msg = pick_object(
        body["tool"], int(body["index"]), S.pts, S.hag, S.feat, S.tree, S.voxel, float(body.get("reach", 0) or 0),
        C.get_profile(S.profile),
    )
    if len(sel) == 0:
        return {"changed": 0, "message": msg}
    return {"changed": int(len(sel)), "cls": int(target if target is not None else C.UNCLASSIFIED),
            "indices": sel.tolist(), "message": msg}


# ------------------------------------------------------------------ tree inventory
def _need_class():
    if S.feat is None or S.pts is None:
        raise HTTPException(400, "classify first - the tree inventory needs the height above ground from a classification run")


def _retree():
    """Recompute heights / crowns / per-point tree ids after any change of the tree list."""
    idx_tid = T.assign(S.pts, S.hag, S.labels, S.trees)
    ids = np.array([t["id"] for t in S.trees] or [0], np.int32)
    S.tid = np.where(idx_tid > 0, ids[np.clip(idx_tid - 1, 0, len(ids) - 1)], 0).astype(np.int16)


def _tree_json():
    return [{"id": t["id"], "x": t["x"], "y": t["y"], "z": t["z"], "dbh": 2 * t["r"], "h": t.get("h", 0.0),
             "crown_r": t.get("crown_r", 0.0), "manual": bool(t["manual"]),
             "cx": t["x"] - float(S.center[0]), "cy": t["y"] - float(S.center[1]), "cz": t["z"] - float(S.center[2])}
            for t in S.trees]


@app.get("/api/trees")
def trees_list():
    return {"trees": _tree_json()}


@app.post("/api/trees/find")
async def trees_find(request: Request):
    _need_class()
    body = await request.json()
    areas = [[[x + float(S.center[0]), y + float(S.center[1])] for x, y in poly] for poly in (body.get("areas") or [])]
    found = T.find_stems(S.pts, S.hag, S.labels, S.voxel, float(body.get("dbh_min", 0.08)), float(body.get("dbh_max", 1.2)),
                         log=_say, areas=areas or None)
    manual = [t for t in S.trees if t["manual"]]
    found.sort(key=lambda t: (round(t["x"] / 5), t["y"]))
    S.trees = found + manual
    for k, t in enumerate(S.trees, 1):
        t["id"] = k
    _retree()
    short = [t for t in found if t.get("h", 0.0) < 2.5]       # no real crown above it: shrub, post, wall detail
    if short:
        S.trees = [t for t in S.trees if t not in short]
        found = [t for t in found if t not in short]
    for k, t in enumerate(S.trees, 1):       # numbering restarts at 1 on every run
        t["id"] = k
    S.tree_next = len(S.trees) + 1
    _retree()
    return {"trees": _tree_json(), "message": f"{len(found)} stems found" + (f" ({len(short)} dropped: no crown above)" if short else "") + (f", {len(manual)} manual kept" if manual else "")}


@app.post("/api/trees/add")
async def trees_add(request: Request):
    _need_class()
    body = await request.json()
    t, msg = T.fit_at(S.pts, S.hag, S.labels, int(body["index"]), float(body.get("radius", 0.7)), S.voxel)
    if t is None:
        return {"ok": False, "message": msg}
    t["id"] = S.tree_next
    S.tree_next += 1
    S.trees.append(t)
    _retree()
    return {"ok": True, "message": f"tree {t['id']}: {msg}", "trees": _tree_json()}


@app.post("/api/trees/crown")
async def trees_crown(request: Request):
    _need_class()
    body = await request.json()
    if not S.trees:
        return {"ok": False, "message": "no trees yet"}
    p = S.pts[int(body["index"]), :2]
    d = [float(np.hypot(t["x"] - p[0], t["y"] - p[1])) for t in S.trees]
    k = int(np.argmin(d))
    t = S.trees[k]
    t["crown_user"] = max(d[k], 0.3)
    _retree()
    return {"ok": True, "message": f"tree {t['id']}: crown radius set to {d[k]:.1f} m", "trees": _tree_json()}


@app.post("/api/trees/delete")
async def trees_delete(request: Request):
    body = await request.json()
    S.trees = [t for t in S.trees if t["id"] != int(body["id"])]
    _retree()
    return {"trees": _tree_json()}


@app.post("/api/trees/remove_last")
def trees_remove_last():
    if S.trees:
        last = max(range(len(S.trees)), key=lambda k: S.trees[k]["id"])
        S.trees.pop(last)
        _retree()
    return {"trees": _tree_json()}


@app.get("/api/tid")
def tid():
    t = S.tid if S.tid is not None else np.zeros(len(S.pts), np.int16)
    return _bytes(t.astype(np.int16))


def _hidden(hide: str):
    return {int(x) for x in hide.split(",") if x.strip().isdigit()}


@app.get("/api/trees.csv")
def trees_csv(hide: str = ""):
    return Response(T.to_csv(S.trees, _hidden(hide)), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{Path(S.name).stem}_trees.csv"'})


@app.get("/api/trees.dxf")
def trees_dxf(hide: str = ""):
    return _file(f"{Path(S.name).stem}_trees.dxf", T.to_dxf(S.trees, _hidden(hide)), "application/dxf")


def _ground_pts(classes: str, csv: int = 0, prio: int = 1):
    """Ground-group points of the scan, plus the loaded CSV points when csv=1. Returns (points, priority flags)."""
    if S.labels is None or S.pts is None:
        raise HTTPException(400, "load and classify a file first")
    ids = [int(c) for c in classes.split(",") if c.strip()] or [C.GROUND]
    m = np.isin(S.labels, ids)
    g = S.pts[m]
    flags = np.zeros(len(g), bool)
    ex = getattr(S, "extra", None)
    if csv and ex is not None and len(ex):
        g = np.vstack([g, ex])
        flags = np.r_[flags, np.full(len(ex), bool(prio))]
    if len(g) < 10:
        raise HTTPException(400, "no ground points: classify first, or tick more ground classes")
    return g, flags


@app.post("/api/csv/preview")
async def csv_preview(request: Request):
    from . import dtm

    try:
        return dtm.csv_preview((await request.body()).decode("utf-8-sig", "replace"))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/csv/load")
async def csv_load(request: Request):
    from . import dtm

    b = await request.json()
    try:
        P = dtm.parse_points(b["text"], int(b["x"]), int(b["y"]), int(b["z"]), b.get("delim", ","), bool(b.get("header")))
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    S.extra = P
    warn = ""
    if S.pts is not None and len(S.pts):
        lo, hi = S.pts.min(0)[:2] - 50, S.pts.max(0)[:2] + 50
        inside = ((P[:, :2] >= lo) & (P[:, :2] <= hi)).all(1).mean()
        if inside < 0.5:
            warn = f"Only {inside:.0%} of the CSV points are within 50 m of the scan: check the X/Y columns and the coordinate system."
    return {"n": len(P), "warn": warn, "abs": P.ravel().tolist(), "zrange": [float(P[:, 2].min()), float(P[:, 2].max())]}


@app.post("/api/csv/delete")
async def csv_delete(request: Request):
    b = await request.json()
    ex = getattr(S, "extra", None)
    if ex is None:
        raise HTTPException(400, "no CSV points loaded")
    idx = np.unique(np.asarray(b.get("idx", []), np.int64))
    idx = idx[(idx >= 0) & (idx < len(ex))]
    S.extra = np.delete(ex, idx, axis=0)
    return {"n": int(len(S.extra))}


@app.post("/api/csv/add")
async def csv_add(request: Request):
    b = await request.json()
    P = np.asarray(b.get("pts", []), np.float64).reshape(-1, 3)
    ex = getattr(S, "extra", None)
    S.extra = P if ex is None else np.vstack([ex, P])
    return {"n": int(len(S.extra))}


@app.post("/api/csv/clear")
def csv_clear():
    S.extra = None
    return {"ok": True}


_GRID = {"key": None, "val": None}


def _grid(classes, cell, gap, csv=0, prio=1):
    """DTM grid for the ground group; the last one is cached so DTM, contours and the viewer overlay share it."""
    from . import dtm

    g, flags = _ground_pts(classes, csv, prio)
    key = (classes, round(cell, 4), round(gap, 3), len(g), tuple(np.round(g.sum(0), 3)), int(flags.sum()))
    if _GRID["key"] != key:
        _GRID["val"] = dtm.ground_grid(g, max(0.02, cell), gap, prio=flags)
        _GRID["key"] = key
    return _GRID["val"]


def _file(name, data, media):
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/dtm.tif")
def dtm_tif(classes: str = "2,64,73", cell: float = 0.25, gap: float = 10.0, epsg: int = 0, kind: str = "elev", csv: int = 0, prio: int = 1):
    from . import dtm

    Z, x0, yt, cs = _grid(classes, cell, gap, csv, prio)
    tag = "dtm" if kind == "elev" else "dtm_shaded"
    return _file(f"{Path(S.name).stem}_{tag}.tif", dtm.tiff_bytes(Z, x0, yt, cs, kind, epsg or None), "image/tiff")


@app.get("/api/dtm.tfw")
def dtm_tfw(classes: str = "2,64,73", cell: float = 0.25, gap: float = 10.0, kind: str = "elev", csv: int = 0, prio: int = 1):
    from . import dtm

    Z, x0, yt, cs = _grid(classes, cell, gap, csv, prio)
    tag = "dtm" if kind == "elev" else "dtm_shaded"
    return _file(f"{Path(S.name).stem}_{tag}.tfw", dtm.world_file(x0, yt, cs), "text/plain")


def _contour_lines(classes, cell, gap, interval, major, smooth, csv=0, prio=1):
    from . import dtm

    Z, x0, yt, cs = _grid(classes, cell, gap, csv, prio)
    return dtm.contours(dtm.smooth(Z, smooth), x0, yt, cs, max(0.01, interval), major)


@app.get("/api/contours.dxf")
def contours_dxf(classes: str = "2,64,73", cell: float = 0.25, gap: float = 10.0, interval: float = 0.25,
                 major: int = 5, smooth: float = 1.0, csv: int = 0, prio: int = 1):
    from . import dtm

    lines = _contour_lines(classes, cell, gap, interval, major, smooth, csv, prio)
    return _file(f"{Path(S.name).stem}_contours.dxf", dtm.contours_dxf(lines), "application/dxf")


@app.get("/api/ground_view/dtm")
def ground_view_dtm(classes: str = "2,64,73", cell: float = 0.25, gap: float = 10.0, csv: int = 0, prio: int = 1):
    """Raw float32 grid (NaN = no data) for drawing the surface in the viewer; geometry in the headers."""
    Z, x0, yt, cs = _grid(classes, cell, gap, csv, prio)
    ny, nx = Z.shape
    return Response(np.ascontiguousarray(Z, dtype="<f4").tobytes(), media_type="application/octet-stream",
                    headers={"X-Grid": json.dumps({"nx": nx, "ny": ny, "x0": x0 - float(S.center[0]), "ytop": yt - float(S.center[1]),
                                                  "cell": cs, "zc": float(S.center[2])}),
                             "Access-Control-Expose-Headers": "X-Grid"})


@app.get("/api/ground_view/contours")
def ground_view_contours(classes: str = "2,64,73", cell: float = 0.25, gap: float = 10.0, interval: float = 0.25,
                         major: int = 5, smooth: float = 1.0, csv: int = 0, prio: int = 1):
    """Contours as line segments in viewer coordinates: flat float32 [x1,y1,z1,x2,y2,z2,...] per kind."""
    lines = _contour_lines(classes, cell, gap, interval, major, smooth, csv, prio)
    out = {"minor": [], "major": []}
    c = S.center
    for lv, mj, xy in lines:
        if len(xy) < 2:
            continue
        a = np.c_[xy[:-1, 0] - c[0], xy[:-1, 1] - c[1], np.full(len(xy) - 1, lv - c[2]), xy[1:, 0] - c[0], xy[1:, 1] - c[1],
                  np.full(len(xy) - 1, lv - c[2])]
        out["major" if mj else "minor"].append(a.astype("<f4"))
    flat = {k: (np.concatenate(v).ravel() if v else np.zeros(0, "<f4")) for k, v in out.items()}
    head = np.array([len(flat["minor"]), len(flat["major"])], "<u4").tobytes()
    return Response(head + flat["minor"].tobytes() + flat["major"].tobytes(), media_type="application/octet-stream")


@app.post("/api/section.dxf")
async def section_dxf(request: Request):
    """Section strip as a CAD drawing: X = chainage, Y = level x vertical exaggeration, points coloured by class or RGB."""
    from . import dtm

    if S.pts is None or S.labels is None:
        raise HTTPException(400, "load a file first")
    b = await request.json()
    A = np.asarray(b["a"], float) + S.center
    B = np.asarray(b["b"], float) + S.center
    hw = max(0.05, float(b.get("width", 1.0))) / 2
    ex = float(b.get("vex", 1.0))
    use_rgb = b.get("color") == "rgb" and S.rgb_w is not None
    hidden = set(int(h) for h in b.get("hidden", []))
    cols = {int(k): int(v) for k, v in (b.get("colors") or {}).items()}
    names = {int(k): str(v) for k, v in (b.get("names") or {}).items()}
    d = B[:2] - A[:2]
    L = float(np.hypot(*d)) or 1e-6
    u = d / L
    rel = S.pts[:, :2] - A[:2]
    al = rel @ u
    off = -rel[:, 0] * u[1] + rel[:, 1] * u[0]
    m = (al >= 0) & (al <= L) & (np.abs(off) <= hw)
    if hidden:
        m &= ~np.isin(S.labels, list(hidden))
    ii = np.flatnonzero(m)
    if not len(ii):
        raise HTTPException(400, "no points inside the strip")
    doc = dtm.new_doc([("SEC_GROUND_PROFILE", 1), ("SEC_MEASURE", 2), ("SEC_BASELINE", 7), ("SEC_TEXT", 7)])
    msp = doc.modelspace()
    seen = set()
    for i in ii.tolist():
        c = int(S.labels[i])
        ln = "SEC_PTS_" + "".join(ch if ch.isalnum() else "_" for ch in names.get(c, str(c)).upper())[:40]
        if ln not in seen:
            seen.add(ln)
            doc.layers.add(ln, color=7)
        if use_rgb:
            r, g, bl = (int(v) for v in S.rgb_w[i])
            tc = (r << 16) | (g << 8) | bl
        else:
            tc = cols.get(c, 0xFFFFFF)
        msp.add_point((float(al[i]), float((S.pts[i, 2]) * ex), 0.0), dxfattribs={"layer": ln, "true_color": tc})
    gi = ii[np.isin(S.labels[ii], [C.GROUND, C.KERB, C.LOWFLAT])]
    if len(gi) > 3:
        bins = np.round(al[gi] / 0.1).astype(int)
        order = np.argsort(bins, kind="stable")
        bs, zs = bins[order], S.pts[gi[order], 2]
        ub, st = np.unique(bs, return_index=True)
        med = [float(np.median(z)) for z in np.split(zs, st[1:])]
        msp.add_lwpolyline([(float(x * 0.1), float(z * ex)) for x, z in zip(ub, med)], dxfattribs={"layer": "SEC_GROUND_PROFILE"})
    meas = b.get("measure") or []
    if len(meas) == 2:
        msp.add_lwpolyline([(float(p[0]), float((p[1] + S.center[2]) * ex)) for p in meas], dxfattribs={"layer": "SEC_MEASURE"})
    zmin = float(S.pts[ii, 2].min())
    msp.add_lwpolyline([(0.0, zmin * ex - 0.5), (L, zmin * ex - 0.5)], dxfattribs={"layer": "SEC_BASELINE"})
    msp.add_text(f"Section A-B  length {L:.2f} m  strip {hw * 2:.2f} m  vertical x{ex:g}  (X = chainage, Y = level)", height=0.3,
                 dxfattribs={"layer": "SEC_TEXT", "insert": (0.0, zmin * ex - 1.0)})
    return _file(f"{Path(S.name).stem}_section.dxf", dtm.doc_bytes(doc), "application/dxf")


@app.post("/api/section_line.dxf")
async def section_line_dxf(request: Request):
    """The section line and strip in plan view, real coordinates."""
    from . import dtm

    b = await request.json()
    A = np.asarray(b["a"], float) + S.center
    B = np.asarray(b["b"], float) + S.center
    hw = max(0.05, float(b.get("width", 1.0))) / 2
    d = B[:2] - A[:2]
    L = float(np.hypot(*d)) or 1e-6
    u = d / L
    n = np.array([-u[1], u[0]]) * hw
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    doc = dtm.new_doc([("SECTION_LINE", 1), ("SECTION_STRIP", 8), ("SECTION_TEXT", 7)])
    msp = doc.modelspace()
    msp.add_line(tuple(map(float, A)), tuple(map(float, B)), dxfattribs={"layer": "SECTION_LINE"})
    msp.add_lwpolyline([tuple(map(float, p)) for p in (A[:2] + n, B[:2] + n, B[:2] - n, A[:2] - n)], close=True,
                       dxfattribs={"layer": "SECTION_STRIP"})
    msp.add_text("A", height=0.5, dxfattribs={"layer": "SECTION_TEXT", "insert": tuple(map(float, A[:2] - u * 1.2))})
    msp.add_text("B", height=0.5, dxfattribs={"layer": "SECTION_TEXT", "insert": tuple(map(float, B[:2] + u * 0.4))})
    mid = (A[:2] + B[:2]) / 2 + n * 1.5 / max(hw, 1e-6) * hw
    msp.add_text(f"Section A-B  {L:.2f} m", height=0.35, rotation=ang, dxfattribs={"layer": "SECTION_TEXT", "insert": tuple(map(float, mid))})
    return _file(f"{Path(S.name).stem}_section_line.dxf", dtm.doc_bytes(doc), "application/dxf")


@app.post("/api/undo")
def undo():
    if not S.undo:
        return {"changed": 0}
    idx, old = S.undo.pop()
    S.labels[idx] = old
    return {"changed": int(len(idx)), "indices": idx.tolist(), "labels": old.tolist()}


@app.get("/api/export")
def export():
    import laspy

    if S.las is None or S.inv is None:
        raise HTTPException(400, "nothing to export")
    full = S.labels[S.inv]
    las = S.las
    if las.header.point_format.id < 6:
        las = laspy.convert(las, point_format_id=7 if S.rgb is not None else 6)
    las.classification = full
    out = WORK / (Path(S.name).stem + "_classified.las")
    las.write(out)
    return FileResponse(out, filename=out.name, media_type="application/octet-stream")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
