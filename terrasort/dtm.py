"""Ground model: DTM grid -> GeoTIFF and contour lines -> DXF.

Only numpy and scipy are needed (no GDAL / matplotlib): the GeoTIFF and the DXF are written by hand.
"""
from __future__ import annotations

import struct

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

NODATA = -9999.0


def ground_grid(pts: np.ndarray, cell: float, max_gap: float = 10.0, max_cells: int = 25_000_000, prio: np.ndarray | None = None):
    """Interpolate a regular grid (TIN, linear) from ground points.

    Returns (Z, x0, y_top, cell): Z[row, col] with row 0 at the top (north); NaN = no data.
    Cell centres are at x0 + (col + 0.5) * cell and y_top - (row + 0.5) * cell.
    """
    if len(pts) < 4:
        raise ValueError("not enough ground points")
    mn, mx = pts[:, :2].min(0), pts[:, :2].max(0)
    while ((mx[0] - mn[0]) / cell + 1) * ((mx[1] - mn[1]) / cell + 1) > max_cells:
        cell *= 1.25
    x0, y0 = np.floor(mn / cell) * cell
    nx = int(np.ceil((mx[0] - x0) / cell)) + 1
    ny = int(np.ceil((mx[1] - y0) / cell)) + 1
    # average the points in each cell first: far fewer points for the triangulation, same surface
    ci = np.floor((pts[:, :2] - [x0, y0]) / cell).astype(np.int64)
    key = ci[:, 1] * nx + ci[:, 0]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    w = np.ones(len(pts))
    if prio is not None and prio.any():
        # a cell that holds surveyed (CSV) points uses only those: the scan points in it get zero weight
        has = np.bincount(inv, weights=prio.astype(float), minlength=len(uk)) > 0
        w = np.where(has[inv] & ~prio, 0.0, 1.0)
    sw = np.maximum(np.bincount(inv, weights=w), 1e-12)
    bx = np.bincount(inv, weights=pts[:, 0] * w) / sw
    by = np.bincount(inv, weights=pts[:, 1] * w) / sw
    bz = np.bincount(inv, weights=pts[:, 2] * w) / sw
    xy = np.c_[bx, by]
    gx = x0 + (np.arange(nx) + 0.5) * cell
    gy = y0 + (np.arange(ny) + 0.5) * cell
    GX, GY = np.meshgrid(gx, gy)
    Z = LinearNDInterpolator(xy, bz)(GX, GY)
    if max_gap and max_gap > 0:
        d, _ = cKDTree(xy).query(np.c_[GX.ravel(), GY.ravel()], k=1, workers=-1)
        Z[(d > max_gap).reshape(Z.shape)] = np.nan
    Z = Z[::-1]  # north up
    return Z.astype(np.float32), float(x0), float(y0 + ny * cell), float(cell)


def smooth(Z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (no-data) cells."""
    if sigma <= 0:
        return Z
    ok = np.isfinite(Z)
    w = gaussian_filter(ok.astype(np.float32), sigma)
    s = gaussian_filter(np.where(ok, Z, 0).astype(np.float32), sigma)
    out = np.where(ok & (w > 1e-3), s / np.maximum(w, 1e-3), np.nan)
    return out.astype(np.float32)


# ------------------------------------------------------------------ TIFF (+ world file)
def hillshade_rgba(Z: np.ndarray, cell: float) -> np.ndarray:
    """8-bit RGBA picture of the surface: elevation colours x hillshade; no-data is transparent."""
    ok = np.isfinite(Z)
    z = np.where(ok, Z, np.nanmean(Z))
    gy, gx = np.gradient(z, cell)
    az, alt = np.radians(315.0), np.radians(45.0)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    hs = np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)
    t = (z - np.nanmin(Z)) / max(np.nanmax(Z) - np.nanmin(Z), 1e-6)
    stops = np.array([[0.00, 0.20, 0.35, 0.80], [0.35, 0.20, 0.75, 0.40], [0.65, 0.95, 0.90, 0.40], [1.00, 0.90, 0.25, 0.20]])
    rgb = np.stack([np.interp(t, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], -1)
    rgb = rgb * (0.35 + 0.65 * hs)[..., None]
    out = np.zeros(Z.shape + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(ok, 255, 0)
    return out


def world_file(x0: float, y_top: float, cell: float) -> str:
    """ESRI world file (.tfw): pixel size, rotation terms, and the centre of the top-left pixel."""
    return f"{cell:.6f}\n0.000000\n0.000000\n{-cell:.6f}\n{x0 + cell / 2:.4f}\n{y_top - cell / 2:.4f}\n"


def tiff_bytes(Z: np.ndarray, x0: float, y_top: float, cell: float, kind: str = "elev", epsg: int | None = None) -> bytes:
    """Uncompressed north-up TIFF. kind="elev": 1 band float32, no-data -9999. kind="shade": 8-bit RGBA picture.
    The georeference is also stored in the file (scale + tie point) but the coordinate system is only tagged if epsg is given;
    the matching world file comes from world_file()."""
    ny, nx = Z.shape
    if kind == "shade":
        data = np.ascontiguousarray(hillshade_rgba(Z, cell)).tobytes()
        spp, bits, fmt, photo = 4, [8, 8, 8, 8], [1, 1, 1, 1], 2
    else:
        data = np.where(np.isfinite(Z), Z, NODATA).astype("<f4").tobytes()
        spp, bits, fmt, photo = 1, [32], [3], 1
    scale = struct.pack("<3d", cell, cell, 0.0)
    tie = struct.pack("<6d", 0.0, 0.0, 0.0, x0, y_top, 0.0)
    blobs, off = [], 0

    E = [(256, 4, 1, nx), (257, 4, 1, ny), (259, 3, 1, 1), (262, 3, 1, photo), (277, 3, 1, spp), (278, 4, 1, ny),
         (279, 4, 1, len(data)), (33550, 12, 3, scale), (33922, 12, 6, tie)]
    E.append((258, 3, len(bits), struct.pack(f"<{len(bits)}H", *bits)))
    E.append((339, 3, len(fmt), struct.pack(f"<{len(fmt)}H", *fmt)))
    if kind == "shade":
        E.append((338, 3, 1, 2))  # ExtraSamples: unassociated alpha
    else:
        E.append((42113, 2, 6, b"-9999\x00"))
    if epsg:
        keys = [1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, int(epsg)]
        E.append((34735, 3, len(keys), struct.pack(f"<{len(keys)}H", *keys)))
    E.append((273, 4, 1, 0))  # strip offset, patched below
    E.sort(key=lambda e: e[0])
    ifd_size = 2 + len(E) * 12 + 4
    off = 8 + ifd_size
    entries = []
    for tag, typ, cnt, val in E:
        if isinstance(val, bytes) and len(val) > 4:
            entries.append((tag, typ, cnt, off))
            blobs.append(val + (b"\x00" if len(val) & 1 else b""))
            off += len(blobs[-1])
        elif isinstance(val, bytes):
            entries.append((tag, typ, cnt, int.from_bytes(val.ljust(4, b"\x00"), "little")))
        else:
            entries.append((tag, typ, cnt, val))
    data_off = off
    out = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(entries)))
    for tag, typ, cnt, val in entries:
        if tag == 273:
            val = data_off
        out += struct.pack("<HHI", tag, typ, cnt)
        out += struct.pack("<HH", val, 0) if (typ == 3 and cnt == 1) else struct.pack("<I", val)
    out += struct.pack("<I", 0)
    for b in blobs:
        out += b
    out += data
    return bytes(out)


# ------------------------------------------------------------------ contours (marching squares)
_SEGS = {  # corner bits: tl=8 tr=4 br=2 bl=1 ; edges: T=0 R=1 B=2 L=3
    1: [(3, 2)], 2: [(2, 1)], 3: [(3, 1)], 4: [(0, 1)], 6: [(0, 2)], 7: [(3, 0)],
    8: [(3, 0)], 9: [(0, 2)], 11: [(0, 1)], 12: [(3, 1)], 13: [(2, 1)], 14: [(3, 2)],
}


def contour_lines(Z: np.ndarray, level: float):
    """Polylines (list of (n,2) arrays in grid units: x = col, y = row, cell centres at integers) at one level."""
    ny, nx = Z.shape
    tl, tr, br, bl = Z[:-1, :-1], Z[:-1, 1:], Z[1:, 1:], Z[1:, :-1]
    valid = np.isfinite(tl) & np.isfinite(tr) & np.isfinite(br) & np.isfinite(bl)
    a = lambda v: v > level  # noqa: E731
    case = a(tl) * 8 + a(tr) * 4 + a(br) * 2 + a(bl) * 1
    case = np.where(valid, case, 0)
    I, J = np.indices(case.shape)
    seg_a, seg_b = [], []  # edge keys

    def ekey(edge, i, j):
        # horizontal edge (between two cells in the same row): id 2*(i*nx+j); vertical: 2*(i*nx+j)+1
        if edge == 0:
            return 2 * (i * nx + j)
        if edge == 2:
            return 2 * ((i + 1) * nx + j)
        if edge == 3:
            return 2 * (i * nx + j) + 1
        return 2 * (i * nx + j + 1) + 1

    def add(mask, e1, e2):
        i, j = I[mask], J[mask]
        if len(i):
            seg_a.append(ekey(e1, i, j))
            seg_b.append(ekey(e2, i, j))

    for c, segs in _SEGS.items():
        m = case == c
        for e1, e2 in segs:
            add(m, e1, e2)
    for c in (5, 10):  # saddles: decide by the centre value
        m = case == c
        if m.any():
            centre = (tl + tr + br + bl) / 4 > level
            for sel, pairs in ((m & centre, [(3, 0), (2, 1)] if c == 5 else [(3, 2), (0, 1)]),
                               (m & ~centre, [(3, 2), (0, 1)] if c == 5 else [(3, 0), (2, 1)])):
                for e1, e2 in pairs:
                    add(sel, e1, e2)
    if not seg_a:
        return []
    ka, kb = np.concatenate(seg_a), np.concatenate(seg_b)
    keys = np.unique(np.concatenate([ka, kb]))
    # position of each edge crossing
    horiz = (keys & 1) == 0
    cellid = keys >> 1
    ci, cj = cellid // nx, cellid % nx
    px = np.empty(len(keys))
    py = np.empty(len(keys))
    z1 = np.where(horiz, Z[ci, cj], Z[ci, cj])
    z2 = np.where(horiz, Z[ci, np.minimum(cj + 1, nx - 1)], Z[np.minimum(ci + 1, ny - 1), cj])
    t = (level - z1) / np.where(z2 == z1, 1e-9, z2 - z1)
    t = np.clip(t, 0, 1)
    px[horiz] = (cj + t)[horiz]
    py[horiz] = ci[horiz]
    px[~horiz] = cj[~horiz]
    py[~horiz] = (ci + t)[~horiz]
    index = {int(k): n for n, k in enumerate(keys)}
    nxt: dict[int, list[int]] = {}
    for s, (x, y) in enumerate(zip(ka.tolist(), kb.tolist())):
        nxt.setdefault(x, []).append(s)
        nxt.setdefault(y, []).append(s)
    used = np.zeros(len(ka), bool)
    ka_l, kb_l = ka.tolist(), kb.tolist()
    out = []

    def walk(start_key, seg):
        chain = [start_key]
        key = start_key
        while True:
            used[seg] = True
            other = kb_l[seg] if ka_l[seg] == key else ka_l[seg]
            chain.append(other)
            key = other
            nxt_seg = [s for s in nxt[key] if not used[s]]
            if not nxt_seg:
                return chain
            seg = nxt_seg[0]

    # open lines first (start at end points), then closed rings
    for k, ss in nxt.items():
        if len(ss) == 1 and not used[ss[0]]:
            out.append(walk(k, ss[0]))
    for s in range(len(ka_l)):
        if not used[s]:
            out.append(walk(ka_l[s], s))
    lines = []
    for ch in out:
        ii = [index[k] for k in ch]
        lines.append(np.c_[px[ii], py[ii]])
    return lines


def contours(Z: np.ndarray, x0: float, y_top: float, cell: float, interval: float, major_every: int = 5):
    """List of (elevation, is_major, xy array in map coordinates) for every contour level."""
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    first = np.ceil(zmin / interval) * interval
    levels = np.arange(first, zmax + 1e-9, interval)
    res = []
    for lv in levels:
        lv = float(round(lv, 6))
        k = int(round(lv / interval))
        is_major = major_every > 0 and k % major_every == 0
        for ln in contour_lines(Z, lv + 1e-6):
            xy = np.c_[x0 + (ln[:, 0] + 0.5) * cell, y_top - (ln[:, 1] + 0.5) * cell]
            res.append((lv, is_major, xy))
    return res


def new_doc(layers):
    """A complete, AutoCAD-safe DXF document (written by ezdxf) with the given (name, aci colour) layers."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    for name, col in layers:
        if name not in doc.layers:
            doc.layers.add(name, color=col)
    return doc


def doc_bytes(doc) -> bytes:
    import io

    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def contours_dxf(lines, label_every: float = 40.0) -> bytes:
    """LWPOLYLINE contours (true elevation) on CONTOUR_MINOR / CONTOUR_MAJOR, elevation text on major ones."""
    doc = new_doc([("CONTOUR_MINOR", 8), ("CONTOUR_MAJOR", 30), ("CONTOUR_LABEL", 7)])
    msp = doc.modelspace()
    for lv, major, xy in lines:
        if len(xy) < 2:
            continue
        closed = bool(np.allclose(xy[0], xy[-1]))
        pts = xy[:-1] if closed else xy
        msp.add_lwpolyline(pts.tolist(), close=closed,
                           dxfattribs={"layer": "CONTOUR_MAJOR" if major else "CONTOUR_MINOR", "elevation": float(lv)})
        if major:
            seg = np.hypot(*np.diff(xy, axis=0).T)
            if seg.sum() >= label_every:
                m = int(np.searchsorted(np.cumsum(seg), seg.sum() / 2))
                p, q = xy[m], xy[min(m + 1, len(xy) - 1)]
                ang = float(np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])))
                ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
                msp.add_text(f"{lv:.2f}", height=0.4, rotation=ang,
                             dxfattribs={"layer": "CONTOUR_LABEL", "insert": (float(p[0]), float(p[1]), float(lv))})
    return doc_bytes(doc)


# ------------------------------------------------------------------ CSV survey points
def _split(line: str, delim: str):
    return [c.strip().strip('"') for c in line.split(delim)] if delim != " " else line.split()


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        try:
            return float(v.replace(",", "."))  # decimal comma
        except ValueError:
            return None


def csv_preview(text: str):
    """Guess delimiter, header row and which columns are X / Y / Z. Returns a dict for the UI."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "//"))]
    if not lines:
        raise ValueError("the file is empty")
    def score(d):  # the delimiter that splits every sampled line into the same, largest number of columns
        n = [len(_split(ln, d)) for ln in lines[:20]]
        return (min(n) if min(n) == max(n) else 0, max(n))

    delim = max([",", ";", "\t", " "], key=score)
    rows = [_split(ln, delim) for ln in lines[:200]]
    ncol = max(len(r) for r in rows)
    first_numeric = sum(_num(c) is not None for c in rows[0]) >= max(2, len(rows[0]) // 2)
    header = not first_numeric
    names = rows[0] if header else [f"col {i + 1}" for i in range(ncol)]
    guess = {"x": None, "y": None, "z": None}
    keys = {"x": ("x", "e", "east", "easting", "eastings", "lon"), "y": ("y", "n", "north", "northing", "northings", "lat"),
            "z": ("z", "h", "height", "elev", "elevation", "level", "alt", "rl", "ortho")}
    if header:
        for ax, ks in keys.items():
            for i, nme in enumerate(names):
                if nme.lower().strip() in ks and i not in guess.values():
                    guess[ax] = i
                    break
    if None in guess.values():  # numeric layout: [id,] E, N, Z [, code]
        body = rows[1:] if header else rows
        numeric = [i for i in range(ncol) if body and all(len(r) > i and _num(r[i]) is not None for r in body[:50])]
        if len(numeric) >= 3:
            big = [i for i in numeric if body and np.mean([abs(_num(r[i])) for r in body[:50]]) > 1000]
            xy = big[:2] if len(big) >= 2 else numeric[:2]
            rest = [i for i in numeric if i not in xy]
            z = next((i for i in rest if i > xy[-1]), rest[0] if rest else None)
            guess = {"x": xy[0], "y": xy[1], "z": z} if z is not None else guess
    return {"delim": delim, "ncol": ncol, "header": header, "names": names, "rows": rows[(1 if header else 0):][:8], "guess": guess, "lines": len(lines)}


def parse_points(text: str, xcol: int, ycol: int, zcol: int, delim: str, header: bool) -> np.ndarray:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "//"))]
    out = []
    for ln in lines[1 if header else 0:]:
        r = _split(ln, delim)
        if len(r) <= max(xcol, ycol, zcol):
            continue
        x, y, z = _num(r[xcol]), _num(r[ycol]), _num(r[zcol])
        if x is None or y is None or z is None or not np.isfinite([x, y, z]).all():
            continue
        out.append((x, y, z))
    if not out:
        raise ValueError("no usable rows with the chosen X / Y / Z columns")
    return np.asarray(out, float)
