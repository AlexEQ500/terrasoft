"""Command line: terrasort serve [file.las]  |  terrasort classify in.las out.las"""
import argparse
import sys
import time
import webbrowser

import numpy as np


def main():
    ap = argparse.ArgumentParser(prog="terrasort")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the local UI")
    s.add_argument("file", nargs="?")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")

    c = sub.add_parser("classify", help="classify a LAS/LAZ without the UI")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--profile", default="static_scan")

    a = ap.parse_args()
    if a.cmd == "serve":
        import uvicorn

        from . import server

        import socket
        import threading

        port = a.port
        for p in range(a.port, a.port + 20):          # an older Terrasort may still hold the port
            with socket.socket() as sk:
                if sk.connect_ex(("127.0.0.1", p)) != 0:
                    port = p
                    break
        if port != a.port:
            print(f"Port {a.port} is already used (an older Terrasort is probably still running) - using {port} instead.")
        if a.file:
            server.load_file(a.file)
        url = f"http://127.0.0.1:{port}"
        print(f"Terrasort {server.VERSION} is running at {url}")
        if not a.no_browser:
            threading.Timer(1.5, lambda: webbrowser.open(f"{url}/?new={int(time.time())}")).start()
        uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="warning")
    else:
        import laspy

        from . import core as C

        las = laspy.read(a.src)
        xyz = np.stack([las.x, las.y, las.z], 1)
        prof = C.get_profile(a.profile)
        pts, inv, v = C.voxelize(xyz, prof.voxel, prof.max_voxels)
        lab, _ = C.classify_points(pts, prof, prog=lambda f, m: print(f"{f:4.0%} {m}", file=sys.stderr))
        if las.header.point_format.id < 6:
            las = laspy.convert(las, point_format_id=6)
        las.classification = lab[inv]
        las.write(a.dst)
        print(f"wrote {a.dst}")
