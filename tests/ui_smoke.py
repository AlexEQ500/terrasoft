"""Headless UI smoke test: upload -> classify -> brush -> section -> screenshots."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

las = sys.argv[1]
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1400, "height": 860})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errs.append("PAGEERROR " + str(e)))
    pg.goto("http://127.0.0.1:8765")
    pg.set_input_files("#file", las)
    pg.wait_for_function("document.querySelector('#fileinfo').textContent.includes('points')", timeout=120000)
    pg.wait_for_timeout(800)
    pg.screenshot(path=str(out / "1_loaded.png"))

    pg.click("#classifyBtn")
    pg.wait_for_function("document.querySelector('#bar').textContent.includes('classified')", timeout=180000)
    pg.wait_for_timeout(800)
    pg.screenshot(path=str(out / "2_classified.png"))
    print("legend:", pg.inner_text("#legend").replace("\n", " | "))

    # brush: switch to top view, relabel a patch of ground to noise, then undo
    pg.click("#topBtn")
    pg.wait_for_timeout(400)
    pg.click("#brushBtn")
    pg.select_option("#paintClass", "7")
    pg.mouse.move(800, 400)
    pg.mouse.down()
    pg.mouse.move(900, 420, steps=6)
    pg.mouse.up()
    pg.wait_for_timeout(800)
    print("after brush:", pg.inner_text("#bar"))
    pg.screenshot(path=str(out / "3_brush.png"))
    pg.click("#undoBtn")
    pg.wait_for_timeout(500)
    print("after undo:", pg.inner_text("#bar"))
    pg.click("#brushBtn")

    # section across the road/kerb: two clicks in top view
    pg.click("#secBtn")
    pg.mouse.click(700, 520)
    pg.mouse.click(700, 330)
    pg.wait_for_timeout(500)
    pg.select_option("#secV", "10")
    pg.wait_for_timeout(300)
    pg.screenshot(path=str(out / "4_section.png"))
    print("readout:", pg.inner_text("#readout"))
    pg.click("#secClose")

    # pick tool: click a tree crown in top view, then a wrong class on it
    pg.click("#topBtn"); pg.wait_for_timeout(500)
    def screen_of(cls):
        return pg.evaluate("""(cls) => { const T = window.__terrasort, D = T.D, cam = T.camera; cam.updateMatrixWorld();
          const r = document.getElementById('gl').getBoundingClientRect(); let best = null;
          for (let i = 0; i < D.n; i += 7) { if (D.labels[i] !== cls) continue;
            const v = cam.position.clone().set(D.pos[i*3], D.pos[i*3+1], D.pos[i*3+2]).project(cam);
            const x = (v.x*0.5+0.5)*r.width + r.left, y = (0.5-v.y*0.5)*r.height + r.top;
            if (x > r.left+60 && x < r.right-60 && y > r.top+60 && y < r.bottom-60) { best = [x, y]; break; } }
          return best; }""", cls)
    xy = screen_of(5)
    print("tree screen pos:", xy)
    pg.click("#pickGrid button[data-tool=wall]")
    pg.mouse.click(*xy); pg.wait_for_timeout(800)
    print("wall tool on tree:", pg.inner_text("#bar"))
    pg.click("#pickGrid button[data-tool=tree]")
    pg.mouse.click(*xy); pg.wait_for_timeout(800)
    print("tree tool on tree (preview):", pg.inner_text("#bar"), "| pending:", pg.evaluate("window.__terrasort.pending && window.__terrasort.pending.set.size"))
    pg.click("#pvApply"); pg.wait_for_timeout(600)
    print("after apply:", pg.inner_text("#bar"))
    pg.click("#undoBtn"); pg.wait_for_timeout(300)
    # area select: drag a rectangle, preview, cancel with Esc, then again and apply
    pg.click("#pickGrid button[data-tool=area]")
    pg.select_option("#areaTo", "7")
    pg.mouse.move(xy[0]-60, xy[1]-60); pg.mouse.down(); pg.mouse.move(xy[0]+60, xy[1]+60, steps=5); pg.mouse.up()
    pg.wait_for_timeout(400)
    print("area preview:", pg.inner_text("#bar"), "|", pg.inner_text("#pvText"))
    pg.screenshot(path=str(out / "5b_area.png"))
    pg.keyboard.press("Escape"); pg.wait_for_timeout(200)
    print("after esc pending:", pg.evaluate("window.__terrasort.pending"))
    pg.mouse.move(xy[0]-60, xy[1]-60); pg.mouse.down(); pg.mouse.move(xy[0]+60, xy[1]+60, steps=5); pg.mouse.up()
    pg.keyboard.press("Enter"); pg.wait_for_timeout(600)
    print("area apply:", pg.inner_text("#bar"))
    pg.click("#pickGrid button[data-tool=tree]")
    pg.screenshot(path=str(out / "5_pick.png"))
    pg.click("#undoBtn"); pg.wait_for_timeout(400)
    # ---- trees: find stems, table, manual circle, csv/dxf, view extras, lasso, was
    pg.click("#stemsBtn"); pg.wait_for_timeout(2500)
    print("stems:", pg.inner_text("#bar"), "| rows:", pg.locator("#treeBody tr[data-id]").count(), "| mode on tab:", pg.inner_text("#tabs button.on"))
    pg.screenshot(path=str(out / "6_stems.png"))
    pg.locator("#treeBody tr[data-id]").first.click(); pg.wait_for_timeout(600)
    pg.screenshot(path=str(out / "6b_zoom.png"))
    pg.click("#isoBtn"); pg.wait_for_timeout(300)
    print("csv:", pg.evaluate("fetch('/api/trees.csv').then(r=>r.text()).then(t=>t.split('\\n').slice(0,3).join(' | '))"))
    print("dxf bytes:", pg.evaluate("fetch('/api/trees.dxf').then(r=>r.text()).then(t=>t.length+' '+(t.match(/CIRCLE/g)||[]).length+' circles')"))
    pg.click("#circBtn"); pg.mouse.click(5, 5)  # click outside canvas area: ignored
    pg.click("#lastBtn"); pg.wait_for_timeout(400)
    print("after remove last:", pg.inner_text("#treeCount"))
    pg.click("#orthoBtn"); pg.wait_for_timeout(300); pg.screenshot(path=str(out / "7_ortho.png")); pg.click("#orthoBtn")
    pg.check("#xray"); pg.fill("#pkeep", "60"); pg.wait_for_timeout(200); pg.screenshot(path=str(out / "8_xray.png")); pg.uncheck("#xray")
    pg.click("#wasBtn"); pg.wait_for_timeout(300); print("was:", pg.inner_text("#bar")); pg.click("#wasBtn")
    pg.click("#pickGrid button[data-tool=area]"); pg.select_option("#areaShape", "lasso"); pg.select_option("#areaTo", "7")
    cx, cy = 700, 400
    pg.mouse.move(cx, cy); pg.mouse.down()
    for k in range(0, 360, 30):
        import math
        pg.mouse.move(cx + 90*math.cos(math.radians(k)), cy + 90*math.sin(math.radians(k)))
    pg.mouse.up(); pg.wait_for_timeout(500)
    print("lasso:", pg.inner_text("#pvText")); pg.keyboard.press("Escape")
    print("console issues:", [e for e in errs if "GPU stall" not in e][:8])
    b.close()
