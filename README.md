# Terrasort

Local tool that classifies survey point clouds (LAS/LAZ) into ground, kerbs, walls, trees,
shrubs, people and noise, with a 3D viewer, a correction brush, a section tool and LAS export.

## Run
    pip install -r requirements.txt
    python -m terrasort serve                 # then open the page and load a LAS/LAZ
    python -m terrasort serve site.las        # or load a file directly
    python -m terrasort classify in.las out.las --profile static_scan   # no UI

Everything runs on your machine (http://127.0.0.1:8765). three.js is bundled, so no internet is needed.
Try it first with `python tests/make_sample.py`, which writes a synthetic `sample_site.las`.

## Output classes (LAS codes)
1 other, 2 ground, 3 low veg, 4 shrub, 5 tree, 6 building, 7 noise, **64 kerb, 65 person, 66 pole/post/rail,
67 roof/canopy (low), 68 wall/fence (low), 69 car**. 64 and up are user-defined codes, so the export is written as point format 6+ even if the input was older.

## How it works
Voxel-average to a working set (default 5 cm), then: noise (isolated voxels), ground (progressive
lowest-point TIN, 8 m to 0.5 m cells), height above ground, kerbs (short, low, elongated height steps on a
2.5D raster), local PCA normals, walls (vertical planar components), objects (tree / shrub / person / other,
by connected component). Labels are mapped back to every original point. Sensor profiles (static scan,
drone photogrammetry) are sets of thresholds in `terrasort/core.py`; override any of them in the UI
under "Advanced".

## Honest status
* Thresholds were tuned and tested ONLY on a synthetic scene (`tests/run_synth.py`). Expect to tune on real data.
* "Person" is a shape heuristic (height, footprint, smooth surface) and is the weakest class. The plan is to
  replace it with a model trained on your corrected labels (brush corrections are the training data).
* Kerbs assume a step of roughly 6-30 cm that is at least 1.5 m long. Dropped kerbs and ramps will be missed.
* Very large clouds are shown as a voxel working set (max ~4 M voxels; the voxel grows automatically).
  Edits apply to every original point inside the voxel.
* The kerb raster is limited to ~25 M cells and coarsens itself on very large extents. Tile big sites.
* Cars need the roof, or both sides, to be visible (a drone sees the roof; a one-sided static scan of a car may look like a fence).
* A building is any wall/roof structure that reaches 2.5 m (`building_min_h`); lower ones stay wall/fence.
* No ML model yet (PTv3 / Pointcept is the planned next step) and no DXF export yet.


## Pick object (click to classify)

Classify first, then use **3 · Pick object**: choose a class button (Tree, Shrub, Building, Roof, Wall / fence, Ground, Kerb, Car, High pole, Low pole / rail, Person, Noise, or Other object with a class dropdown) and click the object in the 3D view. Every class has its own rule for what gets selected (e.g. Tree ignores flat surfaces, Wall takes only vertical planes, Roof only non-vertical planes well above ground, Kerb finds the small step along the kerb line). If the click does not look like the chosen class you get a message and nothing changes. **Reach** limits how far the selection may grow (auto = per-class default). Ctrl+Z undoes. Dragging still orbits the view.

New in Pick object: **preview** (selection shown in yellow; Enter or Apply to confirm, Esc or Cancel to drop, Shift+click adds more; untick "preview before applying" for instant apply), **Area select** (drag a rectangle, front surface only unless "through all depth", sets the "Set to" class, optionally only from one class), and the classes **Street furniture** (71) and **Sign / signal** (72). High pole (66) and Low pole (70) are split at 3 m.

## Tree inventory (section 5)
Classify first, then **Find stems**: the cloud is sliced 1.0–1.6 m above the ground, each slice cluster is fitted with a circle, and only round, vertical, tree-like stems with a crown nearby are kept (walls, fences, hedges, posts and signs fail). Each tree gets a number, diameter (DBH), height (with crown), ground level and crown radius, shown in the table (click a row to zoom, ✕ deletes, the tick hides) and as labels and diameter circles in 3D. Colour mode **Stems** gives every tree its own colour. **Circle by hand**: click a trunk (search radius adjustable). **Crown edge**: click the outer edge of a crown. **Remove last**, **List CSV…** and **DXF…** (circles, crown circles, ground points and text labels in real-world coordinates; hidden trees are left out). Trees added by hand are marked with *.

## Other additions
Terrain presets (flat / hills / slopes / city) for the ground pass, **Ortho** view, **X-ray** with opacity, **Points** slider (show only a share of points when moving the view), **Was** (shows the classification before your manual edits; editing is locked while on), brush **whole column**, **Lasso (contour)** in Area select.

## Licence

MIT, see LICENSE. Bundled three.js is MIT (Three.js authors).
