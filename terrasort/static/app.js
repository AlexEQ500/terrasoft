import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const COLORS = {
  0: '#6b7280', 1: '#c2b5a3', 2: '#b4733b', 3: '#a3e635', 4: '#4ade80', 5: '#15803d',
  6: '#4f6df5', 7: '#f43f5e', 64: '#facc15', 65: '#f472b6', 66: '#22d3ee', 67: '#a78bfa', 68: '#7dd3fc', 69: '#fb923c', 70: '#2dd4bf', 71: '#e879f9', 72: '#fde047', 73: '#94a3b8',
};
let NAMES = { 0: 'Not classified' };
const vis = new Uint8Array(256).fill(1);

// ------------------------------------------------------------------ three.js
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d0f12);
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.screenSpacePanning = true;
controls.mouseButtons.LEFT = 'pivot'; // left-drag orbits around the point that was clicked (see pivot orbit below)

const palData = new Uint8Array(256 * 4);
const palTex = new THREE.DataTexture(palData, 256, 1, THREE.RGBAFormat);
palTex.minFilter = palTex.magFilter = THREE.NearestFilter;

function hexRGB(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
function updatePalette() {
  for (let c = 0; c < 256; c++) {
    const [r, g, b] = hexRGB(COLORS[c] || '#ffffff');
    palData.set([r, g, b, vis[c] ? 255 : 0], c * 4);
  }
  palTex.needsUpdate = true;
}

const material = new THREE.ShaderMaterial({
  uniforms: {
    size: { value: 2 }, mode: { value: 0 }, pal: { value: palTex },
    zr: { value: new THREE.Vector2(0, 1) }, hr: { value: new THREE.Vector2(0, 5) }, keep: { value: 1 }, alpha: { value: 1 }, adj: { value: new THREE.Vector4(0, 1, 1, 0) }, clip: { value: new THREE.Vector2(0, 1) },
  },
  vertexShader: `
    uniform float size; uniform int mode; uniform sampler2D pal; uniform vec2 zr; uniform vec2 hr; uniform float keep; uniform vec4 adj; uniform vec2 clip;
    attribute float cls; attribute float hag; attribute vec3 col; attribute float inten; attribute float tid; attribute float rnd;
    varying vec3 vC;
    vec3 tone(vec3 c){ c = (c - .5) * adj.y + .5 + adj.x; return pow(clamp(c,0.,1.), vec3(1./adj.z)); }
    vec3 ramp(float t){ t=clamp(t,0.,1.);
      return mix(mix(vec3(.12,.25,.95),vec3(.12,.9,.35),clamp(t*2.,0.,1.)),vec3(.95,.2,.12),clamp(t*2.-1.,0.,1.)); }
    void main(){
      vec4 p = texture2D(pal, vec2((cls+0.5)/256., 0.5));
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.);
      if (p.a < 0.5 || rnd > keep) gl_Position = vec4(2.,2.,2.,1.);
      gl_PointSize = size;
      if (mode==0) vC = p.rgb;
      else if (mode==1) vC = tone(col);
      else if (mode==2) vC = ramp((position.z - zr.x)/(zr.y - zr.x));
      else if (mode==3) vC = ramp(hag / hr.y);
      else if (mode==5) {
        if (tid < 0.5) vC = vec3(.30,.20,.12) * (0.55 + 0.45 * clamp(hag / 2., 0., 1.));
        else { float h = fract(sin(tid * 12.9898) * 43758.5453); vec3 q = clamp(abs(mod(h * 6. + vec3(0., 4., 2.), 6.) - 3.) - 1., 0., 1.); vC = mix(vec3(1.), q, .8); }
      }
      else { float t = clamp((inten - clip.x) / max(clip.y - clip.x, .01), 0., 1.); vC = tone(vec3(t)); }
    }`,
  fragmentShader: `
    uniform float alpha; varying vec3 vC;
    void main(){ vec2 c = gl_PointCoord - 0.5; if (dot(c,c) > 0.25) discard; gl_FragColor = vec4(vC, alpha); }`,
});

let D = null; // {n,pos,labels,hag,rgb,inten,points,bbox}
let center = [0, 0, 0];

function resize() {
  const r = $('main').getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

function loop() { requestAnimationFrame(loop); renderer.render(scene, camera); if (window.__treesReady) updateTreeLabels(); }
loop();

// ------------------------------------------------------------------ data
async function getBin(url, Type) {
  const r = await fetch(url);
  if (!r.ok) return null;
  return new Type(await r.arrayBuffer());
}

async function loadArrays(st) {
  const [pos, labels, hag, tid16] = await Promise.all([
    getBin('/api/pos', Float32Array), getBin('/api/labels', Uint8Array), getBin('/api/hag', Float32Array), getBin('/api/tid', Int16Array),
  ]);
  const rgb = st.has_rgb ? await getBin('/api/rgb', Uint8Array) : null;
  const inten = st.has_intensity ? await getBin('/api/intensity', Uint8Array) : null;
  if (D) { scene.remove(D.points); D.points.geometry.dispose(); }
  const n = labels.length;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('cls', new THREE.BufferAttribute(labels, 1));
  g.setAttribute('hag', new THREE.BufferAttribute(hag, 1));
  g.setAttribute('tid', new THREE.BufferAttribute(Float32Array.from(tid16 || new Int16Array(n)), 1));
  const rnd = new Float32Array(n); for (let i = 0; i < n; i++) rnd[i] = Math.random();
  g.setAttribute('rnd', new THREE.BufferAttribute(rnd, 1));
  g.setAttribute('col', new THREE.BufferAttribute(rgb || new Uint8Array(n * 3).fill(160), 3, true));
  g.setAttribute('inten', new THREE.BufferAttribute(inten || new Uint8Array(n).fill(160), 1, true));
  const points = new THREE.Points(g, material);
  points.frustumCulled = false;
  scene.add(points);
  let mn = [1e30, 1e30, 1e30], mx = [-1e30, -1e30, -1e30];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) {
    const v = pos[i * 3 + d]; if (v < mn[d]) mn[d] = v; if (v > mx[d]) mx[d] = v;
  }
  D = { name: String(st.name || 'scan').replace(/\.[^.]+$/, ''), n, pos, labels, hag, rgb, inten, points, bbox: [mn, mx], labels0: labels.slice(), edited: null };
  wasOn = false; $('wasBtn').classList.remove('on');
  if (window.__csvRebuild) window.__csvRebuild();
  center = st.center;
  material.uniforms.zr.value.set(mn[2], mx[2]);
  let hmax = 0; for (let i = 0; i < n; i += 7) if (hag[i] > hmax) hmax = hag[i];
  material.uniforms.hr.value.set(0, Math.max(1, Math.min(hmax, 15)));
  $('tabRgb').style.display = rgb ? '' : 'none';
  $('tabInt').style.display = inten ? '' : 'none';
  if (!rgb && material.uniforms.mode.value === 1) setMode(0);
  $('empty').style.display = 'none';
  resetView();
  refreshLegend();
  enableTools(true);
}

function resetView() {
  if (!D) return;
  const [mn, mx] = D.bbox;
  const c = [0, 1, 2].map((d) => (mn[d] + mx[d]) / 2);
  const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1);
  controls.target.set(...c);
  camera.up.set(0, 0, 1);
  camera.position.set(c[0] - ext * 0.5, c[1] - ext * 0.9, c[2] + ext * 0.7);
  camera.near = ext * 0.001; camera.far = ext * 30; camera.updateProjectionMatrix();
  controls.update();
}
function topView() {
  if (!D) return;
  const [mn, mx] = D.bbox;
  const c = [0, 1, 2].map((d) => (mn[d] + mx[d]) / 2);
  const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], 1);
  controls.target.set(...c);
  camera.up.set(0, 1, 0);
  camera.position.set(c[0], c[1] - 0.001, c[2] + ext * 1.1);
  controls.update();
}

// ------------------------------------------------------------------ UI: legend, modes
function counts() {
  const c = new Uint32Array(256);
  for (let i = 0; i < D.n; i++) c[D.labels[i]]++;
  return c;
}
function refreshLegend() {
  if (window.__groundAuto) window.__groundAuto();
  const c = counts();
  const el = $('legend'); el.innerHTML = '';
  const fill = (sel, extra) => {
    const keep = sel.value; sel.innerHTML = extra || '';
    Object.keys(NAMES).map(Number).sort((a, b) => a - b).forEach((k) => {
      if (k === 0) return; sel.insertAdjacentHTML('beforeend', `<option value="${k}">${NAMES[k]}</option>`);
    });
    if (keep) sel.value = keep;
  };
  const SUB = { 2: [64, 73], 7: [69, 65] }; // kerbs, low flat under Ground; cars, people under Noise
  const subOf = new Set([64, 73, 69, 65]);
  const row = (k, indent) => {
    const d = document.createElement('div');
    if (indent) d.style.paddingLeft = '18px';
    d.innerHTML = `<input type="checkbox" ${vis[k] ? 'checked' : ''}><span class="sw" style="background:${COLORS[k] || '#fff'}"></span><span>${indent ? '↳ ' : ''}${NAMES[k] || k}</span><span class="n">${c[k].toLocaleString()}</span>`;
    d.querySelector('input').onchange = (e) => {
      const on = e.target.checked ? 1 : 0; vis[k] = on;
      if (SUB[k]) SUB[k].forEach((s) => (vis[s] = on)); // parent switches its sub-classes too
      updatePalette(); refreshLegend();
    };
    el.appendChild(d);
  };
  Object.keys(NAMES).map(Number).sort((a, b) => a - b).forEach((k) => {
    if (subOf.has(k)) return; // always drawn under its parent
    if (!c[k] && !(SUB[k] && SUB[k].some((s) => c[s]))) return;
    row(k, false);
    (SUB[k] || []).forEach((s) => { if (c[s]) row(s, true); });
  });
  fill($('paintClass'));
  fill($('fromClass'), '<option value="-1">any visible</option>');
}

function setMode(m) {
  material.uniforms.mode.value = m;
  $('adj').style.display = (m === 1 || m === 4) && D ? '' : 'none';
  $('adjIL').style.display = $('adjIH').style.display = m === 4 ? '' : 'none';
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', +b.dataset.mode === m));
}
document.querySelectorAll('#tabs button').forEach((b) => (b.onclick = () => setMode(+b.dataset.mode)));
$('psize').oninput = (e) => (material.uniforms.size.value = +e.target.value);
$('brushD').oninput = (e) => ($('brushDv').textContent = (e.target.value / 10).toFixed(1) + ' m');
$('topBtn').onclick = topView;
$('isoBtn').onclick = resetView;

function enableTools(on) {
  ['classifyBtn', 'brushBtn', 'secBtn', 'topBtn', 'isoBtn', 'exportBtn', 'stemsBtn', 'circBtn', 'crownBtn', 'areaDrawBtn', 'areaClrBtn', 'lastBtn', 'csvBtn', 'dxfBtn', 'orthoBtn', 'wasBtn', 'dtmBtn', 'conBtn', 'gvBtn', 'gvClr'].forEach((id) => ($(id).disabled = !on));
}

// ------------------------------------------------------------------ loading / classifying
let busy = false;
function setProgress(f) { document.querySelector('#prog i').style.width = (f * 100).toFixed(0) + '%'; }
async function poll(until) {
  for (;;) {
    const st = await (await fetch('/api/status')).json();
    setProgress(st.progress);
    $('bar').textContent = st.message || '';
    if (st.state === 'error') { alert(st.message + '\n' + (st.log.slice(-1)[0] || '')); return st; }
    if (st.state === 'ready') return st;
    await new Promise((r) => setTimeout(r, 350));
  }
}
async function afterReady(st) {
  await loadArrays(st);
  $('fileinfo').textContent = `${st.name} — ${st.n_points.toLocaleString()} points, ${st.n_work.toLocaleString()} shown (voxel ${st.voxel.toFixed(3)} m)`;
  $('undoBtn').disabled = !st.can_undo;
  hidden.clear(); await loadTrees();
  setProgress(0);
  $('bar').textContent = st.message;
}
fetch('/api/status').then((r) => r.json()).then((s) => ($('ver').textContent = s.version));
async function loadClasses() { NAMES = { 0: 'Not classified', ...(await (await fetch('/api/classes')).json()) }; updatePalette(); buildPickGrid(); }

function upload(file) {
  busy = true; $('bar').textContent = `uploading ${file.name}…`;
  const x = new XMLHttpRequest();
  x.open('POST', '/api/upload?name=' + encodeURIComponent(file.name));
  x.upload.onprogress = (e) => setProgress(e.loaded / e.total);
  x.onload = async () => { const st = await poll(); if (st.state === 'ready') await afterReady(st); busy = false; };
  x.send(file);
}
$('openBtn').onclick = () => $('file').click();
$('file').onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const dropEl = $('drop');
addEventListener('dragover', (e) => { e.preventDefault(); dropEl.style.display = 'flex'; });
addEventListener('dragleave', (e) => { if (e.target === dropEl) dropEl.style.display = 'none'; });
addEventListener('drop', (e) => { e.preventDefault(); dropEl.style.display = 'none'; if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]); });

const TERRAIN = {
  hills: {},
  flat: { ground_cells: [4, 2, 1, 0.5], ground_seed_th: [0.6, 0.4, 0.3, 0.2], ground_th: 0.06 },
  slopes: { ground_cells: [10, 5, 2.5, 1, 0.5], ground_seed_th: [3, 1.5, 0.8, 0.4, 0.25], ground_th: 0.15, ground_below: 0.25 },
  city: { ground_cells: [6, 3, 1.5, 0.75, 0.4], ground_seed_th: [1.0, 0.5, 0.3, 0.2, 0.15], ground_th: 0.06, building_min_h: 2.0 },
};
$('classifyBtn').onclick = async () => {
  let overrides = {};
  try { overrides = JSON.parse($('overrides').value || '{}'); } catch { return alert('Overrides must be valid JSON'); }
  overrides = { ...TERRAIN[$('terrain').value || 'hills'], low_flat: $('lowflat').value, ...overrides };
  $('classifyBtn').disabled = true;
  const r = await fetch('/api/classify', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile: $('profile').value, overrides }) });
  if (!r.ok) { alert(await r.text()); $('classifyBtn').disabled = false; return; }
  const st = await poll();
  if (st.state === 'ready') await afterReady(st);
  $('classifyBtn').disabled = false;
};
$('exportBtn').onclick = () => { $('bar').textContent = 'writing LAS…'; location.href = '/api/export'; setTimeout(() => ($('bar').textContent = ''), 4000); };

// ------------------------------------------------------------------ projection / picking
const tmpM = new THREE.Matrix4();
function frame() {
  camera.updateMatrixWorld();
  tmpM.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const r = canvas.getBoundingClientRect();
  return { e: tmpM.elements, W: r.width, H: r.height, r };
}
/** indices of visible points within radius px of (mx,my) on the front surface layer (depth tolerance m). */
function pick(mx, my, rad, tol, only) {
  const { e, W, H } = frame();
  const p = D.pos, lab = D.labels, n = D.n, r2 = rad * rad;
  let cand = new Int32Array(1 << 16), nc = 0, dmin = 1e30;
  const cd = [];
  for (let i = 0; i < n; i++) {
    const l = lab[i]; if (!vis[l]) continue;
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (w <= 0) continue;
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * 0.5 + 0.5) * W - mx;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * 0.5) * H - my;
    if (sx * sx + sy * sy > r2) continue;
    if (nc === cand.length) { const b = new Int32Array(nc * 2); b.set(cand); cand = b; }
    cand[nc++] = i; cd.push(w); if (w < dmin) dmin = w;
  }
  const out = [];
  for (let k = 0; k < nc; k++) {
    if (cd[k] > dmin + tol) continue;
    if (only >= 0 && lab[cand[k]] !== only) continue;
    out.push(cand[k]);
  }
  return { idx: out, dmin };
}
function nearestPoint(mx, my) {
  const { idx } = pick(mx, my, 10, 1e9, -1);
  if (!idx.length) return null;
  const { e, W, H } = frame(); const p = D.pos;
  const wOf = (i) => e[3] * p[i * 3] + e[7] * p[i * 3 + 1] + e[11] * p[i * 3 + 2] + e[15];
  let bw = 1e30; for (const i of idx) bw = Math.min(bw, wOf(i));
  let best = -1, bd = 1e30;
  for (const i of idx) {
    const w = wOf(i); if (w > bw + 0.5) continue;
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * 0.5 + 0.5) * W - mx;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * 0.5) * H - my;
    const d = sx * sx + sy * sy; if (d < bd) { bd = d; best = i; }
  }
  return best;
}

// ------------------------------------------------------------------ pick object (click -> whole object -> class)
const PICKS = [
  ['tree', 'Tree', 5, 'Click a trunk, branch or crown: selects the whole connected tree (not walls or ground).'],
  ['shrub', 'Shrub', 4, 'Click a bush or hedge: selects the connected low vegetation around it.'],
  ['building', 'Building', 6, 'Click a wall or roof: selects all connected wall and roof surfaces of the building.'],
  ['roof', 'Roof', 67, 'Click a roof: selects the connected roof surfaces only (not the walls).'],
  ['wall', 'Wall / fence', 68, 'Click a wall or fence face: selects the connected vertical surfaces.'],
  ['ground', 'Ground', 2, 'Click the ground: selects the connected ground surface (reach 25 m by default).'],
  ['kerb', 'Kerb', 64, 'Click a kerb: finds the small step and selects it along the kerb line.'],
  ['car', 'Car', 69, 'Click a car: selects the car-sized object that is not foliage.'],
  ['pole', 'High pole', 66, 'Click a lamp post or pole: selects the connected thin object (3 m and up).'],
  ['lowpole', 'Low pole / rail', 70, 'Click a post, bollard or handrail: selects the connected thin object up to 3 m high.'],
  ['furniture', 'Street furniture', 71, 'Click a bin, bench, cabinet or similar: selects the connected object above the ground.'],
  ['lowflat', 'Low flat surface', 73, 'Click a step, plinth, planter or paved patch: selects the connected low horizontal surface (up to 0.8 m above ground).'],
  ['sign', 'Sign / signal', 72, 'Click a sign or signal head: selects the connected object above the ground.'],
  ['person', 'Person', 65, 'Click a person: selects the small object around the click, up to head height.'],
  ['noise', 'Noise', 7, 'Click stray points: sets everything within a small sphere to noise (Reach = radius, default 0.4 m).'],
  ['object', 'Other object →', 1, 'Click any object above the ground: selects it and sets the class chosen in "Object as".'],
  ['area', 'Area select ▭', null, 'Drag a rectangle: selects the front-surface points inside it (Depth slider in Brush section sets how deep) and sets "Set to". Shift keeps adding to the preview.'],
];
let pickTool = null, downPos = null, pending = null, overlay = null, dragRect = null;
function buildPickGrid() {
  const g = $('pickGrid'); g.innerHTML = '';
  PICKS.forEach(([t, label, cls]) => {
    const b = document.createElement('button'); b.dataset.tool = t;
    if (t === 'area') b.style.gridColumn = '1 / span 2';
    b.innerHTML = `<span class="sw" style="background:${cls ? COLORS[cls] : 'linear-gradient(135deg,#fff,#888)'}"></span>${label}`;
    b.onclick = () => setPickTool(pickTool === t ? null : t);
    g.appendChild(b);
  });
  const opts = Object.keys(NAMES).map(Number).sort((a, b) => a - b).filter((k) => k).map((k) => `<option value="${k}">${NAMES[k]}</option>`).join('');
  $('objClass').innerHTML = opts; $('areaTo').innerHTML = opts;
  $('areaFrom').innerHTML = '<option value="-1">any visible</option>' + opts;
}
function setMouseForTool(leftFree) {
  controls.mouseButtons.LEFT = leftFree ? null : 'pivot';
  controls.mouseButtons.RIGHT = leftFree ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  controls.mouseButtons.MIDDLE = leftFree ? THREE.MOUSE.PAN : THREE.MOUSE.DOLLY;
}
function setPickTool(t) {
  pickTool = t;
  if (t && treeTool) { treeTool = null; syncTreeBtns(); }
  if (t && brush) setBrush(false);
  setMouseForTool(t === 'area');
  document.querySelectorAll('#pickGrid button').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  $('objRow').style.display = t === 'object' ? 'flex' : 'none';
  $('areaRows').style.display = t === 'area' ? 'block' : 'none';
  $('reachRow').style.display = t && t !== 'area' ? 'flex' : 'none';
  $('pickHint').textContent = t ? PICKS.find((p) => p[0] === t)[3] : 'No pick tool active.';
  canvas.style.cursor = t ? 'crosshair' : '';
}
$('pickReach').oninput = (e) => ($('pickReachV').textContent = +e.target.value ? e.target.value + ' m' : 'auto');

// --- preview of a pending selection (highlighted until applied or cancelled)
function clearPending() {
  pending = null; $('pvRow').style.display = 'none';
  if (overlay) { scene.remove(overlay); overlay.geometry.dispose(); overlay = null; }
}
function showPending() {
  if (overlay) { scene.remove(overlay); overlay.geometry.dispose(); overlay = null; }
  const idx = [...pending.set];
  const pos = new Float32Array(idx.length * 3);
  idx.forEach((i, k) => { pos[k * 3] = D.pos[i * 3]; pos[k * 3 + 1] = D.pos[i * 3 + 1]; pos[k * 3 + 2] = D.pos[i * 3 + 2]; });
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  overlay = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xfff04d, size: +material.uniforms.size.value + 2, sizeAttenuation: false }));
  overlay.frustumCulled = false; scene.add(overlay);
  $('pvRow').style.display = 'flex';
  $('pvText').textContent = `${idx.length.toLocaleString()} voxels → ${NAMES[pending.cls]}`;
}
async function applyIndices(idx, cls) {
  const arr = Array.isArray(idx) ? idx : [...idx];
  await fetch('/api/relabel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ indices: arr, cls }) });
  const lab = D.labels; for (const k of arr) lab[k] = cls;
  D.points.geometry.attributes.cls.needsUpdate = true;
  $('undoBtn').disabled = false; refreshLegend();
  $('bar').textContent = `${arr.length.toLocaleString()} voxels set to ${NAMES[cls]} (Ctrl+Z to undo)`;
}
async function applyPending() { if (!pending) return; const { set, cls } = pending; clearPending(); await applyIndices(set, cls); }
function offer(indices, cls, additive) {
  if (wasOn) { $('bar').textContent = 'switch off "Was" to edit'; return; }
  if (!$('pvOn').checked) { clearPending(); return applyIndices(indices, cls); }
  if (additive && pending && pending.cls === cls) indices.forEach((i) => pending.set.add(i));
  else pending = { set: new Set(indices), cls };
  showPending();
  $('bar').textContent = 'Preview in yellow. Enter = apply, Esc = cancel, Shift+click adds more.';
}
$('pvApply').onclick = applyPending;
$('pvCancel').onclick = () => { clearPending(); $('bar').textContent = 'selection cancelled'; };

// --- click tools
canvas.addEventListener('pointerdown', (ev) => { downPos = [ev.clientX, ev.clientY]; });
canvas.addEventListener('pointerup', async (ev) => {
  if (!pickTool || pickTool === 'area' || !D || secMode || brush || ev.button !== 0 || !downPos) return;
  if (Math.hypot(ev.clientX - downPos[0], ev.clientY - downPos[1]) > 4) return; // it was a drag (orbit)
  const [x, y] = curPos(ev);
  const i = nearestPoint(x, y);
  if (i == null) { $('bar').textContent = 'no visible point under the cursor'; return; }
  $('bar').textContent = 'selecting…';
  const r = await fetch('/api/pick', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tool: pickTool, index: i, reach: +$('pickReach').value }) });
  if (!r.ok) { $('bar').textContent = (await r.json()).detail || 'pick failed'; return; }
  const out = await r.json();
  if (!out.changed) { $('bar').textContent = out.message; return; }
  offer(out.indices, pickTool === 'object' ? +$('objClass').value : out.cls, ev.shiftKey);
});

// --- area tool: rectangle or lasso on screen, front surface only (unless "through")
function areaSelect(xa, ya, xb, yb, poly) {
  const { e, W, H } = frame(); const p = D.pos, lab = D.labels, n = D.n;
  const cs = 6, gw = Math.ceil((xb - xa) / cs) + 1, gh = Math.ceil((yb - ya) / cs) + 1;
  const minw = new Float32Array(gw * gh).fill(1e30);
  const cand = [], cw = [], cell = [];
  const only = +$('areaFrom').value;
  const inPoly = (px, py) => {
    let c = false;
    for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
      const [x1, y1] = poly[a], [x2, y2] = poly[b];
      if ((y1 > py) !== (y2 > py) && px < ((x2 - x1) * (py - y1)) / (y2 - y1) + x1) c = !c;
    }
    return c;
  };
  for (let i = 0; i < n; i++) {
    if (!vis[lab[i]]) continue;
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15]; if (w <= 0) continue;
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * 0.5 + 0.5) * W;
    if (sx < xa || sx > xb) continue;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * 0.5) * H;
    if (sy < ya || sy > yb) continue;
    if (poly && !inPoly(sx, sy)) continue;
    const c = Math.floor((sy - ya) / cs) * gw + Math.floor((sx - xa) / cs);
    cand.push(i); cw.push(w); cell.push(c); if (w < minw[c]) minw[c] = w;
  }
  const tol = +$('brushD').value / 10, through = $('areaThrough').checked, out = [];
  for (let k = 0; k < cand.length; k++) {
    if (!through && cw[k] > minw[cell[k]] + tol) continue;
    if (only >= 0 && lab[cand[k]] !== only) continue;
    out.push(cand[k]);
  }
  return out;
}
let lassoPts = null;
canvas.addEventListener('pointerdown', (ev) => {
  if (pickTool !== 'area' || !D || ev.button !== 0 || secMode) return;
  const [x, y] = curPos(ev);
  canvas.setPointerCapture(ev.pointerId);
  if ($('areaShape').value === 'lasso') { lassoPts = [[x, y]]; drawLasso(); return; }
  dragRect = { x0: x, y0: y, x1: x, y1: y }; drawRect();
});
canvas.addEventListener('pointermove', (ev) => {
  if (pickTool !== 'area') return;
  if (dragRect) { [dragRect.x1, dragRect.y1] = curPos(ev); drawRect(); }
  if (lassoPts) { const q = curPos(ev), l = lassoPts[lassoPts.length - 1]; if (Math.hypot(q[0] - l[0], q[1] - l[1]) > 3) { lassoPts.push(q); drawLasso(); } }
});
canvas.addEventListener('pointerup', (ev) => {
  if (pickTool !== 'area') return;
  let idx = null;
  if (dragRect) {
    const d = dragRect; dragRect = null; $('selrect').style.display = 'none';
    if (Math.abs(d.x1 - d.x0) < 4 || Math.abs(d.y1 - d.y0) < 4) return;
    idx = areaSelect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.max(d.x0, d.x1), Math.max(d.y0, d.y1), null);
  } else if (lassoPts) {
    const poly = lassoPts; lassoPts = null; $('lasso').style.display = 'none';
    if (poly.length < 4) return;
    const xs = poly.map((q) => q[0]), ys = poly.map((q) => q[1]);
    idx = areaSelect(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys), poly);
  } else return;
  if (!idx.length) { $('bar').textContent = 'nothing inside the selection'; return; }
  offer(idx, +$('areaTo').value, ev.shiftKey);
});
function drawLasso() {
  const s = $('lasso'); s.style.display = 'block';
  s.firstElementChild.setAttribute('points', lassoPts.map((q) => q.join(',')).join(' '));
}
function drawRect() {
  const r = $('selrect'), d = dragRect;
  r.style.display = 'block';
  r.style.left = Math.min(d.x0, d.x1) + 'px'; r.style.top = Math.min(d.y0, d.y1) + 'px';
  r.style.width = Math.abs(d.x1 - d.x0) + 'px'; r.style.height = Math.abs(d.y1 - d.y0) + 'px';
}

// ------------------------------------------------------------------ brush
let brush = false, painting = false, stroke = null, touched = null, lastEv = null, raf = 0;
const cur = $('brushcur');
function setBrush(on) {
  if (on) { setPickTool(null); if (treeTool) { treeTool = null; syncTreeBtns(); } }
  brush = on; $('brushBtn').classList.toggle('on', on);
  controls.mouseButtons.LEFT = on ? null : 'pivot';
  controls.mouseButtons.RIGHT = on ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN;
  controls.mouseButtons.MIDDLE = on ? THREE.MOUSE.PAN : THREE.MOUSE.DOLLY;
  cur.style.display = on ? 'block' : 'none';
  canvas.style.cursor = on ? 'none' : '';
}
$('brushBtn').onclick = () => setBrush(!brush);
function curPos(ev) { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; }
function moveCursor(ev) {
  const [x, y] = curPos(ev), R = +$('brushR').value;
  cur.style.width = cur.style.height = R * 2 + 'px';
  cur.style.left = x - R + 'px'; cur.style.top = y - R + 'px';
}
function paintAt(ev) {
  const [x, y] = curPos(ev);
  const { idx } = pick(x, y, +$('brushR').value, $('wholeCol').checked ? 1e9 : +$('brushD').value / 10, +$('fromClass').value);
  const to = +$('paintClass').value;
  const lab = D.labels;
  for (const i of idx) { if (lab[i] === to || touched[i]) continue; touched[i] = 1; stroke.push(i); lab[i] = to; }
  D.points.geometry.attributes.cls.needsUpdate = true;
}
canvas.addEventListener('pointerdown', (ev) => {
  if (secMode) return secPick(ev);
  if (!brush || !D || ev.button !== 0 || wasOn) return;
  painting = true; stroke = []; touched = new Uint8Array(D.n);
  canvas.setPointerCapture(ev.pointerId); paintAt(ev);
});
canvas.addEventListener('pointermove', (ev) => {
  if (brush) moveCursor(ev);
  if (!painting) return;
  lastEv = ev; if (!raf) raf = requestAnimationFrame(() => { raf = 0; paintAt(lastEv); });
});
canvas.addEventListener('pointerup', async () => {
  if (!painting) return; painting = false;
  if (!stroke.length) return;
  const to = +$('paintClass').value;
  await fetch('/api/relabel', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ indices: stroke, cls: to }) });
  $('undoBtn').disabled = false; refreshLegend();
  $('bar').textContent = `${stroke.length.toLocaleString()} voxels set to ${NAMES[to]}`;
});
$('undoBtn').onclick = async () => {
  if (wasOn) return;
  clearPending();
  const r = await (await fetch('/api/undo', { method: 'POST' })).json();
  if (!r.changed) { $('undoBtn').disabled = true; return; }
  for (let k = 0; k < r.indices.length; k++) D.labels[r.indices[k]] = r.labels[k];
  D.points.geometry.attributes.cls.needsUpdate = true; refreshLegend();
  $('bar').textContent = `undid ${r.changed.toLocaleString()} voxels`;
};
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.key === 'b' || e.key === 'B') setBrush(!brush);
  if (e.key === 'Enter' && pending) { e.preventDefault(); applyPending(); }
  if (e.key === 'Escape' && pending) { clearPending(); $('bar').textContent = 'selection cancelled'; }
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); $('undoBtn').click(); }
});

// ------------------------------------------------------------------ tree inventory
let trees = [], hidden = new Set(), treeTool = null, wasOn = false;
const circleGroup = new THREE.Group(); scene.add(circleGroup);
function syncTreeBtns() {
  $('circBtn').classList.toggle('on', treeTool === 'circle'); $('crownBtn').classList.toggle('on', treeTool === 'crown'); $('areaDrawBtn').classList.toggle('on', treeTool === 'area');
  canvas.style.cursor = treeTool ? 'crosshair' : (pickTool ? 'crosshair' : '');
}
function setTreeTool(t) {
  treeTool = treeTool === t ? null : t;
  if (treeTool) { setPickTool(null); if (brush) setBrush(false); }
  setMouseForTool(treeTool === 'area');
  if (treeTool === 'area') topView();
  syncTreeBtns();
  $('treeHint').textContent = treeTool === 'circle' ? 'Click a trunk: the diameter is fitted 1.0–1.6 m above the ground.'
    : treeTool === 'crown' ? 'Click the outer edge of a crown: sets that tree’s crown radius and height range.'
    : treeTool === 'area' ? 'Drag around the trees to search (top view). Draw several areas one after the other; then press Find stems.' : '';
}
$('circBtn').onclick = () => setTreeTool('circle');
$('crownBtn').onclick = () => setTreeTool('crown');
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  if (!r.ok) { $('bar').textContent = (await r.json()).detail || 'failed'; return null; }
  return r.json();
}
async function refreshTid() {
  const t = await getBin('/api/tid', Int16Array);
  if (!t || !D) return;
  const a = D.points.geometry.attributes.tid; a.array.set(Float32Array.from(t)); a.needsUpdate = true;
}
async function setTrees(list) {
  trees = list; rebuildCircles(); renderTable(); await refreshTid();
}
async function loadTrees() { const r = await (await fetch('/api/trees')).json(); await setTrees(r.trees); }
$('stemsBtn').onclick = async () => {
  $('bar').textContent = 'finding stems…';
  const r = await postJSON('/api/trees/find', { dbh_min: +$('dbhMin').value, dbh_max: +$('dbhMax').value, areas: stemAreas });
  if (!r) return;
  hidden.clear(); await setTrees(r.trees); $('bar').textContent = r.message;
  if (r.trees.length) setMode(5);
};
// (search area code below)
$('lastBtn').onclick = async () => { const r = await postJSON('/api/trees/remove_last'); if (r) await setTrees(r.trees); };
const hideQ = () => [...hidden].join(',');
$('csvBtn').onclick = () => (location.href = '/api/trees.csv?hide=' + hideQ());
$('dxfBtn').onclick = () => (location.href = '/api/trees.dxf?hide=' + hideQ());
canvas.addEventListener('pointerup', async (ev) => {
  if (!treeTool || !D || ev.button !== 0 || !downPos) return;
  if (Math.hypot(ev.clientX - downPos[0], ev.clientY - downPos[1]) > 4) return;
  const [x, y] = curPos(ev); const i = nearestPoint(x, y);
  if (i == null) { $('bar').textContent = 'no visible point under the cursor'; return; }
  const r = treeTool === 'circle' ? await postJSON('/api/trees/add', { index: i, radius: +$('circR').value })
    : await postJSON('/api/trees/crown', { index: i });
  if (!r) return;
  $('bar').textContent = r.message;
  if (r.trees) await setTrees(r.trees);
});

// --- search areas for Find stems (drawn in top view; polygons kept in scan coordinates)
let stemAreas = [], stemDrag = null, stemLasso = null;
const areaGroup = new THREE.Group(); scene.add(areaGroup);
function screenToWorld(x, y) {
  const r = canvas.getBoundingClientRect();
  const v = new THREE.Vector3((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1, 0.5).unproject(camera);
  const dir = v.sub(camera.position).normalize();
  const zp = (D.bbox[0][2] + D.bbox[1][2]) / 2;
  if (Math.abs(dir.z) < 1e-4) return null;
  const t = (zp - camera.position.z) / dir.z;
  return [camera.position.x + dir.x * t, camera.position.y + dir.y * t];
}
function drawAreaOutlines() {
  areaGroup.children.slice().forEach((c) => { areaGroup.remove(c); c.geometry.dispose(); });
  const zp = D ? D.bbox[1][2] + 0.5 : 0;
  stemAreas.forEach((poly) => {
    const l = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(poly.map(([x, y]) => new THREE.Vector3(x, y, zp))),
      new THREE.LineBasicMaterial({ color: 0x4ade80, depthTest: false }));
    l.renderOrder = 6; areaGroup.add(l);
  });
  $('areaInfo').textContent = stemAreas.length ? `Search area: ${stemAreas.length} drawn. Find stems looks only inside them.` : 'Search area: whole scan. Draw one or more areas in top view to limit Find stems to them.';
}
$('areaDrawBtn').onclick = () => setTreeTool('area');
$('areaClrBtn').onclick = () => { stemAreas = []; drawAreaOutlines(); };
canvas.addEventListener('pointerdown', (ev) => {
  if (treeTool !== 'area' || !D || ev.button !== 0) return;
  const [x, y] = curPos(ev); canvas.setPointerCapture(ev.pointerId);
  if ($('stemShape').value === 'lasso') { stemLasso = [[x, y]]; lassoPts = stemLasso; drawLasso(); } else { stemDrag = { x0: x, y0: y, x1: x, y1: y }; dragRect = stemDrag; drawRect(); }
});
canvas.addEventListener('pointermove', (ev) => {
  if (treeTool !== 'area') return;
  if (stemDrag) { [stemDrag.x1, stemDrag.y1] = curPos(ev); drawRect(); }
  if (stemLasso) { const q = curPos(ev), l = stemLasso[stemLasso.length - 1]; if (Math.hypot(q[0] - l[0], q[1] - l[1]) > 3) { stemLasso.push(q); drawLasso(); } }
});
canvas.addEventListener('pointerup', () => {
  if (treeTool !== 'area') return;
  let scr = null;
  if (stemDrag) {
    const d = stemDrag; stemDrag = null; dragRect = null; $('selrect').style.display = 'none';
    if (Math.abs(d.x1 - d.x0) > 6 && Math.abs(d.y1 - d.y0) > 6) scr = [[d.x0, d.y0], [d.x1, d.y0], [d.x1, d.y1], [d.x0, d.y1]];
  } else if (stemLasso) {
    const p = stemLasso; stemLasso = null; lassoPts = null; $('lasso').style.display = 'none';
    if (p.length > 4) { const step = Math.ceil(p.length / 80); scr = p.filter((_, i) => i % step === 0); }
  }
  if (!scr) return;
  const poly = scr.map(([x, y]) => screenToWorld(x, y));
  if (poly.some((q) => !q)) { $('bar').textContent = 'switch to Top view to draw the area'; return; }
  stemAreas.push(poly); drawAreaOutlines();
  $('bar').textContent = `search area ${stemAreas.length} added`;
});

function renderTable() {
  $('treeCount').textContent = trees.length;
  $('treePanel').style.display = trees.length ? 'flex' : 'none';
  const rows = trees.map((t) => `<tr data-id="${t.id}"><td><input type="checkbox" ${hidden.has(t.id) ? '' : 'checked'} data-v="${t.id}"></td>` +
    `<td>${t.id}${t.manual ? '*' : ''}</td><td>${t.dbh.toFixed(2)}</td><td>${t.h.toFixed(1)}</td><td>${t.z.toFixed(2)}</td><td class="x" data-d="${t.id}">✕</td></tr>`).join('');
  $('treeBody').innerHTML = `<table><tr><th></th><th>№</th><th>Ø m</th><th>h m</th><th>Z gr.</th><th></th></tr>${rows}</table>`;
  $('treeBody').querySelectorAll('tr[data-id]').forEach((tr) => {
    tr.onclick = (e) => {
      const id = +tr.dataset.id;
      if (e.target.dataset.d) { postJSON('/api/trees/delete', { id }).then((r) => r && setTrees(r.trees)); return; }
      if (e.target.dataset.v) { e.target.checked ? hidden.delete(id) : hidden.add(id); rebuildCircles(); return; }
      zoomTo(trees.find((t) => t.id === id));
    };
  });
}
$('treeHead').onclick = () => { const b = $('treeBody'); b.style.display = b.style.display === 'none' ? '' : 'none'; };
function zoomTo(t) {
  if (!t) return;
  const tg = new THREE.Vector3(t.cx, t.cy, t.cz + Math.max(1.3, t.h * 0.4));
  const d = Math.max(4, t.h * 1.6);
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (Math.abs(dir.z) > 0.95) dir.set(0.3, -0.9, 0.3).normalize();
  controls.target.copy(tg); camera.position.copy(tg).addScaledVector(dir, d); controls.update();
}
function rebuildCircles() {
  circleGroup.children.slice().forEach((c) => { circleGroup.remove(c); c.geometry.dispose(); });
  const mat = new THREE.LineBasicMaterial({ color: 0xff3b30, depthTest: false });
  trees.forEach((t) => {
    if (hidden.has(t.id)) return;
    const pts = []; for (let k = 0; k <= 48; k++) { const a = (k / 48) * Math.PI * 2; pts.push(new THREE.Vector3(t.cx + Math.cos(a) * t.dbh / 2, t.cy + Math.sin(a) * t.dbh / 2, t.cz + 1.3)); }
    const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat); l.renderOrder = 5; circleGroup.add(l);
    const ax = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(t.cx, t.cy, t.cz), new THREE.Vector3(t.cx, t.cy, t.cz + 1.3)]), mat);
    ax.renderOrder = 5; circleGroup.add(ax);
  });
  circleGroup.visible = $('tDiam').checked;
}
$('tDiam').onchange = () => (circleGroup.visible = $('tDiam').checked);
const tlabels = $('tlabels'), tlEls = new Map();
const tv = new THREE.Vector3();
function updateTreeLabels() {
  const on = $('tLab').checked && trees.length > 0 && trees.length <= 600;
  if (!on) { if (tlEls.size) { tlabels.innerHTML = ''; tlEls.clear(); } return; }
  const r = canvas.getBoundingClientRect();
  trees.forEach((t) => {
    let el = tlEls.get(t.id);
    if (hidden.has(t.id)) { if (el) el.style.display = 'none'; return; }
    if (!el) { el = document.createElement('div'); el.className = 'tl'; el.innerHTML = `№${t.id}<br>⌀${t.dbh.toFixed(2)}<br>${t.h.toFixed(1)}`; tlabels.appendChild(el); tlEls.set(t.id, el); }
    tv.set(t.cx, t.cy, t.cz + 1.3).project(camera);
    const vis_ = tv.z > -1 && tv.z < 1;
    el.style.display = vis_ ? '' : 'none';
    el.style.left = (tv.x * 0.5 + 0.5) * r.width + 'px'; el.style.top = (0.5 - tv.y * 0.5) * r.height + 'px';
  });
  tlEls.forEach((el, id) => { if (!trees.find((t) => t.id === id)) { el.remove(); tlEls.delete(id); } });
}
$('tLab').onchange = () => { tlabels.innerHTML = ''; tlEls.clear(); };
window.__treesReady = true;

// ------------------------------------------------------------------ view extras: ortho, x-ray, point count, "Was"
let ortho = false, orthoK = 1;
$('orthoBtn').onclick = () => {
  const dir = camera.position.clone().sub(controls.target); const d = dir.length(); dir.normalize();
  if (!ortho) {
    orthoK = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / Math.tan(THREE.MathUtils.degToRad(0.5));
    camera.userData.fov = camera.fov; camera.fov = 1;
    camera.position.copy(controls.target).addScaledVector(dir, d * orthoK); camera.near *= orthoK; camera.far *= orthoK;
  } else {
    camera.fov = camera.userData.fov;
    camera.position.copy(controls.target).addScaledVector(dir, d / orthoK); camera.near /= orthoK; camera.far /= orthoK;
  }
  ortho = !ortho; $('orthoBtn').classList.toggle('on', ortho);
  camera.updateProjectionMatrix(); controls.update();
};
$('pkeep').oninput = (e) => { material.uniforms.keep.value = e.target.value / 100; $('pkeepV').textContent = e.target.value + '%'; };
function applyXray() {
  const on = $('xray').checked;
  material.transparent = on; material.depthWrite = !on; material.uniforms.alpha.value = on ? $('xrayA').value / 100 : 1;
  material.needsUpdate = true;
}
$('xray').onchange = applyXray; $('xrayA').oninput = applyXray;
$('wasBtn').onclick = () => {
  if (!D) return;
  const a = D.points.geometry.attributes.cls;
  wasOn = !wasOn; $('wasBtn').classList.toggle('on', wasOn);
  if (wasOn) { D.edited = a.array.slice(); a.array.set(D.labels0); $('bar').textContent = 'showing the classification before your manual edits (editing is locked)'; }
  else { a.array.set(D.edited); $('bar').textContent = 'showing your edited version'; }
  a.needsUpdate = true; refreshLegend();
};

// ------------------------------------------------------------------ pivot orbit: left-drag rotates about the clicked point
const pivotDot = document.createElement('div');
pivotDot.style.cssText = 'position:absolute;width:12px;height:12px;margin:-6px 0 0 -6px;border:2px solid #fff04d;border-radius:50%;pointer-events:none;display:none;z-index:3;box-shadow:0 0 4px #000';
canvas.parentElement.appendChild(pivotDot);
let orbit = null;
canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0 || controls.mouseButtons.LEFT !== 'pivot' || secMode) return;
  const [x, y] = curPos(ev);
  const i = D ? nearestPoint(x, y) : null;
  const P = i == null ? controls.target.clone() : new THREE.Vector3(D.pos[i * 3], D.pos[i * 3 + 1], D.pos[i * 3 + 2]);
  orbit = { P, x: ev.clientX, y: ev.clientY, moved: false };
  canvas.setPointerCapture(ev.pointerId);
  const q = P.clone().project(camera);
  const r = canvas.getBoundingClientRect();
  pivotDot.style.left = ((q.x * 0.5 + 0.5) * r.width) + 'px'; pivotDot.style.top = ((0.5 - q.y * 0.5) * r.height) + 'px';
  pivotDot.style.display = 'block';
});
canvas.addEventListener('pointermove', (ev) => {
  if (!orbit) return;
  const dx = ev.clientX - orbit.x, dy = ev.clientY - orbit.y;
  orbit.x = ev.clientX; orbit.y = ev.clientY;
  if (!dx && !dy) return;
  orbit.moved = true;
  const P = orbit.P, k = 0.005;
  const Z = new THREE.Vector3(0, 0, 1);
  const spin = (q) => { camera.position.sub(P).applyQuaternion(q).add(P); camera.quaternion.premultiply(q); };
  spin(new THREE.Quaternion().setFromAxisAngle(Z, -dx * k));
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
  const before = camera.position.clone(), bq = camera.quaternion.clone();
  spin(new THREE.Quaternion().setFromAxisAngle(right, -dy * k));
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  if (Math.abs(fwd.z) > 0.9995) { camera.position.copy(before); camera.quaternion.copy(bq); } // do not flip over the pole
  camera.updateMatrixWorld();
  const f = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
  controls.target.copy(camera.position).addScaledVector(f, Math.max(0.5, P.clone().sub(camera.position).dot(f)));
  controls.update();
  const q = P.clone().project(camera), r = canvas.getBoundingClientRect();
  pivotDot.style.left = ((q.x * 0.5 + 0.5) * r.width) + 'px'; pivotDot.style.top = ((0.5 - q.y * 0.5) * r.height) + 'px';
});
const endOrbit = () => { if (orbit) { orbit = null; pivotDot.style.display = 'none'; } };
canvas.addEventListener('pointerup', endOrbit);
canvas.addEventListener('pointercancel', endOrbit);

// ------------------------------------------------------------------ section tool
let secMode = false, secA = null, secB = null, secData = null, measure = [];
const sec = $('section'), sc = $('secCanvas'), sctx = sc.getContext('2d');
$('secBtn').onclick = () => { sec.style.display = 'block'; startSecPick(); };
$('secPick').onclick = startSecPick;
$('secClose').onclick = () => { sec.style.display = 'none'; secMode = false; setBrushLock(false); };
function setBrushLock(on) { if (on && brush) setBrush(false); }
function startSecPick() {
  setPickTool(null);
  setBrushLock(true); secMode = true; secA = secB = null; measure = [];
  $('readout').textContent = 'Click the first point of the section line in the 3D view…';
  controls.mouseButtons.LEFT = null;
}
function secPick(ev) {
  const [x, y] = curPos(ev);
  const i = nearestPoint(x, y); if (i == null) return;
  const P = [D.pos[i * 3], D.pos[i * 3 + 1], D.pos[i * 3 + 2]];
  if (!secA) { secA = P; $('readout').textContent = 'Now click the second point…'; return; }
  secB = P; secMode = false; controls.mouseButtons.LEFT = brush ? null : 'pivot';
  buildSection();
}
$('secW').onchange = () => secB && buildSection();
$('secV').onchange = () => secB && drawSection();
function buildSection() {
  const hw = Math.max(0.05, +$('secW').value) / 2;
  let ux = secB[0] - secA[0], uy = secB[1] - secA[1];
  const L = Math.hypot(ux, uy) || 1e-6; ux /= L; uy /= L;
  const a = [], z = [], c = [], ix = [];
  const p = D.pos;
  for (let i = 0; i < D.n; i++) {
    if (!vis[D.labels[i]]) continue;
    const dx = p[i * 3] - secA[0], dy = p[i * 3 + 1] - secA[1];
    const al = dx * ux + dy * uy; if (al < 0 || al > L) continue;
    if (Math.abs(-dx * uy + dy * ux) > hw) continue;
    a.push(al); z.push(p[i * 3 + 2]); c.push(D.labels[i]); ix.push(i);
  }
  secData = { a, z, c, ix, L, z0: Math.min(secA[2], secB[2]) };
  measure = [];
  $('readout').textContent = `Section ${L.toFixed(2)} m long, ${(hw * 2).toFixed(2)} m wide, ${a.length.toLocaleString()} points. Click two points on the profile to measure.`;
  drawSection();
}
let view = null;
function drawSection() {
  if (!secData) return;
  const { a, z, c, L } = secData, W = sc.width, H = sc.height, m = 36;
  sctx.fillStyle = '#0a0c0f'; sctx.fillRect(0, 0, W, H);
  if (!a.length) { sctx.fillStyle = '#8a919c'; sctx.fillText('no points inside the strip', 20, 30); return; }
  // fit the vertical range to the 0.5–99.5 percentile so stray noise points don't flatten the profile
  const zs = Float32Array.from(z).sort();
  const zmin = zs[Math.floor(zs.length * 0.005)], zmax = zs[Math.min(zs.length - 1, Math.floor(zs.length * 0.995))];
  const ex = +$('secV').value;
  const s = Math.min((W - 2 * m) / L, (H - 2 * m) / ((zmax - zmin || 1) * ex));
  const ox = m, oy = H - m;
  view = { s, ox, oy, zmin, ex };
  // grid every nice step
  const step = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50].find((v) => v * s > 55) || 100;
  sctx.strokeStyle = '#1d2229'; sctx.fillStyle = '#6a7380'; sctx.font = '14px system-ui'; sctx.lineWidth = 1;
  for (let d = 0; d <= L; d += step) { const x = ox + d * s; sctx.beginPath(); sctx.moveTo(x, m / 2); sctx.lineTo(x, H - m); sctx.stroke(); sctx.fillText(d.toFixed(step < 1 ? 2 : 0), x - 8, H - 12); }
  for (let h = Math.ceil(zmin / step) * step; h <= zmax; h += step) { const y = oy - (h - zmin) * s * ex; sctx.beginPath(); sctx.moveTo(m, y); sctx.lineTo(W - m / 2, y); sctx.stroke(); sctx.fillText((h + center[2]).toFixed(step < 1 ? 2 : 1), 2, y - 3); }
  const cols = {}; for (const k in COLORS) cols[k] = COLORS[k];
  for (let i = 0; i < a.length; i++) {
    if (z[i] < zmin || z[i] > zmax + 1e-9) continue;
    sctx.fillStyle = cols[c[i]] || '#fff';
    sctx.fillRect(ox + a[i] * s - 1, oy - (z[i] - zmin) * s * ex - 1, 2.4, 2.4);
  }
  sctx.strokeStyle = '#fff'; sctx.fillStyle = '#fff'; sctx.lineWidth = 1.5;
  if (measure.length) {
    measure.forEach((q) => { const x = ox + q[0] * s, y = oy - (q[1] - zmin) * s * ex; sctx.beginPath(); sctx.arc(x, y, 5, 0, 7); sctx.stroke(); });
    if (measure.length === 2) {
      const [p, q] = measure;
      sctx.beginPath(); sctx.moveTo(ox + p[0] * s, oy - (p[1] - zmin) * s * ex); sctx.lineTo(ox + q[0] * s, oy - (q[1] - zmin) * s * ex); sctx.stroke();
    }
  }
}
function secXY(ev) {
  const r = sc.getBoundingClientRect();
  const px = (ev.clientX - r.left) * sc.width / r.width, py = (ev.clientY - r.top) * sc.height / r.height;
  return [(px - view.ox) / view.s, view.zmin + (view.oy - py) / (view.s * view.ex)];
}
sc.addEventListener('mousemove', (ev) => {
  if (!view || measure.length === 2) return;
  const [d, h] = secXY(ev);
  $('readout').textContent = `along ${d.toFixed(3)} m   |   level ${(h + center[2]).toFixed(3)} m` + (measure.length === 1 ? `   |   Δ from first: ${(d - measure[0][0]).toFixed(3)} m along, ${(h - measure[0][1]).toFixed(3)} m vertical` : '');
});
sc.addEventListener('click', (ev) => {
  if (!view) return;
  if (measure.length === 2) measure = [];
  measure.push(secXY(ev));
  if (measure.length === 2) {
    const [p, q] = measure, dx = q[0] - p[0], dz = q[1] - p[1];
    $('readout').textContent = `horizontal ${Math.abs(dx).toFixed(3)} m · vertical ${dz.toFixed(3)} m · slope distance ${Math.hypot(dx, dz).toFixed(3)} m` + (Math.abs(dx) > 1e-6 ? ` · gradient ${(dz / Math.abs(dx) * 100).toFixed(1)} %` : '');
  }
  drawSection();
});

// ------------------------------------------------------------------ boot
(async () => {
  await loadClasses();
  const profs = await (await fetch('/api/profiles')).json();
  const names = { static_scan: 'Static laser scan (e.g. BLK360)', drone_photogrammetry: 'Drone photogrammetry' };
  $('profile').innerHTML = Object.keys(profs).map((k) => `<option value="${k}">${names[k] || k}</option>`).join('');
  const st = await (await fetch('/api/status')).json();
  if (st.state === 'loading' || st.state === 'classifying') { const s2 = await poll(); if (s2.state === 'ready') await afterReady(s2); }
  else if (st.state === 'ready' && st.n_points) await afterReady(st);
})();
window.__terrasort = { get D() { return D; }, setMode, camera, controls, setPickTool, get pending() { return pending; }, get trees() { return trees; } };

function applyAdj() {
  const u = material.uniforms;
  u.adj.value.set(+$('adjB').value / 100, +$('adjC').value / 100, +$('adjG').value / 100, 0);
  u.clip.value.set(+$('adjLo').value / 100, +$('adjHi').value / 100);
}
['adjB', 'adjC', 'adjG', 'adjLo', 'adjHi'].forEach((id) => ($(id).oninput = applyAdj));
$('adjR').onclick = () => { $('adjB').value = 0; $('adjC').value = 100; $('adjG').value = 100; $('adjLo').value = 0; $('adjHi').value = 100; applyAdj(); };

function setAllVis(f) {
  const c = counts();
  Object.keys(NAMES).map(Number).forEach((k) => { if (k !== 0) vis[k] = f(k, c) ? 1 : 0; });
  vis[0] = vis[0]; updatePalette(); refreshLegend();
}
$('visAll').onclick = () => D && setAllVis(() => true);
$('visNone').onclick = () => D && setAllVis(() => false);
$('visInv').onclick = () => D && setAllVis((k, c) => (c[k] ? !vis[k] : vis[k]));

// ------------------------------------------------------------------ CAD export of the section and its plan line (DXF is written by the server)
async function saveResponse(r, info) {
  if (!r.ok) { info.textContent = 'Failed: ' + ((await r.json().catch(() => ({}))).detail || r.status); return null; }
  const name = /filename="([^"]+)"/.exec(r.headers.get('Content-Disposition') || '')?.[1] || 'terrasort.dat';
  const blob = await r.blob();
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
  return { name, mb: blob.size / 1e6 };
}
const secBody = () => ({
  a: secA, b: secB, width: +$('secW').value, vex: +$('secV').value, color: $('secCol').value, measure: measure.map((m) => [m[0], m[1]]),
  hidden: [...Array(256).keys()].filter((k) => !vis[k]),
  colors: Object.fromEntries(Object.keys(NAMES).map((k) => [k, hexInt(COLORS[k] || '#ffffff')])), names: NAMES,
});
const hexInt = (h) => parseInt(h.slice(1), 16);
async function secExport(url, what) {
  if (!secA || !secB) { $('readout').textContent = 'Pick a section first.'; return; }
  $('readout').textContent = `Building ${what}…`;
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(secBody()) });
  const s = await saveResponse(r, $('readout'));
  if (s) $('readout').textContent = `${s.name} saved to your Downloads folder (${s.mb.toFixed(1)} MB).`;
}
$('secDxf').onclick = () => secExport('/api/section.dxf', 'section DXF');
$('secPlan').onclick = () => secExport('/api/section_line.dxf', 'section line DXF');

// ------------------------------------------------------------------ ground model: DTM / contours files and viewer overlay
function groundQuery() {
  const cls = [['gG', 2], ['gK', 64], ['gL', 73]].filter(([id]) => $(id).checked).map(([, k]) => k).join(',');
  const csv = csvPts && $('csvUse').checked ? `&csv=1&prio=${$('csvPrio').checked ? 1 : 0}` : '';
  return `classes=${cls || 2}&cell=${+$('dtmCell').value || 0.25}&gap=${+$('dtmGap').value || 10}${csv}`;
}
const conQuery = () => `${groundQuery()}&interval=${+$('conInt').value || 0.25}&major=${+$('conMajor').value || 5}&smooth=${+$('conSmooth').value || 0}`;
async function groundDownload(urls, label) {
  const bar = $('dtmInfo'); bar.textContent = `Building ${label}… (a few seconds to a minute)`;
  try {
    const done = [];
    for (const u of urls) { const s = await saveResponse(await fetch(u), bar); if (!s) return; done.push(`${s.name} (${s.mb.toFixed(1)} MB)`); }
    bar.textContent = `Saved to your Downloads folder: ${done.join(', ')}. If the browser asks to allow multiple downloads, allow it.`;
  } catch (e) { bar.textContent = 'Failed: ' + e; }
}
$('dtmBtn').onclick = () => {
  const kind = $('dtmKind').value, ep = +$('dtmEpsg').value || 0;
  groundDownload([`/api/dtm.tif?${groundQuery()}&kind=${kind}&epsg=${ep}`, `/api/dtm.tfw?${groundQuery()}&kind=${kind}`], 'DTM TIFF + world file');
};
$('conBtn').onclick = () => groundDownload([`/api/contours.dxf?${conQuery()}`], 'contours DXF');

// viewer overlay: the surface as a mesh and the contours as lines, drawn with the point cloud
let dtmMesh = null, conLines = null;
function clearGround() {
  for (const o of [dtmMesh, conLines]) if (o) { scene.remove(o); o.traverse?.((c) => { c.geometry?.dispose(); c.material?.dispose?.(); }); }
  dtmMesh = conLines = null;
}
function applyGroundVis() {
  if (dtmMesh) { dtmMesh.visible = $('showDtm').checked; dtmMesh.material.opacity = +$('dtmOp').value / 100; dtmMesh.material.transparent = +$('dtmOp').value < 100; }
  if (conLines) conLines.visible = $('showCon').checked;
}
['showDtm', 'showCon'].forEach((id) => ($(id).onchange = applyGroundVis));
$('dtmOp').oninput = applyGroundVis;
async function showGround() {
  const bar = $('dtmInfo'); bar.textContent = 'Building surface and contours for the viewer…';
  try {
    clearGround();
    const rd = await fetch(`/api/ground_view/dtm?${groundQuery()}`);
    if (!rd.ok) { bar.textContent = 'Failed: ' + ((await rd.json().catch(() => ({}))).detail || rd.status); return; }
    const g = JSON.parse(rd.headers.get('X-Grid')); const Z = new Float32Array(await rd.arrayBuffer());
    // thin the grid for display: at most ~700k vertices
    const st = Math.max(1, Math.ceil(Math.sqrt((g.nx * g.ny) / 700000)));
    const nx = Math.ceil(g.nx / st), ny = Math.ceil(g.ny / st);
    const pos = new Float32Array(nx * ny * 3), col = new Float32Array(nx * ny * 3), zz = new Float32Array(nx * ny).fill(NaN);
    let zmin = 1e30, zmax = -1e30;
    for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
      const z = Z[Math.min(r * st, g.ny - 1) * g.nx + Math.min(c * st, g.nx - 1)]; zz[r * nx + c] = z;
      if (z === z) { if (z < zmin) zmin = z; if (z > zmax) zmax = z; }
    }
    const cs = g.cell * st, ramp = (t) => [0.2 + 0.7 * t, 0.35 + 0.55 * Math.sin(Math.PI * t), 0.8 - 0.6 * t];
    for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
      const k = r * nx + c, z = zz[k];
      pos[k * 3] = g.x0 + (Math.min(c * st, g.nx - 1) + 0.5) * g.cell; pos[k * 3 + 1] = g.ytop - (Math.min(r * st, g.ny - 1) + 0.5) * g.cell; pos[k * 3 + 2] = (z === z ? z : 0) - g.zc + 0.05;
      const zr = zz[r * nx + Math.min(c + 1, nx - 1)], zl = zz[r * nx + Math.max(c - 1, 0)], zu = zz[Math.max(r - 1, 0) * nx + c], zd = zz[Math.min(r + 1, ny - 1) * nx + c];
      const gx = ((zr === zr ? zr : z) - (zl === zl ? zl : z)) / (2 * cs), gy = ((zu === zu ? zu : z) - (zd === zd ? zd : z)) / (2 * cs);
      const sh = Math.max(0, Math.min(1, (0.7 + 0.7 * (-gx * 0.5 + gy * 0.5)) / Math.sqrt(1 + gx * gx + gy * gy))) * 0.65 + 0.35;
      const rgb = ramp((z - zmin) / Math.max(zmax - zmin, 1e-6));
      col[k * 3] = rgb[0] * sh; col[k * 3 + 1] = rgb[1] * sh; col[k * 3 + 2] = rgb[2] * sh;
    }
    const idx = [];
    for (let r = 0; r < ny - 1; r++) for (let c = 0; c < nx - 1; c++) {
      const a = r * nx + c, b = a + 1, d = a + nx, e = d + 1;
      if (zz[a] !== zz[a] || zz[b] !== zz[b] || zz[d] !== zz[d] || zz[e] !== zz[e]) continue;
      idx.push(a, d, b, b, d, e);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx);
    dtmMesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    dtmMesh.frustumCulled = false; scene.add(dtmMesh);
    const rc = await fetch(`/api/ground_view/contours?${conQuery()}`);
    if (rc.ok) {
      const buf = await rc.arrayBuffer(), h = new Uint32Array(buf, 0, 2);
      const minor = new Float32Array(buf, 8, h[0]), major = new Float32Array(buf, 8 + h[0] * 4, h[1]);
      conLines = new THREE.Group();
      for (const [arr, color] of [[minor, 0xffd54a], [major, 0xff8a1f]]) {
        if (!arr.length) continue;
        const lg = new THREE.BufferGeometry(); const up = arr.slice(); for (let q = 2; q < up.length; q += 3) up[q] += 0.1; lg.setAttribute('position', new THREE.BufferAttribute(up, 3));
        const ls = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color })); ls.frustumCulled = false; conLines.add(ls);
      }
      scene.add(conLines);
    }
    $('showDtm').checked = $('showCon').checked = true; applyGroundVis();
    bar.textContent = `Surface ${g.nx}×${g.ny} cells shown (every ${st > 1 ? st + 'th' : ''} cell), contours ${(+$('conInt').value || 0.25)} m. Use the tick boxes to show/hide; Clear removes them.`;
  } catch (e) { bar.textContent = 'Failed: ' + e; }
}
$('gvBtn').onclick = showGround;
let gTimer = 0;
/** called after any class edit or CSV edit: rebuild the surface and contours if they are on screen */
window.__groundAuto = () => {
  if (!dtmMesh || !$('gvAuto').checked) return;
  clearTimeout(gTimer); gTimer = setTimeout(showGround, 700);
};
$('gvClr').onclick = () => { clearGround(); $('dtmInfo').textContent = 'Cleared.'; };

// ------------------------------------------------------------------ CSV survey points added to the ground model
let csvPts = null, csvText = '', csvInfo = null;
$('csvLoad').onclick = () => $('csvFile').click();
$('csvFile').onchange = async (ev) => {
  const f = ev.target.files[0]; ev.target.value = ''; if (!f) return;
  const bar = $('dtmInfo');
  csvText = await f.text();
  const r = await fetch('/api/csv/preview', { method: 'POST', body: csvText });
  if (!r.ok) { bar.textContent = 'CSV: ' + ((await r.json().catch(() => ({}))).detail || r.status); return; }
  csvInfo = await r.json();
  const opts = Array.from({ length: csvInfo.ncol }, (_, i) => `<option value="${i}">${i + 1}: ${(csvInfo.names[i] || '').slice(0, 14)}</option>`).join('');
  ['cvX', 'cvY', 'cvZ'].forEach((id, k) => { $(id).innerHTML = opts; const g = csvInfo.guess['xyz'[k]]; if (g != null) $(id).value = g; });
  $('cvHead').checked = csvInfo.header;
  $('cvPrev').textContent = `${f.name}: ${csvInfo.lines.toLocaleString()} lines. First rows: ` + csvInfo.rows.slice(0, 3).map((r) => r.slice(0, 6).join(' | ')).join('   ·   ') + '. Check the X, Y and Z columns, then press Use points.';
  $('csvMap').style.display = 'block';
};
let csvObj = null, csvAbs = null, csvOff = [0, 0, 0], csvDel = false;
const csvUndo = [];
function csvBuild(fit) {
  if (csvObj) { scene.remove(csvObj); csvObj.geometry.dispose(); csvObj = null; }
  if (!csvAbs) { if (!D) $('empty').style.display = ''; return; }
  $('empty').style.display = 'none';
  const n = csvAbs.length / 3;
  let off = center;
  if (!D) { off = [0, 0, 0]; for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) off[d] += csvAbs[i * 3 + d] / n; center = off; }
  csvOff = off.slice();
  const pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) pos[i * 3 + d] = csvAbs[i * 3 + d] - off[d];
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  csvObj = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x00e5ff, size: 6, sizeAttenuation: false })); csvObj.frustumCulled = false; scene.add(csvObj);
  if (fit && !D) {
    let mn = [1e30, 1e30, 1e30], mx = [-1e30, -1e30, -1e30];
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) { const v = pos[i * 3 + d]; if (v < mn[d]) mn[d] = v; if (v > mx[d]) mx[d] = v; }
    const ext = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1), c = [0, 1, 2].map((d) => (mn[d] + mx[d]) / 2);
    controls.target.set(...c); camera.up.set(0, 0, 1);
    camera.position.set(c[0], c[1] - ext * 0.05, c[2] + ext * 1.2);
    camera.near = ext * 0.001; camera.far = ext * 30; camera.updateProjectionMatrix(); controls.update();
  }
}
$('cvUse').onclick = async () => {
  const bar = $('dtmInfo'); bar.textContent = 'Loading CSV points…';
  const r = await fetch('/api/csv/load', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text: csvText, x: +$('cvX').value, y: +$('cvY').value, z: +$('cvZ').value, delim: csvInfo.delim, header: $('cvHead').checked }) });
  const j = await r.json();
  if (!r.ok) { bar.textContent = 'CSV: ' + (j.detail || r.status); return; }
  csvPts = j.n; $('csvMap').style.display = 'none'; $('csvClr').disabled = false; $('csvUse').checked = true;
  csvAbs = new Float64Array(j.abs); csvUndo.length = 0; csvBuild(true); csvBtns();
  bar.textContent = `${j.n.toLocaleString()} CSV points loaded (levels ${j.zrange[0].toFixed(2)} to ${j.zrange[1].toFixed(2)}), shown in cyan. ` + (j.warn || (D ? 'They are used for the DTM and contours; press Show in viewer to see the result.' : 'Now open and classify your scan: the points stay loaded and are used for the DTM.'));
};
$('csvClr').onclick = async () => {
  await fetch('/api/csv/clear', { method: 'POST' });
  csvPts = null; $('csvClr').disabled = true;
  csvAbs = null; csvUndo.length = 0; csvDel = false; csvBuild(); csvBtns();
  $('dtmInfo').textContent = 'CSV points removed.';
};
window.__csvRebuild = () => { if (csvAbs) csvBuild(false); };

// ------------------------------------------------------------------ delete individual CSV points by clicking them
function csvBtns() {
  $('csvDelBtn').disabled = !csvAbs; $('csvDelBtn').classList.toggle('on', csvDel && !!csvAbs);
  $('csvUndoBtn').disabled = !csvUndo.length;
  canvas.style.cursor = csvDel && csvAbs ? 'crosshair' : '';
}
$('csvDelBtn').onclick = () => { csvDel = !csvDel; csvBtns(); $('dtmInfo').textContent = csvDel ? 'Delete mode: click a cyan CSV point to remove it (drag still rotates). Click the button again to stop.' : 'Delete mode off.'; };
$('csvUndoBtn').onclick = async () => {
  const u = csvUndo.pop(); if (!u) return;
  await fetch('/api/csv/add', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pts: u }) });
  const a = new Float64Array(csvAbs.length + u.length * 3); a.set(csvAbs); u.forEach((p, k) => a.set(p, csvAbs.length + k * 3));
  csvAbs = a; csvPts = a.length / 3; csvBuild(false); csvBtns();
  $('dtmInfo').textContent = `Restored ${u.length} point(s); ${csvPts.toLocaleString()} CSV points. Press Show in viewer to update the surface.`;
  if (window.__groundAuto) window.__groundAuto();
};
function csvNearest(mx, my, rad) {
  const { e, W, H } = frame(); let best = -1, bd = rad * rad; const n = csvAbs.length / 3;
  for (let i = 0; i < n; i++) {
    const x = csvAbs[i * 3] - csvOff[0], y = csvAbs[i * 3 + 1] - csvOff[1], z = csvAbs[i * 3 + 2] - csvOff[2];
    const w = e[3] * x + e[7] * y + e[11] * z + e[15]; if (w <= 0) continue;
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / w * 0.5 + 0.5) * W - mx;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * 0.5) * H - my;
    const d = sx * sx + sy * sy; if (d < bd) { bd = d; best = i; }
  }
  return best;
}
let csvDown = null;
canvas.addEventListener('pointerdown', (ev) => { csvDown = [ev.clientX, ev.clientY]; });
canvas.addEventListener('pointerup', async (ev) => {
  if (!csvDel || !csvAbs || ev.button !== 0 || !csvDown) return;
  if (Math.hypot(ev.clientX - csvDown[0], ev.clientY - csvDown[1]) > 4) return;
  const [x, y] = curPos(ev); const i = csvNearest(x, y, 14);
  if (i < 0) { $('dtmInfo').textContent = 'No CSV point near the click (click closer to a cyan point).'; return; }
  const p = Array.from(csvAbs.subarray(i * 3, i * 3 + 3));
  await fetch('/api/csv/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ idx: [i] }) });
  const a = new Float64Array(csvAbs.length - 3); a.set(csvAbs.subarray(0, i * 3)); a.set(csvAbs.subarray(i * 3 + 3), i * 3);
  csvAbs = a; csvPts = a.length / 3; csvUndo.push([p]); csvBuild(false); csvBtns();
  $('dtmInfo').textContent = `Removed point at level ${p[2].toFixed(3)}. ${csvPts.toLocaleString()} CSV points left. ` + ($('gvAuto').checked && dtmMesh ? 'Updating surface…' : 'Press Show in viewer to update the surface.');
  if (window.__groundAuto) window.__groundAuto();
});
