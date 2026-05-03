"""Timelapse HTML template — cinematic vessel corridor visualization.

Produces a standalone HTML file with Three.js WebGL rendering over a
MapLibre GL dark basemap. Line segments with custom glow shaders and
UnrealBloomPass post-processing create neon-tube corridor effects.

Assembled from named ``string.Template`` sections and rendered by
:func:`viz.generate_timelapse`. Uses ``$var`` substitution so that
JavaScript braces are literal.

This module is internal (``_`` prefix). Import only via
``viz.generate_timelapse()``.
"""

from __future__ import annotations

from string import Template

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
  background: #050510; color: #e0e0e0; overflow: hidden;
}
#container { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }

/* ── single panel ──────────────────────────── */
.panel-cell { position: relative; width: 100%; height: 100%; overflow: hidden; }
.panel-map {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  filter: brightness(0.35) saturate(0.3);
}
.panel-darken {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(5, 5, 16, 0.4);
  pointer-events: none; z-index: 1;
}
.panel-overlay {
  position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  pointer-events: none; z-index: 2;
}
.panel-overlay canvas {
  position: absolute; top: 0; left: 0;
  pointer-events: none;
  mix-blend-mode: screen;
}

/* ── multi-panel grid ──────────────────────── */
.grid-vertical {
  display: grid; width: 100%; height: 100vh;
}
.grid-horizontal {
  display: grid; height: 100vh;
}
.grid-auto {
  display: grid; width: 100%; height: 100vh;
}
.panel-label {
  position: absolute; top: 12px; left: 16px; z-index: 10;
}
.panel-label h3 {
  font-size: 16px; font-weight: 700; letter-spacing: 1px;
  text-shadow: 0 1px 8px rgba(0,0,0,0.8);
}
.panel-label p {
  font-size: 10px; color: rgba(255,255,255,0.5);
  text-transform: uppercase; letter-spacing: 0.5px;
}

/* ── overlays ──────────────────────────────── */
.overlay-panel {
  position: absolute; z-index: 20;
  background: rgba(5, 5, 16, 0.75);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px; padding: 12px 16px;
  backdrop-filter: blur(10px);
}
#title-panel { top: 20px; left: 20px; }
#title-panel h1 {
  font-size: 14px; font-weight: 700; letter-spacing: 2px;
  text-transform: uppercase;
}
#title-panel .subtitle {
  font-size: 11px; color: rgba(255,255,255,0.45);
  letter-spacing: 1px; margin-top: 2px;
}
#vessel-count-panel { top: 20px; left: 20px; margin-top: 70px; }
.big-num {
  font-size: 36px; font-weight: 700; line-height: 1;
  font-variant-numeric: tabular-nums; color: #00c8ff;
}
.count-label {
  font-size: 9px; text-transform: uppercase;
  letter-spacing: 1px; color: rgba(255,255,255,0.4); margin-top: 2px;
}
#timestamp-panel { top: 20px; right: 20px; }
#timestamp {
  font-size: 15px; font-weight: 600;
  font-variant-numeric: tabular-nums; color: rgba(255,255,255,0.7);
}

/* ── legend ────────────────────────────────── */
#legend {
  position: absolute; bottom: 60px; left: 20px; z-index: 20;
  display: flex; gap: 12px; flex-wrap: wrap;
  background: rgba(5, 5, 16, 0.6); border-radius: 6px;
  padding: 8px 12px;
}
.legend-item {
  display: flex; align-items: center; gap: 5px;
  font-size: 10px; color: rgba(255,255,255,0.6);
  text-transform: capitalize;
}
.legend-dot {
  width: 8px; height: 8px; border-radius: 50%;
}

/* ── controls ──────────────────────────────── */
#controls {
  position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
  z-index: 20; display: flex; align-items: center; gap: 10px;
  background: rgba(5, 5, 16, 0.75); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px; padding: 8px 16px;
  backdrop-filter: blur(10px);
}
#controls button {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15);
  color: #ccc; padding: 4px 10px; border-radius: 4px; cursor: pointer;
  font-size: 12px; font-family: inherit; transition: all 0.15s;
}
#controls button:hover { background: rgba(255,255,255,0.15); color: #fff; }
#controls button.active {
  background: rgba(0,200,255,0.25); border-color: rgba(0,200,255,0.4); color: #00c8ff;
}
#slider {
  width: 300px; accent-color: #00c8ff; height: 3px;
}
#speed-label {
  font-size: 11px; color: rgba(255,255,255,0.5);
  font-variant-numeric: tabular-nums; min-width: 32px;
}

/* ── maplibre overrides ────────────────────── */
.maplibregl-ctrl-bottom-left,
.maplibregl-ctrl-bottom-right { display: none !important; }
"""

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_IMPORTMAP = """\
<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.170.0/build/three.module.js",
  "three/addons/":"https://unpkg.com/three@0.170.0/examples/jsm/"
}}
</script>
"""

_SINGLE_PANEL_HTML = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — $title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
""" + _IMPORTMAP + """
<style>
""" + _CSS + """
</style>
</head>
<body>
<div id="container">
  <div class="panel-cell" id="panel-0">
    <div class="panel-map" id="map-0"></div>
    <div class="panel-darken"></div>
    <div class="panel-overlay" id="display-0"></div>
  </div>
</div>

<div class="overlay-panel" id="title-panel">
  <h1>$title</h1>
  <p class="subtitle">$subtitle</p>
</div>

<div class="overlay-panel" id="vessel-count-panel">
  <div class="big-num" id="vessel-count">0</div>
  <div class="count-label">UNIQUE VESSELS</div>
</div>

<div class="overlay-panel" id="timestamp-panel">
  <span id="timestamp">—</span>
</div>

<div id="legend">$legend_html</div>

<div id="controls">
  <button id="play-btn" class="active" title="Play / Pause (Space)">&#9654;</button>
  <input type="range" id="slider" min="0" max="1000" value="0">
  <button id="speed-btn" title="Playback speed">$speed_label</button>
  <span id="speed-label">$speed_label</span>
</div>

<script type="module">
""")

_MULTI_PANEL_HTML = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — $title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
""" + _IMPORTMAP + """
<style>
""" + _CSS + """
#container {
  $grid_style
}
</style>
</head>
<body>
<div id="container">
  $panel_cells
</div>

<div class="overlay-panel" id="title-panel">
  <h1>$title</h1>
  <p class="subtitle">$subtitle</p>
</div>

<div class="overlay-panel" id="timestamp-panel">
  <span id="timestamp">—</span>
</div>

<div id="legend">$legend_html</div>

<div id="controls">
  <button id="play-btn" class="active" title="Play / Pause (Space)">&#9654;</button>
  <input type="range" id="slider" min="0" max="1000" value="0">
  <button id="speed-btn" title="Playback speed">$speed_label</button>
  <span id="speed-label">$speed_label</span>
</div>

<script type="module">
""")

# ---------------------------------------------------------------------------
# JavaScript — data sections (unchanged from Canvas 2D version)
# ---------------------------------------------------------------------------

_JS_SINGLE_DATA = Template("""\
// ── Data ──────────────────────────────────────
const BINS = $bins_json;
const CUMUL_VESSELS = $cumul_vessels_json;
const BIN_TIMESTAMPS = $bin_timestamps_ms_json;
const PALETTE = $palette_json;
const TYPE_NAMES = $type_names_json;
const COLOR_BY_TYPE = $color_by_type;
const COLOR_MODE = "$color_mode";
const CONFIG = {
  dotRadius: $dot_radius,
  dotAlpha: $dot_alpha,
  fadeFactor: $fade_factor,
  bloom: $bloom,
  speed: $speed,
  style: "$style",
  showClock: $show_clock,
};
const DATE_RANGE_LABEL = "$date_range_label";
const N_PANELS = 1;
const PANELS_CFG = [{
  centerLat: $center_lat,
  centerLon: $center_lon,
  zoom: $zoom,
  bins: BINS,
  cumulVessels: CUMUL_VESSELS,
  binTimestamps: BIN_TIMESTAMPS,
  dotRadius: CONFIG.dotRadius,
  dotAlpha: CONFIG.dotAlpha,
  fadeFactor: CONFIG.fadeFactor,
  bloom: CONFIG.bloom,
  style: CONFIG.style,
}];
""")

_JS_MULTI_DATA = Template("""\
// ── Data ──────────────────────────────────────
const PALETTE = $palette_json;
const TYPE_NAMES = $type_names_json;
const COLOR_BY_TYPE = $color_by_type;
const COLOR_MODE = "$color_mode";
const CONFIG = {
  dotRadius: $dot_radius,
  dotAlpha: $dot_alpha,
  fadeFactor: $fade_factor,
  bloom: $bloom,
  speed: $speed,
  style: "$style",
  showClock: $show_clock,
};
const DATE_RANGE_LABEL = "$date_range_label";
const N_PANELS = $n_panels;
const PANELS_RAW = $panels_json;
const PANELS_CFG = PANELS_RAW.map(function(p) {
  return {
    centerLat: p.center_lat,
    centerLon: p.center_lon,
    zoom: p.zoom,
    bins: JSON.parse(p.bins_json || '[]'),
    cumulVessels: JSON.parse(p.cumul_vessels_json || '[]'),
    binTimestamps: JSON.parse(p.bin_timestamps_ms_json || '[]'),
    dotRadius: (p.config && p.config.dot_radius) || CONFIG.dotRadius,
    dotAlpha: (p.config && p.config.dot_alpha) || CONFIG.dotAlpha,
    fadeFactor: (p.config && p.config.fade_factor) || CONFIG.fadeFactor,
    bloom: (p.config && p.config.bloom !== undefined) ? p.config.bloom : CONFIG.bloom,
    style: (p.config && p.config.style) || CONFIG.style,
  };
});
""")

# ---------------------------------------------------------------------------
# JavaScript — Three.js WebGL engine
# ---------------------------------------------------------------------------

_JS_ENGINE = """\
// ── Three.js WebGL engine ─────────────────────
// Two-layer rendering with per-vessel trailing polylines:
//   accumulation RT — persistent corridor traces (dim, slow fade)
//   active RT — bright vessel trails (last ~25 positions) + head dots
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// ── Shaders ───────────────────────────────────
// Line segment quad shader — used for both accumulation and trail rendering
const LINE_VERT = `
attribute vec2 aStart;
attribute vec2 aEnd;
attribute float aQuadVertex;
attribute vec3 aColor;
attribute float aAlpha;
varying float vDist;
varying vec3 vColor;
varying float vAlpha;
uniform vec2 uResolution;
uniform float uLineWidth;
void main() {
  vec2 dir = aEnd - aStart;
  float len = length(dir);
  if (len < 0.001) { gl_Position = vec4(2.0,2.0,0.0,1.0); return; }
  vec2 fwd = dir / len;
  vec2 nrm = vec2(-fwd.y, fwd.x);
  float side = (mod(aQuadVertex, 2.0) < 0.5) ? -1.0 : 1.0;
  float along = (aQuadVertex < 1.5) ? 0.0 : 1.0;
  vec2 pos = mix(aStart, aEnd, along) + nrm * side * uLineWidth * 0.5;
  vec2 ndc = (pos / uResolution) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);
  vDist = side;
  vColor = aColor;
  vAlpha = aAlpha;
}`;

// Corridor accumulation — wide soft glow so overlapping halos merge into bands
const LINE_FRAG_ACCUM = `
precision highp float;
varying float vDist;
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = abs(vDist);
  float core = smoothstep(0.25, 0.0, d);
  float glow = exp(-d * d * 2.0);
  float intensity = core * 0.6 + glow * 0.5;
  gl_FragColor = vec4(vColor * intensity * 1.8 * vAlpha, intensity * vAlpha);
}`;

// Active trail — brighter, more vivid for moving vessel trails
const LINE_FRAG_TRAIL = `
precision highp float;
varying float vDist;
varying vec3 vColor;
varying float vAlpha;
void main() {
  float d = abs(vDist);
  float core = smoothstep(0.25, 0.0, d);
  float glow = exp(-d * d * 3.5);
  float intensity = core * 1.0 + glow * 0.4;
  gl_FragColor = vec4(vColor * intensity * 2.0 * vAlpha, intensity * vAlpha);
}`;

const FADE_FRAG = `
precision highp float;
uniform sampler2D tDiffuse;
uniform float uFadeFactor;
varying vec2 vUv;
void main() {
  gl_FragColor = texture2D(tDiffuse, vUv) * uFadeFactor;
}`;

const COMPOSITE_FRAG = `
precision highp float;
uniform sampler2D tAccum;
uniform sampler2D tActive;
varying vec2 vUv;
void main() {
  gl_FragColor = texture2D(tAccum, vUv) + texture2D(tActive, vUv);
}`;

const HEAD_VERT = `
precision highp float;
attribute vec3 position;
attribute vec3 aHeadColor;
varying vec3 vColor;
uniform float uPointSize;
uniform vec2 uResolution;
void main() {
  vColor = aHeadColor;
  gl_PointSize = uPointSize;
  vec2 ndc = (position.xy / uResolution) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

const HEAD_FRAG = `
precision highp float;
varying vec3 vColor;
void main() {
  float d = length(gl_PointCoord - 0.5) * 2.0;
  if (d > 1.0) discard;
  float core = smoothstep(0.25, 0.0, d);
  float glow = exp(-d * d * 3.0);
  float intensity = core * 0.8 + glow * 0.4;
  gl_FragColor = vec4(vColor * intensity * 1.2, intensity);
}`;

// ── State ─────────────────────────────────────
const dpr = window.devicePixelRatio || 1;
const speedSteps = [1, 2, 4, 8, 16, 32];
let speedIdx = speedSteps.indexOf(CONFIG.speed);
if (speedIdx < 0) speedIdx = 2;
const MAX_TRAIL_DIST = 150;
const BIN_MAX_SEGS = 8000;
const TRAIL_MAX_SEGS = 50000;  // for all vessel trails combined
const MAX_HEADS = 10000;
const TRAIL_LENGTH = 25;       // keep last N positions per vessel

const state = {
  playing: false,
  currentBin: 0,
  accumBins: [],
  projected: [],
  speed: CONFIG.speed,
  vesselPos: [],    // per-panel: Map of mmsiIdx → {px, py, typeIdx}
  vesselTrail: [],  // per-panel: Map of mmsiIdx → [{px, py, typeIdx}, ...]
};

const panelCtx = [];

// ── Panel init ────────────────────────────────
function initPanel(idx) {
  const cfg = PANELS_CFG[idx];
  const mapContainer = document.getElementById('map-' + idx);
  const overlayDiv = document.getElementById('display-' + idx);
  const cell = overlayDiv.parentElement;

  const map = new maplibregl.Map({
    container: mapContainer,
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json',
    center: [cfg.centerLon, cfg.centerLat],
    zoom: cfg.zoom,
    interactive: false,
    attributionControl: false,
  });

  const rect = cell.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  const pw = Math.floor(w * dpr), ph = Math.floor(h * dpr);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ alpha: true, premultipliedAlpha: false });
  renderer.setSize(w, h);
  renderer.setPixelRatio(dpr);
  renderer.setClearColor(0x000000, 0);
  renderer.autoClear = false;
  overlayDiv.appendChild(renderer.domElement);

  // Camera (pixel-space orthographic, Y-down screen convention)
  const camera = new THREE.OrthographicCamera(0, w, h, 0, -1, 1);
  // Fullscreen quad camera for post-processing
  const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // HDR render targets
  const rtOpts = { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter,
                   magFilter: THREE.LinearFilter, format: THREE.RGBAFormat };
  let rtA = new THREE.WebGLRenderTarget(pw, ph, rtOpts);
  let rtB = new THREE.WebGLRenderTarget(pw, ph, rtOpts);
  const rtActive = new THREE.WebGLRenderTarget(pw, ph, rtOpts);

  // ── Line segment geometry (per-bin, reusable) ──
  const startArr = new Float32Array(BIN_MAX_SEGS * 4 * 2);
  const endArr = new Float32Array(BIN_MAX_SEGS * 4 * 2);
  const qvArr = new Float32Array(BIN_MAX_SEGS * 4);
  const colArr = new Float32Array(BIN_MAX_SEGS * 4 * 3);
  const alpArr = new Float32Array(BIN_MAX_SEGS * 4);
  const posArr = new Float32Array(BIN_MAX_SEGS * 4 * 3); // dummy position

  const lineGeom = new THREE.BufferGeometry();
  const mkAttr = (arr, sz) => { const a = new THREE.BufferAttribute(arr, sz);
    a.setUsage(THREE.DynamicDrawUsage); return a; };
  lineGeom.setAttribute('position', mkAttr(posArr, 3));
  lineGeom.setAttribute('aStart', mkAttr(startArr, 2));
  lineGeom.setAttribute('aEnd', mkAttr(endArr, 2));
  lineGeom.setAttribute('aQuadVertex', mkAttr(qvArr, 1));
  lineGeom.setAttribute('aColor', mkAttr(colArr, 3));
  lineGeom.setAttribute('aAlpha', mkAttr(alpArr, 1));

  // Index buffer (static pattern)
  const idxArr = new Uint32Array(BIN_MAX_SEGS * 6);
  for (let i = 0; i < BIN_MAX_SEGS; i++) {
    const b = i * 4, o = i * 6;
    idxArr[o]=b; idxArr[o+1]=b+1; idxArr[o+2]=b+2;
    idxArr[o+3]=b; idxArr[o+4]=b+2; idxArr[o+5]=b+3;
  }
  lineGeom.setIndex(new THREE.BufferAttribute(idxArr, 1));
  lineGeom.setDrawRange(0, 0);

  // Accumulation line material — wide soft glow for river-like corridor bands
  const lineMat = new THREE.RawShaderMaterial({
    vertexShader: LINE_VERT, fragmentShader: LINE_FRAG_ACCUM,
    uniforms: { uResolution: { value: new THREE.Vector2(w, h) },
                uLineWidth: { value: Math.max(10.0, cfg.dotRadius * 12.0) } },
    blending: THREE.AdditiveBlending, transparent: true,
    depthTest: false, depthWrite: false,
    glslVersion: THREE.GLSL1,
  });

  const lineMesh = new THREE.Mesh(lineGeom, lineMat);
  const lineScene = new THREE.Scene();
  lineScene.add(lineMesh);

  // ── Trail line geometry (per-vessel recent polylines, brighter) ──
  const tStartArr = new Float32Array(TRAIL_MAX_SEGS * 4 * 2);
  const tEndArr = new Float32Array(TRAIL_MAX_SEGS * 4 * 2);
  const tQvArr = new Float32Array(TRAIL_MAX_SEGS * 4);
  const tColArr = new Float32Array(TRAIL_MAX_SEGS * 4 * 3);
  const tAlpArr = new Float32Array(TRAIL_MAX_SEGS * 4);
  const tPosArr = new Float32Array(TRAIL_MAX_SEGS * 4 * 3);

  const trailGeom = new THREE.BufferGeometry();
  trailGeom.setAttribute('position', mkAttr(tPosArr, 3));
  trailGeom.setAttribute('aStart', mkAttr(tStartArr, 2));
  trailGeom.setAttribute('aEnd', mkAttr(tEndArr, 2));
  trailGeom.setAttribute('aQuadVertex', mkAttr(tQvArr, 1));
  trailGeom.setAttribute('aColor', mkAttr(tColArr, 3));
  trailGeom.setAttribute('aAlpha', mkAttr(tAlpArr, 1));
  const tIdxArr = new Uint32Array(TRAIL_MAX_SEGS * 6);
  for (let i = 0; i < TRAIL_MAX_SEGS; i++) {
    const b = i * 4, o = i * 6;
    tIdxArr[o]=b; tIdxArr[o+1]=b+1; tIdxArr[o+2]=b+2;
    tIdxArr[o+3]=b; tIdxArr[o+4]=b+2; tIdxArr[o+5]=b+3;
  }
  trailGeom.setIndex(new THREE.BufferAttribute(tIdxArr, 1));
  trailGeom.setDrawRange(0, 0);

  // Trail material (brighter glow for active vessel movement)
  const trailMat = new THREE.RawShaderMaterial({
    vertexShader: LINE_VERT, fragmentShader: LINE_FRAG_TRAIL,
    uniforms: { uResolution: { value: new THREE.Vector2(w, h) },
                uLineWidth: { value: Math.max(6.0, cfg.dotRadius * 7.0) } },
    blending: THREE.AdditiveBlending, transparent: true,
    depthTest: false, depthWrite: false,
    glslVersion: THREE.GLSL1,
  });

  const trailMesh = new THREE.Mesh(trailGeom, trailMat);
  const trailScene = new THREE.Scene();
  trailScene.add(trailMesh);

  // ── Fade fullscreen quad ──
  const fadeGeom = new THREE.PlaneGeometry(2, 2);
  const fadeMat = new THREE.ShaderMaterial({
    vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
    fragmentShader: FADE_FRAG,
    uniforms: { tDiffuse: { value: null }, uFadeFactor: { value: cfg.fadeFactor } },
    depthTest: false, depthWrite: false,
  });
  const fadeQuad = new THREE.Mesh(fadeGeom, fadeMat);
  const fadeScene = new THREE.Scene();
  fadeScene.add(fadeQuad);

  // ── Vessel head points ──
  const headPosArr = new Float32Array(MAX_HEADS * 3);
  const headColArr = new Float32Array(MAX_HEADS * 3);
  const headGeom = new THREE.BufferGeometry();
  const hpAttr = new THREE.BufferAttribute(headPosArr, 3);
  hpAttr.setUsage(THREE.DynamicDrawUsage);
  headGeom.setAttribute('position', hpAttr);
  const hcAttr = new THREE.BufferAttribute(headColArr, 3);
  hcAttr.setUsage(THREE.DynamicDrawUsage);
  headGeom.setAttribute('aHeadColor', hcAttr);
  headGeom.setDrawRange(0, 0);

  const headMat = new THREE.RawShaderMaterial({
    vertexShader: HEAD_VERT, fragmentShader: HEAD_FRAG,
    uniforms: {
      uPointSize: { value: Math.max(8.0, cfg.dotRadius * 8.0) * dpr },
      uResolution: { value: new THREE.Vector2(w, h) },
    },
    blending: THREE.AdditiveBlending, transparent: true,
    depthTest: false, depthWrite: false,
    glslVersion: THREE.GLSL1,
  });
  const headPoints = new THREE.Points(headGeom, headMat);
  const headScene = new THREE.Scene();
  headScene.add(headPoints);

  // ── Phosphor accumulation points ──
  // Per-bin additive splats drawn straight into the accumulation RT.
  // No per-vessel polylines, no head dots — the corridor *is* the
  // density of overlapping fading points (Kpler-reference style).
  const PHOSPHOR_MAX = BIN_MAX_SEGS;  // points per bin; same budget as segments
  const accPointPosArr = new Float32Array(PHOSPHOR_MAX * 3);
  const accPointColArr = new Float32Array(PHOSPHOR_MAX * 3);
  const accPointGeom = new THREE.BufferGeometry();
  const appAttr = new THREE.BufferAttribute(accPointPosArr, 3);
  appAttr.setUsage(THREE.DynamicDrawUsage);
  accPointGeom.setAttribute('position', appAttr);
  const apcAttr = new THREE.BufferAttribute(accPointColArr, 3);
  apcAttr.setUsage(THREE.DynamicDrawUsage);
  accPointGeom.setAttribute('aHeadColor', apcAttr);
  accPointGeom.setDrawRange(0, 0);

  // Smaller, dimmer phosphor splats — let density build by overlap,
  // not by single-dot brightness.
  const accPointMat = new THREE.RawShaderMaterial({
    vertexShader: HEAD_VERT, fragmentShader: HEAD_FRAG,
    uniforms: {
      uPointSize: { value: Math.max(3.0, cfg.dotRadius * 3.0) * dpr },
      uResolution: { value: new THREE.Vector2(w, h) },
    },
    blending: THREE.AdditiveBlending, transparent: true,
    depthTest: false, depthWrite: false,
    glslVersion: THREE.GLSL1,
  });
  const accPoints = new THREE.Points(accPointGeom, accPointMat);
  const accPointScene = new THREE.Scene();
  accPointScene.add(accPoints);

  // ── EffectComposer ──
  // First pass: composite accumulation + active textures
  const compShader = {
    uniforms: {
      tDiffuse: { value: null },  // required by ShaderPass convention (unused)
      tAccum: { value: null },
      tActive: { value: null },
    },
    vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader: COMPOSITE_FRAG,
  };
  const composer = new EffectComposer(renderer);
  const compPass = new ShaderPass(compShader);
  composer.addPass(compPass);
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(pw, ph),
    cfg.bloom ? 1.0 : 0.0,
    0.5,
    0.15
  );
  composer.addPass(bloomPass);

  let binSegCount = 0;

  const panel = {
    map, renderer, camera, quadCamera,
    rtA, rtB, rtActive,
    lineScene, lineGeom, lineMat, startArr, endArr, qvArr, colArr, alpArr,
    trailScene, trailGeom, trailMat, tStartArr, tEndArr, tQvArr, tColArr, tAlpArr,
    fadeScene, fadeMat,
    headScene, headGeom, headPosArr, headColArr,
    accPointScene, accPointGeom, accPointPosArr, accPointColArr,
    composer, compPass, bloomPass,
    w, h, pw, ph, cfg, binSegCount,
    _trailCount: 0,
    style: cfg.style || 'trails',
  };

  panelCtx.push(panel);
  state.accumBins.push(0);
  state.projected.push(false);
  state.vesselPos.push(new Map());
  state.vesselTrail.push(new Map());

  // Project coordinates once map loads
  // Point format: [lat, lon, typeIdx, mmsiIdx] → append [px, py] at indices 4,5
  map.on('load', function() {
    const bins = cfg.bins;
    for (let b = 0; b < bins.length; b++) {
      const bin = bins[b];
      for (let i = 0; i < bin.length; i++) {
        const pt = bin[i];
        const proj = map.project([pt[1], pt[0]]);
        pt.push(proj.x, proj.y);
      }
    }
    state.projected[idx] = true;
    checkAllProjected();
  });

  return panel;
}

function checkAllProjected() {
  for (let i = 0; i < N_PANELS; i++) {
    if (!state.projected[i]) return;
  }
  state.playing = true;
  document.getElementById('play-btn').classList.add('active');
  document.getElementById('play-btn').textContent = '\\u23F8';
  requestAnimationFrame(animate);
}

// ── Line segment buffer ───────────────────────
function addSegment(p, sx, sy, ex, ey, r, g, b, alpha) {
  if (p.binSegCount >= BIN_MAX_SEGS) return;
  const i = p.binSegCount;
  const b4 = i * 4;
  for (let v = 0; v < 4; v++) {
    const vi = b4 + v;
    p.startArr[vi * 2] = sx; p.startArr[vi * 2 + 1] = sy;
    p.endArr[vi * 2] = ex; p.endArr[vi * 2 + 1] = ey;
    p.qvArr[vi] = v;
    p.colArr[vi * 3] = r; p.colArr[vi * 3 + 1] = g; p.colArr[vi * 3 + 2] = b;
    p.alpArr[vi] = alpha;
  }
  p.binSegCount++;
}

function flushSegments(p) {
  if (p.binSegCount === 0) return;
  const g = p.lineGeom;
  g.attributes.aStart.needsUpdate = true;
  g.attributes.aEnd.needsUpdate = true;
  g.attributes.aQuadVertex.needsUpdate = true;
  g.attributes.aColor.needsUpdate = true;
  g.attributes.aAlpha.needsUpdate = true;
  g.setDrawRange(0, p.binSegCount * 6);
}

// ── Trail segment helper ─────────────────────
function addTrailSeg(p, sx, sy, ex, ey, r, g, b, alpha) {
  if (p._trailCount >= TRAIL_MAX_SEGS) return;
  const i = p._trailCount;
  const b4 = i * 4;
  for (let v = 0; v < 4; v++) {
    const vi = b4 + v;
    p.tStartArr[vi * 2] = sx; p.tStartArr[vi * 2 + 1] = sy;
    p.tEndArr[vi * 2] = ex; p.tEndArr[vi * 2 + 1] = ey;
    p.tQvArr[vi] = v;
    p.tColArr[vi * 3] = r; p.tColArr[vi * 3 + 1] = g; p.tColArr[vi * 3 + 2] = b;
    p.tAlpArr[vi] = alpha;
  }
  p._trailCount++;
}

// ── Build vessel heads + trail polylines on active RT ──
function renderActive(pIdx) {
  const p = panelCtx[pIdx];
  const vpos = state.vesselPos[pIdx];
  const vtrail = state.vesselTrail[pIdx];

  // Build trail line segments from per-vessel recent positions
  p._trailCount = 0;
  vtrail.forEach(function(trail, mmsiIdx) {
    if (trail.length < 2) return;
    const col = PALETTE[trail[0].typeIdx] || PALETTE[0];
    const r = col[0]/255, g = col[1]/255, b = col[2]/255;
    const len = trail.length;

    for (let i = 1; i < len; i++) {
      const prev = trail[i - 1], cur = trail[i];
      const dx = cur.px - prev.px, dy = cur.py - prev.py;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if (dist < 0.5 || dist > MAX_TRAIL_DIST) continue;

      // Alpha gradient: oldest segment = 0.15, newest = 0.9
      const frac = i / (len - 1);
      const alpha = 0.15 + frac * 0.75;
      addTrailSeg(p, prev.px, prev.py, cur.px, cur.py, r, g, b, alpha);
    }
  });

  // Flush trail geometry
  if (p._trailCount > 0) {
    const tg = p.trailGeom;
    tg.attributes.aStart.needsUpdate = true;
    tg.attributes.aEnd.needsUpdate = true;
    tg.attributes.aQuadVertex.needsUpdate = true;
    tg.attributes.aColor.needsUpdate = true;
    tg.attributes.aAlpha.needsUpdate = true;
    tg.setDrawRange(0, p._trailCount * 6);
  } else {
    p.trailGeom.setDrawRange(0, 0);
  }

  // Build head points
  let headCount = 0;
  vpos.forEach(function(v) {
    if (headCount >= MAX_HEADS) return;
    p.headPosArr[headCount*3] = v.px;
    p.headPosArr[headCount*3+1] = v.py;
    p.headPosArr[headCount*3+2] = 0;
    const col = PALETTE[v.typeIdx] || PALETTE[0];
    p.headColArr[headCount*3] = col[0]/255;
    p.headColArr[headCount*3+1] = col[1]/255;
    p.headColArr[headCount*3+2] = col[2]/255;
    headCount++;
  });
  p.headGeom.setDrawRange(0, headCount);
  p.headGeom.attributes.position.needsUpdate = true;
  p.headGeom.attributes.aHeadColor.needsUpdate = true;

  // Render trails + heads to active RT
  p.renderer.setRenderTarget(p.rtActive);
  p.renderer.clear();
  if (p._trailCount > 0) {
    p.renderer.render(p.trailScene, p.camera);
  }
  p.renderer.render(p.headScene, p.camera);
}

// ── Per-frame render ──────────────────────────
function renderFrame(pIdx, binIdx) {
  const p = panelCtx[pIdx];
  const bin = p.cfg.bins[binIdx];
  const phosphor = p.style === 'phosphor';

  // Step 1: Fade accumulation rtA → rtB
  p.fadeMat.uniforms.tDiffuse.value = p.rtA.texture;
  p.fadeMat.uniforms.uFadeFactor.value = p.cfg.fadeFactor;
  p.renderer.setRenderTarget(p.rtB);
  p.renderer.clear();
  p.renderer.render(p.fadeScene, p.quadCamera);

  // Step 2: Process new bin
  if (bin && bin.length > 0) {
    if (phosphor) {
      // Phosphor mode: splat one additive point per ping into rtB.
      // No vessel-position tracking, no trails — density emerges from
      // overlap of fading points alone.
      const n = Math.min(bin.length, p.accPointPosArr.length / 3);
      for (let i = 0; i < n; i++) {
        const pt = bin[i];
        const col = PALETTE[pt[2] || 0] || PALETTE[0];
        p.accPointPosArr[i*3]     = pt[4];
        p.accPointPosArr[i*3 + 1] = pt[5];
        p.accPointPosArr[i*3 + 2] = 0;
        p.accPointColArr[i*3]     = col[0]/255;
        p.accPointColArr[i*3 + 1] = col[1]/255;
        p.accPointColArr[i*3 + 2] = col[2]/255;
      }
      p.accPointGeom.setDrawRange(0, n);
      p.accPointGeom.attributes.position.needsUpdate = true;
      p.accPointGeom.attributes.aHeadColor.needsUpdate = true;
      p.renderer.setRenderTarget(p.rtB);
      p.renderer.render(p.accPointScene, p.camera);
    } else {
      // Trails mode: per-vessel polylines + head dots (analytical).
      p.binSegCount = 0;
      const vpos = state.vesselPos[pIdx];
      const vtrail = state.vesselTrail[pIdx];
      const alpha = p.cfg.dotAlpha;

      for (let i = 0; i < bin.length; i++) {
        const pt = bin[i];
        const px = pt[4], py = pt[5];
        const typeIdx = pt[2] || 0;
        const mmsiIdx = pt[3] || 0;
        const col = PALETTE[typeIdx] || PALETTE[0];
        const r = col[0]/255, g = col[1]/255, b = col[2]/255;

        const prev = vpos.get(mmsiIdx);
        if (prev) {
          const dx = px - prev.px, dy = py - prev.py;
          const dist = Math.sqrt(dx*dx + dy*dy);
          if (dist > 0.5 && dist < MAX_TRAIL_DIST) {
            addSegment(p, prev.px, prev.py, px, py, r, g, b, alpha);
          }
        }

        vpos.set(mmsiIdx, { px, py, typeIdx });

        let trail = vtrail.get(mmsiIdx);
        if (!trail) { trail = []; vtrail.set(mmsiIdx, trail); }
        trail.push({ px, py, typeIdx });
        if (trail.length > TRAIL_LENGTH) trail.shift();
      }

      flushSegments(p);
      p.renderer.setRenderTarget(p.rtB);
      p.renderer.render(p.lineScene, p.camera);
    }
  }

  // Step 3: Swap render targets
  const tmp = p.rtA; p.rtA = p.rtB; p.rtB = tmp;

  // Step 4: Render active layer (skipped in phosphor mode — corridor
  // is the subject, no separate "now" overlay).
  if (phosphor) {
    p.renderer.setRenderTarget(p.rtActive);
    p.renderer.clear();
  } else {
    renderActive(pIdx);
  }

  // Step 5: Composite + bloom → screen
  p.compPass.uniforms.tAccum.value = p.rtA.texture;
  p.compPass.uniforms.tActive.value = p.rtActive.texture;
  p.renderer.setRenderTarget(null);
  p.renderer.clear();
  p.composer.render();
}

// Fade-only frame (no new bin, still render active trails)
function renderIdle(pIdx) {
  const p = panelCtx[pIdx];
  p.fadeMat.uniforms.tDiffuse.value = p.rtA.texture;
  p.renderer.setRenderTarget(p.rtB);
  p.renderer.clear();
  p.renderer.render(p.fadeScene, p.quadCamera);
  const tmp = p.rtA; p.rtA = p.rtB; p.rtB = tmp;

  if (p.style === 'phosphor') {
    p.renderer.setRenderTarget(p.rtActive);
    p.renderer.clear();
  } else {
    renderActive(pIdx);
  }

  p.compPass.uniforms.tAccum.value = p.rtA.texture;
  p.compPass.uniforms.tActive.value = p.rtActive.texture;
  p.renderer.setRenderTarget(null);
  p.renderer.clear();
  p.composer.render();
}

// Scrub/seek — replay all bins to rebuild state
function renderUpTo(pIdx, targetBin) {
  const p = panelCtx[pIdx];
  p.renderer.setRenderTarget(p.rtA); p.renderer.clear();
  p.renderer.setRenderTarget(p.rtB); p.renderer.clear();
  state.vesselPos[pIdx] = new Map();
  state.vesselTrail[pIdx] = new Map();
  for (let i = 0; i <= targetBin; i++) {
    renderFrame(pIdx, i);
  }
  state.accumBins[pIdx] = targetBin + 1;
}

function maxBins() {
  let m = 0;
  for (let i = 0; i < N_PANELS; i++) m = Math.max(m, PANELS_CFG[i].bins.length);
  return m;
}

function formatTimestamp(ms) {
  if (!ms) return '—';
  const d = new Date(ms);
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return months[d.getUTCMonth()] + ' ' + String(d.getUTCDate()).padStart(2,'0') + ' ' +
         String(d.getUTCHours()).padStart(2,'0') + ':' + String(d.getUTCMinutes()).padStart(2,'0');
}

// ── Animation loop ────────────────────────────
let lastFrame = performance.now();
const MB = maxBins();
const tsEl = document.getElementById('timestamp');
const vcEl = document.getElementById('vessel-count');
const sliderEl = document.getElementById('slider');

// When showClock is false, render the date range as a static caption
// once and never touch the DOM again. This re-frames the clip from
// "live feed" to "composed piece" — and is also a tiny perf win.
if (tsEl && !CONFIG.showClock) {
  tsEl.textContent = DATE_RANGE_LABEL || '\\u2014';
}

function animate(now) {
  if (state.playing) {
    const dt = (now - lastFrame) / 1000;
    state.currentBin += dt * state.speed;

    if (state.currentBin >= MB) {
      state.currentBin = 0;
      for (let i = 0; i < N_PANELS; i++) {
        state.accumBins[i] = 0;
        const p = panelCtx[i];
        p.renderer.setRenderTarget(p.rtA); p.renderer.clear();
        p.renderer.setRenderTarget(p.rtB); p.renderer.clear();
        state.vesselPos[i] = new Map();
        state.vesselTrail[i] = new Map();
      }
    }

    const targetBin = Math.min(Math.floor(state.currentBin), MB - 1);
    let rendered = false;

    for (let i = 0; i < N_PANELS; i++) {
      const nBins = PANELS_CFG[i].bins.length;
      const panelTarget = Math.min(targetBin, nBins - 1);

      if (state.accumBins[i] <= panelTarget) {
        while (state.accumBins[i] <= panelTarget) {
          renderFrame(i, state.accumBins[i]);
          state.accumBins[i]++;
        }
        rendered = true;
      } else {
        renderIdle(i);
      }
    }

    sliderEl.value = Math.floor((targetBin / Math.max(MB - 1, 1)) * 1000);
    if (CONFIG.showClock) {
      const ts0 = PANELS_CFG[0].binTimestamps;
      const tIdx = Math.min(targetBin, ts0.length - 1);
      if (tIdx >= 0) tsEl.textContent = formatTimestamp(ts0[tIdx]);
    }
    if (vcEl && N_PANELS === 1) {
      const cv = PANELS_CFG[0].cumulVessels;
      const vIdx = Math.min(targetBin, cv.length - 1);
      if (vIdx >= 0) vcEl.textContent = cv[vIdx].toLocaleString();
    }
  }

  lastFrame = now;
  requestAnimationFrame(animate);
}

// ── Controls ──────────────────────────────────
document.getElementById('play-btn').addEventListener('click', function() {
  state.playing = !state.playing;
  this.textContent = state.playing ? '\\u23F8' : '\\u25B6';
  this.classList.toggle('active', state.playing);
});

sliderEl.addEventListener('input', function(e) {
  const frac = parseInt(e.target.value) / 1000;
  const target = Math.floor(frac * (MB - 1));
  state.currentBin = target;
  for (let i = 0; i < N_PANELS; i++) {
    renderUpTo(i, Math.min(target, PANELS_CFG[i].bins.length - 1));
  }
  const ts0 = PANELS_CFG[0].binTimestamps;
  const tIdx = Math.min(target, ts0.length - 1);
  if (tIdx >= 0) tsEl.textContent = formatTimestamp(ts0[tIdx]);
  if (vcEl && N_PANELS === 1) {
    const cv = PANELS_CFG[0].cumulVessels;
    const vIdx = Math.min(target, cv.length - 1);
    if (vIdx >= 0) vcEl.textContent = cv[vIdx].toLocaleString();
  }
});

document.getElementById('speed-btn').addEventListener('click', function() {
  speedIdx = (speedIdx + 1) % speedSteps.length;
  state.speed = speedSteps[speedIdx];
  this.textContent = state.speed + 'x';
  document.getElementById('speed-label').textContent = state.speed + 'x';
});

document.addEventListener('keydown', function(e) {
  if (e.code === 'Space') {
    e.preventDefault();
    document.getElementById('play-btn').click();
  } else if (e.code === 'ArrowRight') {
    state.currentBin = Math.min(state.currentBin + state.speed, MB - 1);
  } else if (e.code === 'ArrowLeft') {
    const target = Math.max(Math.floor(state.currentBin - state.speed), 0);
    state.currentBin = target;
    for (let i = 0; i < N_PANELS; i++) {
      renderUpTo(i, Math.min(target, PANELS_CFG[i].bins.length - 1));
    }
  }
});

// ── Init ──────────────────────────────────────
for (let i = 0; i < N_PANELS; i++) {
  initPanel(i);
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Template assembly
# ---------------------------------------------------------------------------


def _build_legend_html(palette: list[list[int]], type_names: list[str],
                       color_by_type: bool) -> str:
    """Build legend HTML from palette and type names."""
    if not color_by_type:
        return ""
    items = []
    for i, name in enumerate(type_names):
        if i >= len(palette):
            break
        c = palette[i]
        items.append(
            f'<div class="legend-item">'
            f'<span class="legend-dot" style="background:rgb({c[0]},{c[1]},{c[2]})"></span>'
            f'{name}</div>'
        )
    return "\n  ".join(items)


def _speed_label(speed: float) -> str:
    return f"{speed:.0f}x" if speed == int(speed) else f"{speed}x"


def render_timelapse(data: dict) -> str:
    """Assemble the timelapse HTML from template sections.

    Args:
        data: Dict produced by ``generate_timelapse()`` with all
            template variables.

    Returns:
        Complete HTML string.
    """
    import json

    palette = json.loads(data["palette_json"])
    type_names = json.loads(data["type_names_json"])
    legend_html = _build_legend_html(
        palette, type_names, data["color_by_type"],
    )
    speed_label = _speed_label(data["speed"])

    js_bool = lambda v: "true" if v else "false"  # noqa: E731

    if data.get("multi"):
        # Multi-panel mode.
        n = data["n_panels"]
        layout = data.get("layout", "vertical")

        # Grid CSS.
        if layout == "horizontal":
            grid_style = f"grid-template-columns: repeat({n}, 1fr);"
        elif layout == "grid":
            cols = 2 if n <= 4 else 3
            grid_style = (
                f"grid-template-columns: repeat({cols}, 1fr); "
                f"grid-auto-rows: 1fr;"
            )
        else:  # vertical
            grid_style = f"grid-template-rows: repeat({n}, 1fr);"

        # Panel cells HTML.
        panels_raw = json.loads(data["panels_json"])
        cells = []
        for i, p in enumerate(panels_raw):
            label = p.get("label", "")
            cells.append(
                f'<div class="panel-cell" id="panel-cell-{i}">\n'
                f'    <div class="panel-map" id="map-{i}"></div>\n'
                f'    <div class="panel-darken"></div>\n'
                f'    <div class="panel-overlay" id="display-{i}"></div>\n'
                f'    <div class="panel-label"><h3>{label}</h3></div>\n'
                f'  </div>'
            )

        html_section = _MULTI_PANEL_HTML.substitute(
            title=data["title"],
            subtitle=data.get("subtitle", ""),
            grid_style=grid_style,
            panel_cells="\n  ".join(cells),
            legend_html=legend_html,
            speed_label=speed_label,
        )

        js_data = _JS_MULTI_DATA.substitute(
            palette_json=data["palette_json"],
            type_names_json=data["type_names_json"],
            color_by_type=js_bool(data["color_by_type"]),
            color_mode=data.get("color_mode", "type"),
            dot_radius=data["dot_radius"],
            dot_alpha=data["dot_alpha"],
            fade_factor=data["fade_factor"],
            bloom=js_bool(data["bloom"]),
            speed=data["speed"],
            style=data.get("style", "trails"),
            show_clock=js_bool(data.get("show_clock", True)),
            date_range_label=_js_string_escape(data.get("date_range_label", "")),
            n_panels=n,
            panels_json=data["panels_json"],
        )

    else:
        # Single-panel mode.
        html_section = _SINGLE_PANEL_HTML.substitute(
            title=data["title"],
            subtitle=data.get("subtitle", ""),
            legend_html=legend_html,
            speed_label=speed_label,
        )

        js_data = _JS_SINGLE_DATA.substitute(
            bins_json=data["bins_json"],
            cumul_vessels_json=data["cumul_vessels_json"],
            bin_timestamps_ms_json=data["bin_timestamps_ms_json"],
            palette_json=data["palette_json"],
            type_names_json=data["type_names_json"],
            color_by_type=js_bool(data["color_by_type"]),
            color_mode=data.get("color_mode", "type"),
            dot_radius=data["dot_radius"],
            dot_alpha=data["dot_alpha"],
            fade_factor=data["fade_factor"],
            bloom=js_bool(data["bloom"]),
            speed=data["speed"],
            style=data.get("style", "trails"),
            show_clock=js_bool(data.get("show_clock", True)),
            date_range_label=_js_string_escape(data.get("date_range_label", "")),
            center_lat=data["center_lat"],
            center_lon=data["center_lon"],
            zoom=data["zoom"],
        )

    return html_section + "\n" + js_data + "\n" + _JS_ENGINE


def _js_string_escape(s: str) -> str:
    """Escape a string for safe interpolation inside a JS double-quoted literal."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
        .replace("<", "\\u003c")
    )
