"""Timelapse HTML template — cinematic vessel corridor visualization.

Produces a standalone HTML file with Canvas 2D additive-blending
accumulation over a MapLibre GL dark basemap. Positions build up
over time to reveal shipping corridors — like long-exposure
photography of maritime traffic.

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

_SINGLE_PANEL_HTML = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — $title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
""" + _CSS + """
</style>
</head>
<body>
<div id="container">
  <div class="panel-cell" id="panel-0">
    <div class="panel-map" id="map-0"></div>
    <div class="panel-darken"></div>
    <canvas class="panel-overlay" id="display-0"></canvas>
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

<script>
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

<script>
""")

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS_SINGLE_DATA = Template("""\
// ── Data ──────────────────────────────────────
const BINS = $bins_json;
const CUMUL_VESSELS = $cumul_vessels_json;
const BIN_TIMESTAMPS = $bin_timestamps_ms_json;
const PALETTE = $palette_json;
const TYPE_NAMES = $type_names_json;
const COLOR_BY_TYPE = $color_by_type;
const CONFIG = {
  dotRadius: $dot_radius,
  dotAlpha: $dot_alpha,
  fadeFactor: $fade_factor,
  bloom: $bloom,
  speed: $speed,
};
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
}];
""")

_JS_MULTI_DATA = Template("""\
// ── Data ──────────────────────────────────────
const PALETTE = $palette_json;
const TYPE_NAMES = $type_names_json;
const COLOR_BY_TYPE = $color_by_type;
const CONFIG = {
  dotRadius: $dot_radius,
  dotAlpha: $dot_alpha,
  fadeFactor: $fade_factor,
  bloom: $bloom,
  speed: $speed,
};
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
  };
});
""")

_JS_ENGINE = """\
// ── Canvas engine ─────────────────────────────
// Two-layer rendering:
//   accumCanvas — persistent corridor traces (line segments, slow fade)
//   activeCanvas — bright moving vessel heads + short trails (cleared each frame)
const dpr = window.devicePixelRatio || 1;
const speedSteps = [1, 2, 4, 8, 16, 32];
let speedIdx = speedSteps.indexOf(CONFIG.speed);
if (speedIdx < 0) speedIdx = 2;
const MAX_TRAIL_DIST = 150; // max pixel distance for line continuity

const state = {
  playing: false,
  currentBin: 0,
  accumBins: [],
  projected: [],
  speed: CONFIG.speed,
  vesselPos: [],   // per-panel: Map of mmsiIdx → {px, py, typeIdx}
};

const panelCtx = [];

function initPanel(idx) {
  const cfg = PANELS_CFG[idx];
  const mapContainer = document.getElementById('map-' + idx);
  const displayCanvas = document.getElementById('display-' + idx);
  const cell = displayCanvas.parentElement;

  const map = new maplibregl.Map({
    container: mapContainer,
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json',
    center: [cfg.centerLon, cfg.centerLat],
    zoom: cfg.zoom,
    interactive: false,
    attributionControl: false,
  });

  const rect = cell.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;

  function sizeCanvas(canvas) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return ctx;
  }

  const displayCtx = sizeCanvas(displayCanvas);
  const accumCanvas = document.createElement('canvas');
  const accumCtx = sizeCanvas(accumCanvas);
  const activeCanvas = document.createElement('canvas');
  const activeCtx = sizeCanvas(activeCanvas);
  const bloomCanvas = document.createElement('canvas');
  const bloomCtx = sizeCanvas(bloomCanvas);

  // Dot stamps for accumulation layer (dim, for corridor traces).
  const dotStamps = [];
  const dotSize = Math.ceil(cfg.dotRadius * 4);
  for (let t = 0; t < PALETTE.length; t++) {
    const c = document.createElement('canvas');
    c.width = dotSize * dpr; c.height = dotSize * dpr;
    const dc = c.getContext('2d'); dc.scale(dpr, dpr);
    const cx = dotSize / 2, cy = dotSize / 2;
    const grad = dc.createRadialGradient(cx, cy, 0, cx, cy, dotSize / 2);
    const col = PALETTE[t]; const a = cfg.dotAlpha;
    grad.addColorStop(0, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + a + ')');
    grad.addColorStop(0.3, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (a * 0.4) + ')');
    grad.addColorStop(1, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0)');
    dc.fillStyle = grad; dc.fillRect(0, 0, dotSize, dotSize);
    dotStamps.push(c);
  }

  // Bright dot stamps for active layer (vessel heads).
  const brightDotStamps = [];
  const brightSize = Math.ceil(cfg.dotRadius * 6);
  for (let t = 0; t < PALETTE.length; t++) {
    const c = document.createElement('canvas');
    c.width = brightSize * dpr; c.height = brightSize * dpr;
    const dc = c.getContext('2d'); dc.scale(dpr, dpr);
    const cx = brightSize / 2, cy = brightSize / 2;
    const grad = dc.createRadialGradient(cx, cy, 0, cx, cy, brightSize / 2);
    const col = PALETTE[t];
    grad.addColorStop(0, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0.9)');
    grad.addColorStop(0.2, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0.5)');
    grad.addColorStop(0.5, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0.15)');
    grad.addColorStop(1, 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',0)');
    dc.fillStyle = grad; dc.fillRect(0, 0, brightSize, brightSize);
    brightDotStamps.push(c);
  }

  const panel = {
    map, displayCanvas, displayCtx,
    accumCanvas, accumCtx,
    activeCanvas, activeCtx,
    bloomCanvas, bloomCtx,
    dotStamps, brightDotStamps,
    w, h, dotSize, brightSize, cfg,
  };

  panelCtx.push(panel);
  state.accumBins.push(0);
  state.projected.push(false);
  state.vesselPos.push(new Map());

  // Project coordinates once map loads.
  // Point format: [lat, lon, typeIdx, mmsiIdx, px, py]
  map.on('load', function() {
    const bins = cfg.bins;
    for (let b = 0; b < bins.length; b++) {
      const bin = bins[b];
      for (let p = 0; p < bin.length; p++) {
        const pt = bin[p];
        const proj = map.project([pt[1], pt[0]]);
        pt.push(proj.x, proj.y);  // indices 4, 5
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

// Render a bin: draw line segments + dots on accumulation canvas,
// update per-vessel positions for active layer.
function renderBin(pIdx, binIdx) {
  const p = panelCtx[pIdx];
  const bin = p.cfg.bins[binIdx];
  if (!bin) return;

  const vpos = state.vesselPos[pIdx];
  const half = p.dotSize / 2;

  p.accumCtx.globalCompositeOperation = 'lighter';

  for (let i = 0; i < bin.length; i++) {
    const pt = bin[i];
    const px = pt[4], py = pt[5];
    const typeIdx = pt[2] || 0;
    const mmsiIdx = pt[3] || 0;
    const col = PALETTE[typeIdx] || PALETTE[0];
    const a = p.cfg.dotAlpha;

    // Draw line segment from previous position (corridor trace).
    const prev = vpos.get(mmsiIdx);
    if (prev) {
      const dx = px - prev.px, dy = py - prev.py;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 0.5 && dist < MAX_TRAIL_DIST) {
        p.accumCtx.strokeStyle = 'rgba(' + col[0] + ',' + col[1] + ',' + col[2] + ',' + (a * 0.7) + ')';
        p.accumCtx.lineWidth = Math.max(0.8, p.cfg.dotRadius * 0.8);
        p.accumCtx.beginPath();
        p.accumCtx.moveTo(prev.px, prev.py);
        p.accumCtx.lineTo(px, py);
        p.accumCtx.stroke();
      }
    }

    // Draw dot at position.
    p.accumCtx.drawImage(p.dotStamps[typeIdx], px - half, py - half, p.dotSize, p.dotSize);

    // Update vessel position.
    vpos.set(mmsiIdx, {px, py, typeIdx});
  }
}

// Render bright active vessel heads on the active canvas.
function renderActive(pIdx) {
  const p = panelCtx[pIdx];
  p.activeCtx.clearRect(0, 0, p.w, p.h);
  p.activeCtx.globalCompositeOperation = 'lighter';

  const vpos = state.vesselPos[pIdx];
  const half = p.brightSize / 2;

  vpos.forEach(function(v) {
    p.activeCtx.drawImage(
      p.brightDotStamps[v.typeIdx],
      v.px - half, v.py - half,
      p.brightSize, p.brightSize
    );
  });
}

// Apply fade to accumulated canvas.
function applyFade(pIdx) {
  const p = panelCtx[pIdx];
  const ff = p.cfg.fadeFactor;
  if (ff >= 1.0) return;
  p.accumCtx.globalCompositeOperation = 'destination-out';
  p.accumCtx.fillStyle = 'rgba(0,0,0,' + (1 - ff) + ')';
  p.accumCtx.fillRect(0, 0, p.w, p.h);
}

// Composite all layers to display.
function compositeDisplay(pIdx) {
  const p = panelCtx[pIdx];
  p.displayCtx.clearRect(0, 0, p.w, p.h);

  if (p.cfg.bloom) {
    p.bloomCtx.clearRect(0, 0, p.w, p.h);
    p.bloomCtx.filter = 'blur(4px)';
    p.bloomCtx.globalCompositeOperation = 'source-over';
    p.bloomCtx.drawImage(p.accumCanvas, 0, 0, p.w * dpr, p.h * dpr, 0, 0, p.w, p.h);
    p.bloomCtx.filter = 'none';

    p.displayCtx.globalCompositeOperation = 'lighter';
    p.displayCtx.globalAlpha = 0.3;
    p.displayCtx.drawImage(p.bloomCanvas, 0, 0, p.w * dpr, p.h * dpr, 0, 0, p.w, p.h);
    p.displayCtx.globalAlpha = 1.0;
  }

  // Corridor traces.
  p.displayCtx.globalCompositeOperation = 'lighter';
  p.displayCtx.drawImage(p.accumCanvas, 0, 0, p.w * dpr, p.h * dpr, 0, 0, p.w, p.h);

  // Bright vessel heads on top.
  p.displayCtx.drawImage(p.activeCanvas, 0, 0, p.w * dpr, p.h * dpr, 0, 0, p.w, p.h);
}

// Re-render all bins up to target (for scrub/seek).
function renderUpTo(pIdx, targetBin) {
  const p = panelCtx[pIdx];
  p.accumCtx.clearRect(0, 0, p.w, p.h);
  state.vesselPos[pIdx] = new Map();

  for (let i = 0; i <= targetBin; i++) {
    // Apply fade between bins during rebuild for consistent look.
    applyFade(pIdx);
    renderBin(pIdx, i);
  }
  state.accumBins[pIdx] = targetBin + 1;
  renderActive(pIdx);
  compositeDisplay(pIdx);
}

function maxBins() {
  let m = 0;
  for (let i = 0; i < N_PANELS; i++) {
    m = Math.max(m, PANELS_CFG[i].bins.length);
  }
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

function animate(now) {
  if (state.playing) {
    const dt = (now - lastFrame) / 1000;
    state.currentBin += dt * state.speed;

    if (state.currentBin >= MB) {
      state.currentBin = 0;
      for (let i = 0; i < N_PANELS; i++) {
        state.accumBins[i] = 0;
        panelCtx[i].accumCtx.clearRect(0, 0, panelCtx[i].w, panelCtx[i].h);
        state.vesselPos[i] = new Map();
      }
    }

    const targetBin = Math.min(Math.floor(state.currentBin), MB - 1);

    for (let i = 0; i < N_PANELS; i++) {
      const nBins = PANELS_CFG[i].bins.length;
      const panelTarget = Math.min(targetBin, nBins - 1);

      applyFade(i);

      while (state.accumBins[i] <= panelTarget) {
        renderBin(i, state.accumBins[i]);
        state.accumBins[i]++;
      }

      renderActive(i);
      compositeDisplay(i);
    }

    sliderEl.value = Math.floor((targetBin / Math.max(MB - 1, 1)) * 1000);

    const ts0 = PANELS_CFG[0].binTimestamps;
    const tIdx = Math.min(targetBin, ts0.length - 1);
    if (tIdx >= 0) tsEl.textContent = formatTimestamp(ts0[tIdx]);

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
                f'    <canvas class="panel-overlay" id="display-{i}"></canvas>\n'
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

        # For multi-panel, we need to serialize each panel's bins
        # individually and pass them via PANELS_RAW. The bins_json in
        # each panel dict is already a JSON string from _safe_json_embed.
        js_data = _JS_MULTI_DATA.substitute(
            palette_json=data["palette_json"],
            type_names_json=data["type_names_json"],
            color_by_type=js_bool(data["color_by_type"]),
            dot_radius=data["dot_radius"],
            dot_alpha=data["dot_alpha"],
            fade_factor=data["fade_factor"],
            bloom=js_bool(data["bloom"]),
            speed=data["speed"],
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
            dot_radius=data["dot_radius"],
            dot_alpha=data["dot_alpha"],
            fade_factor=data["fade_factor"],
            bloom=js_bool(data["bloom"]),
            speed=data["speed"],
            center_lat=data["center_lat"],
            center_lon=data["center_lon"],
            zoom=data["zoom"],
        )

    return html_section + "\n" + js_data + "\n" + _JS_ENGINE
