"""Timelapse HTML template — D3.js + Canvas 2D renderer.

A pure-D3/Canvas reimplementation of the cinematic vessel corridor
timelapse. Produces a self-contained HTML file with no external tile
requests: land geometry is embedded as TopoJSON, vessels are rendered
with additive Canvas 2D blending to produce neon-tube glow trails.

Parallel to :mod:`neptune_ais._timelapse_template` (the Three.js
renderer). Assembled from :class:`string.Template` sections and rendered
by :func:`viz.generate_timelapse_d3`. This module is internal
(``_`` prefix) — import only via ``viz.generate_timelapse_d3()``.
"""

from __future__ import annotations

import base64
import gzip
import json
import urllib.error
import urllib.request
from pathlib import Path
from string import Template

# ---------------------------------------------------------------------------
# TopoJSON acquisition (cached on disk, embedded into the HTML)
# ---------------------------------------------------------------------------

# Cache directory under ~/.neptune for the downloaded TopoJSON files.
# These are tiny (<200 KB each) so we just keep them alongside other caches.
_CACHE_DIR = Path.home() / ".neptune" / "viz_assets"

# Sources — small, well-maintained TopoJSON collections from d3 maintainers.
#   - world-atlas land-50m: global landmasses at 1:50M scale, ~100 KB.
#   - us-atlas states-10m: US states at 1:10M scale for port-level detail.
_TOPOJSON_SOURCES = {
    "land_50m": "https://cdn.jsdelivr.net/npm/world-atlas@2/land-50m.json",
    "us_states_10m": "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
}


def _fetch_topojson(name: str, url: str) -> bytes:
    """Fetch a TopoJSON file to disk cache and return its raw bytes.

    Files are persisted under ``~/.neptune/viz_assets/`` so subsequent
    template renders don't re-download.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{name}.json"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "neptune-ais/timelapse-d3"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as err:
        raise RuntimeError(
            f"Failed to download TopoJSON for {name} from {url}: {err}. "
            "Ensure network access during template render, or pre-populate "
            f"{cache_path} manually."
        ) from err

    cache_path.write_bytes(data)
    return data


def _embed_topojson() -> dict[str, str]:
    """Return a dict of TopoJSON payloads base64-gzip-encoded for inlining.

    Inlining keeps the output HTML self-contained (no CDN calls at view
    time). Gzip + base64 is roughly break-even on size for these small
    files but decompresses to strict UTF-8 JSON in the browser.
    """
    result: dict[str, str] = {}
    for name, url in _TOPOJSON_SOURCES.items():
        raw = _fetch_topojson(name, url)
        compressed = gzip.compress(raw, compresslevel=9)
        result[name] = base64.b64encode(compressed).decode("ascii")
    return result


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body {
  width: 100%; height: 100%;
  font-family: 'SF Mono', 'JetBrains Mono', 'Cascadia Code', 'Fira Code', ui-monospace, monospace;
  background: #04060d; color: #e8ecf4; overflow: hidden;
  -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision;
}

#container {
  position: absolute; inset: 0;
  display: grid;
  grid-template-rows: repeat(var(--n-panels, 3), 1fr);
}

.panel-cell {
  position: relative; width: 100%; height: 100%;
  overflow: hidden;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.panel-cell:last-child { border-bottom: none; }

canvas.basemap, canvas.trails {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
canvas.basemap { z-index: 1; }
canvas.trails  { z-index: 2; mix-blend-mode: screen; }

.panel-overlay {
  position: absolute; inset: 0;
  z-index: 3; pointer-events: none;
}

.panel-label {
  position: absolute; left: 24px; bottom: 40px;
  color: #f0f4ff;
}
.panel-label .region-index {
  font-size: 10px; letter-spacing: 3px; color: rgba(255,255,255,0.35);
  margin-bottom: 4px;
}
.panel-label h2 {
  font-size: 28px; font-weight: 300; letter-spacing: 0.5px;
  font-family: 'Iowan Old Style', Georgia, 'Times New Roman', serif;
  font-style: italic;
  color: #f8faff;
  text-shadow: 0 2px 12px rgba(0,0,0,0.9);
}
.panel-label .sea {
  font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase;
  color: #ff3a82; margin-top: 6px; font-weight: 600;
}
.panel-label .date {
  font-size: 9px; letter-spacing: 2px; text-transform: uppercase;
  color: rgba(255,255,255,0.35); margin-top: 3px;
  font-variant-numeric: tabular-nums;
}

.panel-count {
  position: absolute; left: 24px; top: 24px;
  color: rgba(255,255,255,0.55);
}
.panel-count .num {
  font-size: 22px; font-weight: 500;
  font-variant-numeric: tabular-nums;
  color: #9ed7ff; letter-spacing: 0.5px;
}
.panel-count .label {
  font-size: 8px; letter-spacing: 2px; text-transform: uppercase;
  color: rgba(255,255,255,0.3); margin-top: 2px;
}

.panel-timestamp {
  position: absolute; right: 24px; top: 24px;
  font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
  color: rgba(255,255,255,0.5);
  font-variant-numeric: tabular-nums;
}

/* ── Header ──────────────────────────────── */
#header {
  position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
  z-index: 10; text-align: center; pointer-events: none;
}
#header .eyebrow {
  font-size: 9px; letter-spacing: 4px; text-transform: uppercase;
  color: rgba(255,255,255,0.4);
}
#header h1 {
  font-size: 13px; font-weight: 500; letter-spacing: 2.5px;
  text-transform: uppercase; color: #f0f4ff; margin-top: 4px;
}

/* ── Legend ──────────────────────────────── */
#legend {
  position: absolute; left: 50%; transform: translateX(-50%);
  bottom: 56px; z-index: 10;
  display: flex; gap: 18px;
  background: rgba(6, 9, 20, 0.72);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 100px; padding: 7px 18px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.legend-item {
  display: flex; align-items: center; gap: 6px;
  font-size: 9px; letter-spacing: 1.5px;
  text-transform: uppercase; color: rgba(255,255,255,0.6);
}
.legend-dot {
  width: 7px; height: 7px; border-radius: 50%;
  box-shadow: 0 0 6px currentColor;
}

/* ── Controls ────────────────────────────── */
#controls {
  position: absolute; left: 50%; transform: translateX(-50%);
  bottom: 14px; z-index: 10;
  display: flex; align-items: center; gap: 12px;
  background: rgba(6, 9, 20, 0.78);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 100px; padding: 6px 14px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
#controls button {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  color: #d8e0f0;
  width: 28px; height: 24px; border-radius: 100px;
  cursor: pointer; font-size: 11px; font-family: inherit;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
}
#controls button:hover {
  background: rgba(255,255,255,0.12); color: #fff;
  border-color: rgba(255,255,255,0.22);
}
#controls button.active {
  background: rgba(0, 200, 255, 0.18);
  border-color: rgba(0, 200, 255, 0.42);
  color: #6ed6ff;
}
#slider {
  width: 260px; height: 3px; accent-color: #6ed6ff;
  background: transparent;
}
#speed-label {
  font-size: 10px; letter-spacing: 1.5px;
  color: rgba(255,255,255,0.45);
  font-variant-numeric: tabular-nums;
  min-width: 24px; text-align: center;
}

/* ── Corner marks (cinematic) ────────────── */
.panel-cell::before, .panel-cell::after {
  content: ''; position: absolute; z-index: 4;
  width: 14px; height: 14px;
  border: 1px solid rgba(255,255,255,0.18);
  pointer-events: none;
}
.panel-cell::before { top: 12px; right: 12px; border-left: 0; border-bottom: 0; }
.panel-cell::after  { bottom: 12px; left: 12px; border-right: 0; border-top: 0; }
"""


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

_HTML_SHELL = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — $title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>
<style>
$css
#container { --n-panels: $n_panels; }
</style>
</head>
<body>
<div id="header">
  <div class="eyebrow">$eyebrow</div>
  <h1>$title</h1>
</div>

<div id="container">
$panel_cells
</div>

<div id="legend">$legend_html</div>

<div id="controls">
  <button id="play-btn" class="active" title="Play / Pause (Space)">&#9654;</button>
  <input type="range" id="slider" min="0" max="1000" value="0" aria-label="Timeline">
  <button id="speed-btn" title="Playback speed">$speed_label</button>
  <span id="speed-label">$speed_label</span>
</div>

<script>
$js
</script>
</body>
</html>
""")


_PANEL_CELL = Template(r"""  <div class="panel-cell" data-panel="$idx">
    <canvas class="basemap"></canvas>
    <canvas class="trails"></canvas>
    <div class="panel-overlay">
      <div class="panel-count">
        <div class="num" id="count-$idx">0</div>
        <div class="label">Unique Vessels</div>
      </div>
      <div class="panel-timestamp" id="ts-$idx">—</div>
      <div class="panel-label">
        <div class="region-index">— $idx_label</div>
        <h2>$label</h2>
        <div class="sea">$sea</div>
        <div class="date" id="daterange-$idx">$daterange</div>
      </div>
    </div>
  </div>
""")


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

# The engine is a single module that:
#   1. Decodes the base64+gzip TopoJSON blobs into GeoJSON features
#   2. For each panel: fits a d3.geoMercator to the panel bbox,
#      renders the basemap once to its basemap canvas, then runs the
#      per-frame trail loop on its trails canvas
#   3. Uses one d3.timer to drive all panels in lockstep
#   4. Exposes window.__timelapse_ready and window.__timelapse_seek for
#      the external MP4 recorder

_JS = Template(r"""
(function () {
'use strict';

// ── Embedded data ─────────────────────────────
const LAND_50M_B64      = "$land_50m_b64";
const US_STATES_10M_B64 = "$us_states_10m_b64";

const PANELS_CFG = $panels_json;
const PALETTE    = $palette_json;
const TYPE_NAMES = $type_names_json;

const CFG = {
  binsPerSecond: $bins_per_second,
  fadeAlpha: $fade_alpha,
  haloRadius: $halo_radius,
  haloAlpha:  $halo_alpha,
  coreRadius: $core_radius,
  coreAlpha:  $core_alpha,
  totalDurationSec: $total_duration_sec,
  includeUsStates: $include_us_states,
};

// ── Decode TopoJSON blobs ─────────────────────
function b64ToTopo(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const text = pako.ungzip(bytes, { to: 'string' });
  return JSON.parse(text);
}

const LAND_TOPO = b64ToTopo(LAND_50M_B64);
const LAND_FEAT = topojson.feature(LAND_TOPO, LAND_TOPO.objects.land);
let US_STATES_FEAT = null;
if (CFG.includeUsStates && US_STATES_10M_B64) {
  const statesTopo = b64ToTopo(US_STATES_10M_B64);
  US_STATES_FEAT = topojson.feature(statesTopo, statesTopo.objects.states);
}

// Explode a MultiPolygon feature into one Feature per individual Polygon ring.
// Needed because world-atlas ``land`` is a single MultiPolygon covering every
// landmass on Earth — we can't filter by bbox until we split it up.
function explodeMultiPolygon(feat) {
  const g = feat.geometry || feat;
  if (g.type === 'Polygon') {
    return [{ type: 'Feature', geometry: g, properties: {} }];
  }
  if (g.type !== 'MultiPolygon') return [feat];
  return g.coordinates.map(function (polyCoords) {
    return {
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: polyCoords },
      properties: {},
    };
  });
}

// Manual flat-bounds computation over a polygon's coordinate arrays.
// Avoids d3.geoBounds' spherical-polygon interpretation which for
// individual landmasses often returns world bounds and breaks the
// bbox filter.
function flatPolygonBounds(poly) {
  let w = Infinity, s = Infinity, e = -Infinity, n = -Infinity;
  const rings = poly.type === 'Polygon' ? poly.coordinates : [];
  for (let r = 0; r < rings.length; r++) {
    const ring = rings[r];
    for (let i = 0; i < ring.length; i++) {
      const x = ring[i][0], y = ring[i][1];
      if (x < w) w = x; if (x > e) e = x;
      if (y < s) s = y; if (y > n) n = y;
    }
  }
  return [w, s, e, n];
}

// Keep only features whose flat lon/lat bounds overlap the expanded
// bbox. At port zoom (bbox ≈ 0.4°), drawing every coastline on Earth
// creates zigzag artifacts because far-away vertices still get
// projected and stroked.
function clipFeaturesToBbox(features, bbox, pad) {
  pad = pad || 2.0;
  const [w, s, e, n] = bbox;
  const kept = [];
  for (const f of features) {
    const g = f.geometry || f;
    if (g.type !== 'Polygon') continue;
    const [fw, fs, fe, fn] = flatPolygonBounds(g);
    if (fe < w - pad || fw > e + pad) continue;
    if (fn < s - pad || fs > n + pad) continue;
    kept.push(f);
  }
  return { type: 'FeatureCollection', features: kept };
}

// Pre-split once — cheap since land-50m has ~1400 polygons. For the US
// states FeatureCollection we flatten each state's MultiPolygon into
// individual polygons too so the bbox filter is per-polygon.
const LAND_PARTS = explodeMultiPolygon(LAND_FEAT);
const US_STATE_PARTS = (function () {
  if (!US_STATES_FEAT) return [];
  const out = [];
  const feats = US_STATES_FEAT.features || [US_STATES_FEAT];
  for (let i = 0; i < feats.length; i++) {
    const parts = explodeMultiPolygon(feats[i]);
    for (let j = 0; j < parts.length; j++) out.push(parts[j]);
  }
  return out;
})();

// ── Panel setup ───────────────────────────────
const panels = [];
document.querySelectorAll('.panel-cell').forEach(function (cellEl, i) {
  const cfg = PANELS_CFG[i];
  const basemap = cellEl.querySelector('canvas.basemap');
  const trails  = cellEl.querySelector('canvas.trails');
  const panel = {
    idx: i,
    cfg: cfg,
    cellEl: cellEl,
    basemap: basemap,
    trails:  trails,
    baseCtx: basemap.getContext('2d'),
    trailCtx: trails.getContext('2d', { willReadFrequently: false, alpha: true }),
    projection: null,
    width: 0, height: 0, dpr: 1,
    lastBinDrawn: -1,
    countEl: document.getElementById('count-' + i),
    tsEl:    document.getElementById('ts-' + i),
  };
  panels.push(panel);
});

// ── Sizing + projection fit ──────────────────
function sizePanel(p) {
  const rect = p.cellEl.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  p.width  = Math.max(1, Math.floor(rect.width));
  p.height = Math.max(1, Math.floor(rect.height));
  p.dpr = dpr;
  [p.basemap, p.trails].forEach(function (c) {
    c.width  = p.width  * dpr;
    c.height = p.height * dpr;
    c.style.width  = p.width  + 'px';
    c.style.height = p.height + 'px';
  });
  p.baseCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  p.trailCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const bbox = p.cfg.bbox; // [west, south, east, north]
  // MultiPoint (not Polygon) avoids d3.geoBounds' spherical-hole ambiguity
  // where a closed ring may be interpreted as "everywhere except this" and
  // force the projection to world scale.
  const bboxGeom = {
    type: 'MultiPoint',
    coordinates: [
      [bbox[0], bbox[1]], [bbox[2], bbox[1]],
      [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ],
  };
  p.projection = d3.geoMercator().fitExtent(
    [[6, 6], [p.width - 6, p.height - 6]],
    bboxGeom,
  );
}

// ── Basemap render (static) ──────────────────
function drawBasemap(p) {
  const ctx = p.baseCtx;
  const path = d3.geoPath(p.projection, ctx);
  ctx.clearRect(0, 0, p.width, p.height);

  // Clip to the panel rect — far-away projected vertices otherwise
  // produce long stroked lines that sweep across the canvas.
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, p.width, p.height);
  ctx.clip();

  // Ocean gradient.
  const grad = ctx.createLinearGradient(0, 0, 0, p.height);
  grad.addColorStop(0.0, '#030510');
  grad.addColorStop(1.0, '#060a18');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, p.width, p.height);

  // Local landmasses only — filter to features intersecting this bbox.
  const localLand = clipFeaturesToBbox(LAND_PARTS, p.cfg.bbox, 3.0);
  ctx.beginPath();
  path(localLand);
  ctx.fillStyle = '#0a1021';
  ctx.fill();
  ctx.lineWidth = 0.7;
  ctx.strokeStyle = 'rgba(82, 120, 180, 0.35)';
  ctx.stroke();

  // US states overlay — only those intersecting the bbox.
  if (US_STATE_PARTS.length) {
    const localStates = clipFeaturesToBbox(US_STATE_PARTS, p.cfg.bbox, 3.0);
    if (localStates.features.length) {
      ctx.beginPath();
      path(localStates);
      ctx.lineWidth = 0.35;
      ctx.strokeStyle = 'rgba(82, 120, 180, 0.22)';
      ctx.stroke();
    }
  }

  // Faint graticule at port scale (0.05° steps).
  const bbox = p.cfg.bbox;
  const span = Math.max(bbox[2] - bbox[0], bbox[3] - bbox[1]);
  const step = span < 0.8 ? 0.1 : span < 3 ? 0.5 : 1.0;
  const graticule = d3.geoGraticule().step([step, step]);
  ctx.beginPath();
  path(graticule());
  ctx.lineWidth = 0.3;
  ctx.strokeStyle = 'rgba(82, 120, 180, 0.07)';
  ctx.stroke();

  ctx.restore();
}

// ── Draw one bin of positions into the trail canvas ──
// Three-pass neon bloom: outer glow (wide, very dim) → mid halo → sharp core.
// Each pass is batched per-color so the whole bin is ~3N fills where N is
// the number of active vessel-type buckets.
function drawBin(p, binIdx) {
  const bin = p.cfg.bins[binIdx];
  if (!bin || !bin.length) return;
  const ctx = p.trailCtx;
  ctx.globalCompositeOperation = 'lighter';

  // Bucket by type-index so each color is one fill call.
  const buckets = new Array(PALETTE.length);
  for (let i = 0; i < PALETTE.length; i++) buckets[i] = [];
  for (let i = 0; i < bin.length; i++) {
    const row = bin[i];
    const lat = row[0], lon = row[1];
    const tidx = row[2] | 0;
    const xy = p.projection([lon, lat]);
    if (!xy) continue;
    // Skip positions that project off-canvas (cheap bounds check).
    if (xy[0] < -20 || xy[0] > p.width + 20) continue;
    if (xy[1] < -20 || xy[1] > p.height + 20) continue;
    const b = buckets[tidx] || buckets[PALETTE.length - 1];
    b.push(xy[0], xy[1]);
  }

  const passes = [
    { r: CFG.haloRadius * 2.2, a: CFG.haloAlpha * 0.4, tint: 0  },  // outer bloom
    { r: CFG.haloRadius,       a: CFG.haloAlpha,       tint: 0  },  // mid halo
    { r: CFG.coreRadius,       a: CFG.coreAlpha,       tint: 80 },  // sharp core
  ];

  for (let pi = 0; pi < passes.length; pi++) {
    const pass = passes[pi];
    for (let t = 0; t < PALETTE.length; t++) {
      const arr = buckets[t];
      if (!arr.length) continue;
      const c = PALETTE[t];
      const r = Math.min(255, c[0] + pass.tint);
      const g = Math.min(255, c[1] + pass.tint);
      const b = Math.min(255, c[2] + pass.tint);
      ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + pass.a + ')';
      ctx.beginPath();
      for (let k = 0; k < arr.length; k += 2) {
        const x = arr[k], y = arr[k + 1];
        ctx.moveTo(x + pass.r, y);
        ctx.arc(x, y, pass.r, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }
}

// ── Per-frame fade (persistence decay) ───────
function fadePanel(p) {
  const ctx = p.trailCtx;
  ctx.globalCompositeOperation = 'destination-in';
  ctx.fillStyle = 'rgba(0,0,0,' + CFG.fadeAlpha + ')';
  ctx.fillRect(0, 0, p.width, p.height);
}

// ── State ────────────────────────────────────
let playing = true;
let startTime = null;
let virtualElapsedMs = 0;  // elapsed in "animation time" — respects pauses / speed
let lastTs = null;
let speedIdx = 0;
const speeds = [1, 2, 4];

function maxBins() {
  return Math.max.apply(null, panels.map(function (p) {
    return p.cfg.bins.length;
  }));
}

function resetTrails() {
  panels.forEach(function (p) {
    p.trailCtx.globalCompositeOperation = 'source-over';
    p.trailCtx.clearRect(0, 0, p.width, p.height);
    p.lastBinDrawn = -1;
  });
}

function sizeAll() {
  panels.forEach(function (p) {
    sizePanel(p);
    drawBasemap(p);
  });
  resetTrails();
}

sizeAll();

// Debounced resize.
let resizeTid = null;
const ro = new ResizeObserver(function () {
  if (resizeTid) clearTimeout(resizeTid);
  resizeTid = setTimeout(function () { sizeAll(); }, 150);
});
panels.forEach(function (p) { ro.observe(p.cellEl); });

// ── Animation loop ───────────────────────────
const dateFmt = d3.utcFormat('%b %d %H:%M');

function frame(now) {
  if (lastTs == null) lastTs = now;
  const dt = now - lastTs;
  lastTs = now;

  if (playing) {
    virtualElapsedMs += dt * speeds[speedIdx];
  }

  // Loop the full animation every totalDurationSec (scaled by speed).
  const loopMs = CFG.totalDurationSec * 1000;
  const progress = (virtualElapsedMs % loopMs) / loopMs; // 0..1
  const n = maxBins();
  const curBinFloat = progress * n;
  const curBin = Math.floor(curBinFloat) % n;

  panels.forEach(function (p) {
    fadePanel(p);
    if (!p.cfg.bins.length) return;
    const pb = Math.floor(curBinFloat * (p.cfg.bins.length / n)) % p.cfg.bins.length;
    if (pb !== p.lastBinDrawn) {
      // Draw any bins we skipped during a jump (scrub or startup).
      if (p.lastBinDrawn < 0 || ((pb - p.lastBinDrawn + p.cfg.bins.length) % p.cfg.bins.length) > 4) {
        drawBin(p, pb);
      } else {
        for (let b = (p.lastBinDrawn + 1) % p.cfg.bins.length;
             b !== pb;
             b = (b + 1) % p.cfg.bins.length) {
          drawBin(p, b);
        }
        drawBin(p, pb);
      }
      p.lastBinDrawn = pb;

      // Update overlay text for this panel.
      const cum = p.cfg.cumulVessels[pb] || 0;
      p.countEl.textContent = d3.format(',')(cum);
      const ts = p.cfg.binTimestamps[pb];
      if (ts) p.tsEl.textContent = dateFmt(new Date(ts));
    }
  });

  // Slider reflect.
  if (!scrubbing) slider.value = Math.round(progress * 1000);

  window.__timelapse_frame = (window.__timelapse_frame | 0) + 1;
  if (window.__timelapse_frame === 2 && !window.__timelapse_ready) {
    window.__timelapse_ready = true;
  }
  requestAnimationFrame(frame);
}

// ── Controls wiring ──────────────────────────
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play-btn');
const speedBtn = document.getElementById('speed-btn');
const speedLabel = document.getElementById('speed-label');
let scrubbing = false;

playBtn.addEventListener('click', function () {
  playing = !playing;
  playBtn.textContent = playing ? '▶' : '⏸';
  playBtn.classList.toggle('active', playing);
});

document.addEventListener('keydown', function (ev) {
  if (ev.code === 'Space') {
    ev.preventDefault();
    playBtn.click();
  }
});

speedBtn.addEventListener('click', function () {
  speedIdx = (speedIdx + 1) % speeds.length;
  const lbl = speeds[speedIdx] + 'x';
  speedBtn.textContent = lbl;
  speedLabel.textContent = lbl;
});

slider.addEventListener('input', function () {
  scrubbing = true;
  const p = (+slider.value) / 1000;
  virtualElapsedMs = p * CFG.totalDurationSec * 1000;
  resetTrails();
});
slider.addEventListener('change', function () {
  scrubbing = false;
});

// External recorder hook.
window.__timelapse_seek = function (seconds) {
  virtualElapsedMs = (seconds || 0) * 1000;
  resetTrails();
};

requestAnimationFrame(frame);
})();
""")


# ---------------------------------------------------------------------------
# Template assembly
# ---------------------------------------------------------------------------


def _build_legend_html(palette: list[list[int]],
                       type_names: list[str],
                       color_by_type: bool) -> str:
    """Build legend HTML from palette and type names."""
    if not color_by_type:
        return ""
    items: list[str] = []
    for i, name in enumerate(type_names):
        if i >= len(palette):
            break
        c = palette[i]
        rgb = f"rgb({c[0]},{c[1]},{c[2]})"
        items.append(
            f'<div class="legend-item" style="color:{rgb}">'
            f'<span class="legend-dot" style="background:{rgb}"></span>'
            f'<span>{name}</span></div>'
        )
    return "\n  ".join(items)


def _safe_json_embed(obj: object) -> str:
    """Serialize *obj* to JSON safe for embedding in a <script> tag.

    Escapes ``<`` to ``\\u003c`` so neither ``</script>`` nor ``<!--``
    can break out of the enclosing script block. JS parses
    ``\\u003c`` back to ``<`` transparently.
    """
    return json.dumps(obj).replace("<", "\\u003c")


def render_timelapse_d3(data: dict) -> str:
    """Assemble the D3 timelapse HTML from template sections.

    ``data`` is produced by ``viz.generate_timelapse_d3()``. See that
    function for the expected keys. Performs network I/O on first call
    to fetch TopoJSON — subsequent calls read from
    ``~/.neptune/viz_assets/``.
    """
    panels = data["panels"]
    n_panels = len(panels)
    palette = data["palette"]
    type_names = data["type_names"]

    topo_blobs = _embed_topojson()

    # Build per-panel JS config — includes bins and timestamps inline
    # (prepare_timelapse already compacts these).
    panels_js: list[dict] = []
    for p in panels:
        panels_js.append({
            "label": p["label"],
            "sea": p.get("sea", ""),
            "bbox": list(p["bbox"]),
            "bins": p["bins"],
            "cumulVessels": p["cumul_vessels"],
            "binTimestamps": p["bin_timestamps_ms"],
            "nBins": len(p["bins"]),
        })

    # Panel cell HTML — one <div class="panel-cell"> per panel.
    cells_html: list[str] = []
    for i, p in enumerate(panels):
        cells_html.append(_PANEL_CELL.substitute(
            idx=str(i),
            idx_label=f"0{i + 1}",
            label=p["label"],
            sea=p.get("sea", ""),
            daterange=p.get("daterange", ""),
        ))

    legend_html = _build_legend_html(palette, type_names, data.get("color_by_type", True))

    js_body = _JS.substitute(
        land_50m_b64=topo_blobs["land_50m"],
        us_states_10m_b64=topo_blobs["us_states_10m"],
        panels_json=_safe_json_embed(panels_js),
        palette_json=_safe_json_embed(palette),
        type_names_json=_safe_json_embed(type_names),
        bins_per_second=data.get("bins_per_second", 3.0),
        fade_alpha=data.get("fade_alpha", 0.94),
        halo_radius=data.get("halo_radius", 4.2),
        halo_alpha=data.get("halo_alpha", 0.16),
        core_radius=data.get("core_radius", 1.3),
        core_alpha=data.get("core_alpha", 0.92),
        total_duration_sec=data.get("total_duration_sec", 15.0),
        include_us_states="true" if data.get("include_us_states", True) else "false",
    )

    html = _HTML_SHELL.substitute(
        title=data.get("title", "AIS TIMELAPSE"),
        eyebrow=data.get("eyebrow", "Vessel Movement — AIS Corridors"),
        css=_CSS,
        n_panels=str(n_panels),
        panel_cells="\n".join(cells_html),
        legend_html=legend_html,
        speed_label=data.get("speed_label", "1x"),
        js=js_body,
    )
    return html
