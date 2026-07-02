"""Timelapse HTML template — D3.js + Canvas 2D renderer.

Cinematic long-exposure vessel corridor timelapse in pure D3 + additive
Canvas 2D. Replicates the reference aesthetic: continuous neon-tube
trails drawn as interpolated per-vessel line segments accumulating on a
slow-fading canvas, white vessel "heads" on an ephemeral layer, dark
TopoJSON basemap, monospace typography, no UI chrome.

Two layouts, selected by panel count:

- ``panels`` (2+): stacked rounded panels, each with a large city name,
  red sea label and static date range (reference: ais_timelapse2.mp4).
- ``single`` (1): framed full-bleed map with a live red clock, city
  labels and a scale bar (reference: ais_timelapse1.mp4).

The animation clock can be driven deterministically via
``window.__tl_render_frame(i)`` so an external recorder produces
perfectly smooth fixed-fps video regardless of screenshot latency.

Internal (``_`` prefix) — import via ``viz.generate_timelapse_d3()``.
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

_CACHE_DIR = Path.home() / ".neptune" / "viz_assets"

_TOPOJSON_SOURCES = {
    "land_50m": "https://cdn.jsdelivr.net/npm/world-atlas@2/land-50m.json",
    "us_states_10m": "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json",
}


def _fetch_topojson(name: str, url: str) -> bytes:
    """Fetch a TopoJSON file to disk cache and return its raw bytes."""
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


def _embed_topojson(include_us_states: bool) -> dict[str, str]:
    """Return TopoJSON payloads base64-gzip-encoded for inlining."""
    result: dict[str, str] = {}
    for name, url in _TOPOJSON_SOURCES.items():
        if name == "us_states_10m" and not include_us_states:
            result[name] = ""
            continue
        raw = _fetch_topojson(name, url)
        compressed = gzip.compress(raw, compresslevel=9)
        result[name] = base64.b64encode(compressed).decode("ascii")
    return result


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = r"""
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
  --font: 'Courier Prime', 'Courier New', ui-monospace, Menlo, monospace;
  --ink: #c6cbd8;
  --ink-dim: #5b6072;
  --red: #e8384f;
  --page: #10101a;
}
html, body {
  width: 100%; height: 100%;
  font-family: var(--font);
  background: var(--page); color: var(--ink); overflow: hidden;
  -webkit-font-smoothing: antialiased; text-rendering: geometricPrecision;
}

/* ── Page header ─────────────────────────── */
#header { position: absolute; top: 26px; left: 32px; right: 32px; z-index: 10; }
#header .eyebrow {
  font-size: 10px; letter-spacing: 3px; text-transform: uppercase;
  color: var(--ink-dim);
}
.layout-panels #header h1 {
  font-size: 15px; font-weight: 400; letter-spacing: 1.5px;
  color: var(--ink); margin-top: 6px;
}
.layout-single #header .eyebrow { color: #b6bcca; }
.layout-single #header h1 {
  font-size: 11px; font-weight: 700; letter-spacing: 2.5px;
  text-transform: uppercase; color: var(--red); margin-top: 6px;
}

#clock {
  position: absolute; top: 30px; right: 34px; z-index: 10;
  font-size: 20px; font-weight: 700; letter-spacing: 2px;
  color: var(--red); font-variant-numeric: tabular-nums;
  display: none;
}
.layout-single #clock { display: block; }

/* ── Panel stack ─────────────────────────── */
#container {
  position: absolute; left: 30px; right: 30px; top: 84px; bottom: 48px;
  display: grid;
  grid-template-rows: repeat(var(--n-panels, 3), 1fr);
  gap: 14px;
}

.panel-cell {
  position: relative; width: 100%; height: 100%;
  overflow: hidden;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  background: #0e0f1a;
}
.layout-single .panel-cell {
  border-radius: 3px;
  border: 1px solid rgba(190, 200, 225, 0.28);
}

canvas.basemap, canvas.trails, canvas.heads {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
canvas.basemap { z-index: 1; }
canvas.trails  { z-index: 2; mix-blend-mode: screen; }
canvas.heads   { z-index: 3; mix-blend-mode: screen; }

.panel-overlay { position: absolute; inset: 0; z-index: 4; pointer-events: none; }

.panel-index {
  position: absolute; left: 18px; top: 14px;
  font-size: 10px; letter-spacing: 2px; color: rgba(255,255,255,0.30);
}
.layout-single .panel-index { display: none; }

.panel-label { position: absolute; left: 20px; bottom: 18px; display: none; }
.layout-panels .panel-label { display: block; }
.panel-label h2 {
  font-size: 24px; font-weight: 700; letter-spacing: 1px;
  color: #f2f5fd;
  text-shadow: 0 1px 10px rgba(0,0,0,0.9);
}
.panel-label .sea {
  font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
  color: var(--red); margin-top: 5px; font-weight: 700;
}
.panel-label .date {
  font-size: 9px; letter-spacing: 1.5px;
  color: rgba(255,255,255,0.32); margin-top: 4px;
  font-variant-numeric: tabular-nums;
}

.city {
  position: absolute; z-index: 4;
  font-size: 11px; letter-spacing: 1px; color: #838b9e;
  white-space: nowrap;
  text-shadow: 0 1px 6px rgba(0,0,0,0.95);
}
.city .tick {
  display: inline-block; width: 9px; height: 1px;
  background: #6a7286; vertical-align: middle; margin: 0 5px;
}

#scalebar {
  position: absolute; left: 20px; bottom: 14px; z-index: 4;
  display: none;
}
.layout-single #scalebar { display: block; }
#scalebar .bar { height: 2px; background: rgba(190, 200, 225, 0.45); }
#scalebar .lbl {
  font-size: 9px; letter-spacing: 1.5px; color: #7d8598; margin-top: 4px;
}

#watermark {
  position: absolute; right: 32px; bottom: 16px; z-index: 10;
  font-size: 11px; letter-spacing: 1px; color: #6a7185;
}
.layout-single #watermark {
  font-size: 17px; font-weight: 700; color: #eef1f8; letter-spacing: 0.5px;
}
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
<body class="layout-$layout">
<div id="header">
  <div class="eyebrow">$eyebrow</div>
  <h1>$title</h1>
</div>
<div id="clock">—</div>

<div id="container">
$panel_cells
</div>

<div id="watermark">$watermark</div>

<script>
$js
</script>
</body>
</html>
""")


_PANEL_CELL = Template(r"""  <div class="panel-cell" data-panel="$idx">
    <canvas class="basemap"></canvas>
    <canvas class="trails"></canvas>
    <canvas class="heads"></canvas>
    <div class="panel-overlay">
      <div class="panel-index">— $idx_label</div>
      <div class="panel-label">
        <h2>$label</h2>
        <div class="sea">$sea</div>
        <div class="date">$daterange</div>
      </div>
      <div class="cities"></div>
    </div>
  </div>
""")

_SCALEBAR_HTML = '<div id="scalebar"><div class="bar"></div><div class="lbl"></div></div>'


# ---------------------------------------------------------------------------
# JavaScript engine
# ---------------------------------------------------------------------------

# Renders per-vessel interpolated trail segments with additive blending:
#   trails canvas — accumulates halo+core strokes, fades by a per-frame
#     destination-in alpha multiply (long-exposure look)
#   heads canvas  — cleared each frame, bright white-tinted glow dots
# The clock is driven either by requestAnimationFrame (interactive) or
# deterministically by window.__tl_render_frame(i) (recorder).

_JS = Template(r"""
(function () {
'use strict';

const LAND_50M_B64      = "$land_50m_b64";
const US_STATES_10M_B64 = "$us_states_10m_b64";

const PANELS_CFG = $panels_json;
const CFG = $cfg_json;
const PAL = CFG.palette;

// ── Decode TopoJSON blobs ─────────────────────
function b64ToTopo(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return JSON.parse(pako.ungzip(bytes, { to: 'string' }));
}

const LAND_TOPO = b64ToTopo(LAND_50M_B64);
const LAND_FEAT = topojson.feature(LAND_TOPO, LAND_TOPO.objects.land);
let US_STATES_FEAT = null;
if (US_STATES_10M_B64) {
  const statesTopo = b64ToTopo(US_STATES_10M_B64);
  US_STATES_FEAT = topojson.feature(statesTopo, statesTopo.objects.states);
}

// Explode a MultiPolygon into one Feature per Polygon so the bbox
// filter below is per-landmass (world-atlas 'land' is one MultiPolygon).
function explodeMultiPolygon(feat) {
  const g = feat.geometry || feat;
  if (g.type === 'Polygon') {
    return [{ type: 'Feature', geometry: g, properties: {} }];
  }
  if (g.type !== 'MultiPolygon') return [feat];
  return g.coordinates.map(function (polyCoords) {
    return { type: 'Feature',
             geometry: { type: 'Polygon', coordinates: polyCoords },
             properties: {} };
  });
}

// Flat (non-spherical) bounds — d3.geoBounds treats each polygon as a
// spherical region and can return world bounds for coastline rings.
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

function clipFeaturesToBbox(features, bbox, pad) {
  pad = pad || 2.0;
  const w = bbox[0], s = bbox[1], e = bbox[2], n = bbox[3];
  const kept = [];
  for (const f of features) {
    const g = f.geometry || f;
    if (g.type !== 'Polygon') continue;
    const b = flatPolygonBounds(g);
    if (b[2] < w - pad || b[0] > e + pad) continue;
    if (b[3] < s - pad || b[1] > n + pad) continue;
    kept.push(f);
  }
  return { type: 'FeatureCollection', features: kept };
}

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

// ── Color helpers ─────────────────────────────
function rgba(c, a) {
  return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
}
function lighten(c, amt) {
  return [Math.min(255, c[0] + amt), Math.min(255, c[1] + amt), Math.min(255, c[2] + amt)];
}

// ── Panel setup ───────────────────────────────
const panels = [];
document.querySelectorAll('.panel-cell').forEach(function (cellEl, i) {
  panels.push({
    idx: i,
    cfg: PANELS_CFG[i],
    cellEl: cellEl,
    baseCtx:  cellEl.querySelector('canvas.basemap').getContext('2d'),
    trailCtx: cellEl.querySelector('canvas.trails').getContext('2d'),
    headCtx:  cellEl.querySelector('canvas.heads').getContext('2d'),
    citiesEl: cellEl.querySelector('.cities'),
    projection: null,
    vessels: [],
    width: 0, height: 0,
  });
});

function sizePanel(p) {
  const rect = p.cellEl.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  p.width  = Math.max(1, Math.floor(rect.width));
  p.height = Math.max(1, Math.floor(rect.height));
  p.cellEl.querySelectorAll('canvas').forEach(function (c) {
    c.width  = p.width  * dpr;
    c.height = p.height * dpr;
  });
  [p.baseCtx, p.trailCtx, p.headCtx].forEach(function (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  });

  const bbox = p.cfg.bbox;
  // MultiPoint avoids d3.geoBounds' spherical-hole ambiguity for rings.
  p.projection = d3.geoMercator().fitExtent(
    [[0, 0], [p.width, p.height]],
    { type: 'MultiPoint',
      coordinates: [[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]], [bbox[0], bbox[3]]] },
  );
}

// ── Basemap (static) ─────────────────────────
function drawBasemap(p) {
  const ctx = p.baseCtx;
  const path = d3.geoPath(p.projection, ctx);
  ctx.clearRect(0, 0, p.width, p.height);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, p.width, p.height);
  ctx.clip();

  const grad = ctx.createLinearGradient(0, 0, 0, p.height);
  grad.addColorStop(0.0, '#0e1020');
  grad.addColorStop(1.0, '#0c0d18');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, p.width, p.height);

  const localLand = clipFeaturesToBbox(LAND_PARTS, p.cfg.bbox, 3.0);
  ctx.beginPath();
  path(localLand);
  ctx.fillStyle = '#1d2233';
  ctx.fill();
  ctx.lineWidth = 0.8;
  ctx.strokeStyle = 'rgba(125, 145, 195, 0.28)';
  ctx.stroke();

  if (US_STATE_PARTS.length) {
    const localStates = clipFeaturesToBbox(US_STATE_PARTS, p.cfg.bbox, 3.0);
    if (localStates.features.length) {
      ctx.beginPath();
      path(localStates);
      ctx.lineWidth = 0.4;
      ctx.strokeStyle = 'rgba(120, 140, 190, 0.10)';
      ctx.stroke();
    }
  }
  ctx.restore();
}

// ── Decode quantized tracks → screen space ───
// Track wire format: [typeIdx, t,x,y, t,x,y, ...] with t,x,y uint16
// quantized over [t0,t1] × bbox. Projected once per resize.
function buildVessels(p) {
  const cfg = p.cfg;
  const w = cfg.bbox[0], s = cfg.bbox[1], e = cfg.bbox[2], n = cfg.bbox[3];
  const tSpan = cfg.t1 - cfg.t0;
  p.vessels = cfg.tracks.map(function (arr) {
    const m = ((arr.length - 1) / 3) | 0;
    const T = new Float64Array(m), X = new Float32Array(m), Y = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      T[i] = cfg.t0 + (arr[1 + i * 3] / 65535) * tSpan;
      const lon = w + (arr[2 + i * 3] / 65535) * (e - w);
      const lat = s + (arr[3 + i * 3] / 65535) * (n - s);
      const xy = p.projection([lon, lat]);
      X[i] = xy[0]; Y[i] = xy[1];
    }
    return { t: arr[0], T: T, X: X, Y: Y, cur: 0, px: 0, py: 0, has: false };
  });
}

// ── City labels + scale bar (DOM, static) ────
function placeCities(p) {
  p.citiesEl.innerHTML = '';
  (p.cfg.cities || []).forEach(function (c) {
    const xy = p.projection([c.lon, c.lat]);
    if (!xy) return;
    const el = document.createElement('div');
    el.className = 'city';
    const anchorRight = c.anchor === 'right';
    el.innerHTML = anchorRight
      ? '<span class="tick"></span>' + c.name
      : c.name + '<span class="tick"></span>';
    el.style.top = (xy[1] - 7) + 'px';
    if (anchorRight) el.style.left = (xy[0] + 3) + 'px';
    p.citiesEl.appendChild(el);
    if (!anchorRight) {
      // Right-align text so the tick ends at the city point.
      el.style.left = (xy[0] - el.offsetWidth - 3) + 'px';
    }
  });
}

function placeScalebar(p) {
  const el = document.getElementById('scalebar');
  if (!el || CFG.layout !== 'single') return;
  // km per pixel at panel centre via haversine on two projected points.
  const cy = (p.cfg.bbox[1] + p.cfg.bbox[3]) / 2;
  const cx = (p.cfg.bbox[0] + p.cfg.bbox[2]) / 2;
  const a = p.projection([cx, cy]), b = p.projection([cx + 0.5, cy]);
  const kmPerHalfDeg = 111.32 * 0.5 * Math.cos(cy * Math.PI / 180);
  const kmPerPx = kmPerHalfDeg / Math.abs(b[0] - a[0]);
  const nice = [5, 10, 20, 25, 50, 100, 200, 500];
  let best = nice[0];
  for (const k of nice) {
    if (Math.abs(k / kmPerPx - 95) < Math.abs(best / kmPerPx - 95)) best = k;
  }
  el.querySelector('.bar').style.width = Math.round(best / kmPerPx) + 'px';
  el.querySelector('.lbl').textContent = best + 'KM';
}

// ── Trail rendering ──────────────────────────
// Per frame: multiply trail alpha by a constant (slow long-exposure
// fade), then stroke this frame's movement segments additively in two
// passes (wide dim halo, thin bright core), batched per colour class.
const fadeKeep = Math.pow(0.5, 1 / (CFG.fadeHalfLifeSec * CFG.fps));
const maxJump2 = CFG.maxJumpPx * CFG.maxJumpPx;
const minMove2 = CFG.minMovePx * CFG.minMovePx;

function fadePanel(p) {
  const ctx = p.trailCtx;
  ctx.globalCompositeOperation = 'destination-in';
  ctx.fillStyle = 'rgba(0,0,0,' + fadeKeep + ')';
  ctx.fillRect(0, 0, p.width, p.height);
}

function drawStep(p, dataT) {
  const segPaths = new Array(PAL.length);
  const headPts = new Array(PAL.length);

  for (let vi = 0; vi < p.vessels.length; vi++) {
    const v = p.vessels[vi];
    const m = v.T.length;
    if (dataT < v.T[0] || dataT > v.T[m - 1]) {
      v.has = false; v.cur = 0;
      continue;
    }
    let c = v.cur;
    if (v.T[c] > dataT) c = 0;
    while (c < m - 2 && v.T[c + 1] < dataT) c++;
    v.cur = c;
    const sdx = v.X[c + 1] - v.X[c], sdy = v.Y[c + 1] - v.Y[c];
    let x, y;
    if (sdx * sdx + sdy * sdy > maxJump2) {
      // Over-long segment (bbox exit/re-entry, AIS dropout): park at
      // the last real point instead of sweeping the chord.
      x = v.X[c]; y = v.Y[c];
    } else {
      const span = v.T[c + 1] - v.T[c];
      const f = span > 0 ? (dataT - v.T[c]) / span : 0;
      x = v.X[c] + sdx * f;
      y = v.Y[c] + sdy * f;
    }

    if (v.has) {
      const dx = x - v.px, dy = y - v.py;
      const d2 = dx * dx + dy * dy;
      if (d2 >= maxJump2) {
        v.px = x; v.py = y;  // data gap / teleport: restart trail silently
      } else if (d2 > minMove2) {
        let path = segPaths[v.t];
        if (!path) path = segPaths[v.t] = new Path2D();
        path.moveTo(v.px, v.py);
        path.lineTo(x, y);
        v.px = x; v.py = y;
      }
      // Below minMove: keep the anchor so slow drift accumulates into
      // one segment instead of anchored GPS jitter stamping blobs.
    } else {
      v.px = x; v.py = y; v.has = true;
    }

    let hp = headPts[v.t];
    if (!hp) hp = headPts[v.t] = [];
    hp.push(x, y);
  }

  // Trails — additive bloom + halo + core (wide→narrow, dim→bright).
  const ctx = p.trailCtx;
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (let k = 0; k < PAL.length; k++) {
    const path = segPaths[k];
    if (!path) continue;
    ctx.lineWidth = CFG.bloomWidth;
    ctx.strokeStyle = rgba(PAL[k], CFG.bloomAlpha);
    ctx.stroke(path);
    ctx.lineWidth = CFG.haloWidth;
    ctx.strokeStyle = rgba(PAL[k], CFG.haloAlpha);
    ctx.stroke(path);
    ctx.lineWidth = CFG.coreWidth;
    ctx.strokeStyle = rgba(PAL[k], CFG.coreAlpha);
    ctx.stroke(path);
  }

  // Heads — ephemeral bright dots, white-tinted per class.
  const h = p.headCtx;
  h.clearRect(0, 0, p.width, p.height);
  h.globalCompositeOperation = 'lighter';
  for (let k = 0; k < PAL.length; k++) {
    const pts = headPts[k];
    if (!pts) continue;
    const glow = new Path2D(), core = new Path2D();
    for (let i = 0; i < pts.length; i += 2) {
      glow.moveTo(pts[i] + CFG.headGlowR, pts[i + 1]);
      glow.arc(pts[i], pts[i + 1], CFG.headGlowR, 0, 6.2832);
      core.moveTo(pts[i] + CFG.headCoreR, pts[i + 1]);
      core.arc(pts[i], pts[i + 1], CFG.headCoreR, 0, 6.2832);
    }
    h.fillStyle = rgba(lighten(PAL[k], 90), CFG.headGlowAlpha);
    h.fill(glow);
    h.fillStyle = rgba(lighten(PAL[k], 200), CFG.headCoreAlpha);
    h.fill(core);
  }
}

// ── Clock ────────────────────────────────────
const clockEl = document.getElementById('clock');
const clockFmt = d3.utcFormat('%b %d %H:%M');

// ── Render clock-step ────────────────────────
let lastVideoMs = null;

function hardReset() {
  panels.forEach(function (p) {
    p.trailCtx.clearRect(0, 0, p.width, p.height);
    p.headCtx.clearRect(0, 0, p.width, p.height);
    p.vessels.forEach(function (v) { v.cur = 0; v.has = false; });
  });
  lastVideoMs = null;
}

function renderStep(videoMs) {
  if (lastVideoMs !== null && videoMs < lastVideoMs) hardReset();
  lastVideoMs = videoMs;
  const prog = Math.min(videoMs / (CFG.totalDurationSec * 1000), 1);
  panels.forEach(function (p) {
    const dataT = p.cfg.t0 + prog * (p.cfg.t1 - p.cfg.t0);
    fadePanel(p);
    drawStep(p, dataT);
  });
  if (CFG.layout === 'single' && clockEl) {
    const t0 = panels[0].cfg.t0, t1 = panels[0].cfg.t1;
    clockEl.textContent = clockFmt(new Date(t0 + prog * (t1 - t0)));
  }
}

// ── Boot ─────────────────────────────────────
function sizeAll() {
  panels.forEach(function (p) {
    sizePanel(p);
    drawBasemap(p);
    buildVessels(p);
    placeCities(p);
  });
  placeScalebar(panels[0]);
  hardReset();
}

sizeAll();

let resizeTid = null;
window.addEventListener('resize', function () {
  if (resizeTid) clearTimeout(resizeTid);
  resizeTid = setTimeout(sizeAll, 150);
});

// Interactive playback via rAF; the recorder switches to manual
// stepping so every captured frame advances exactly 1/fps.
let manual = false;
let rafStart = null;

function rafLoop(now) {
  if (manual) return;
  if (rafStart === null) rafStart = now;
  renderStep((now - rafStart) % (CFG.totalDurationSec * 1000));
  requestAnimationFrame(rafLoop);
}
requestAnimationFrame(rafLoop);

window.__tl_set_manual = function (m) {
  manual = !!m;
  if (m) { hardReset(); } else { rafStart = null; requestAnimationFrame(rafLoop); }
};
window.__tl_render_frame = function (i) {
  renderStep(i * 1000 / CFG.fps);
};
window.__tl_ready = true;
})();
""")


# ---------------------------------------------------------------------------
# Template assembly
# ---------------------------------------------------------------------------


def _safe_json_embed(obj: object) -> str:
    """Serialize *obj* to JSON safe for embedding in a <script> tag."""
    return json.dumps(obj, separators=(",", ":")).replace("<", "\\u003c")


def render_timelapse_d3(data: dict) -> str:
    """Assemble the D3 timelapse HTML from template sections.

    ``data`` is produced by ``viz.generate_timelapse_d3()``. Performs
    network I/O on first call to fetch TopoJSON — subsequent calls read
    from ``~/.neptune/viz_assets/``.
    """
    panels = data["panels"]
    n_panels = len(panels)
    layout = "single" if n_panels == 1 else "panels"

    include_us = bool(data.get("include_us_states", True))
    topo_blobs = _embed_topojson(include_us)

    panels_js: list[dict] = []
    cells_html: list[str] = []
    for i, p in enumerate(panels):
        panels_js.append({
            "label": p["label"],
            "sea": p.get("sea", ""),
            "bbox": list(p["bbox"]),
            "t0": p["t0_ms"],
            "t1": p["t1_ms"],
            "tracks": p["tracks"],
            "cities": p.get("cities", []),
        })
        cells_html.append(_PANEL_CELL.substitute(
            idx=str(i),
            idx_label=f"0{i + 1}",
            label=p["label"],
            sea=p.get("sea", ""),
            daterange=p.get("daterange", ""),
        ))
    if layout == "single":
        # Scale bar lives inside the (only) panel overlay.
        cells_html[0] = cells_html[0].replace(
            '<div class="cities"></div>',
            '<div class="cities"></div>\n      ' + _SCALEBAR_HTML,
        )

    cfg = {
        "fps": data.get("fps", 60),
        "totalDurationSec": data.get("total_duration_sec", 15.0),
        "fadeHalfLifeSec": data.get("fade_half_life_sec", 3.0),
        "layout": layout,
        "palette": data["palette"],
        "bloomWidth": data.get("bloom_width", 9.0),
        "bloomAlpha": data.get("bloom_alpha", 0.0025),
        "haloWidth": data.get("halo_width", 3.2),
        "haloAlpha": data.get("halo_alpha", 0.010),
        "coreWidth": data.get("core_width", 1.1),
        "coreAlpha": data.get("core_alpha", 0.055),
        "headGlowR": data.get("head_glow_r", 2.0),
        "headGlowAlpha": data.get("head_glow_alpha", 0.30),
        "headCoreR": data.get("head_core_r", 0.9),
        "headCoreAlpha": data.get("head_core_alpha", 0.85),
        "minMovePx": data.get("min_move_px", 1.6),
        "maxJumpPx": data.get("max_jump_px", 40.0),
    }

    js_body = _JS.substitute(
        land_50m_b64=topo_blobs["land_50m"],
        us_states_10m_b64=topo_blobs["us_states_10m"],
        panels_json=_safe_json_embed(panels_js),
        cfg_json=_safe_json_embed(cfg),
    )

    return _HTML_SHELL.substitute(
        title=data.get("title", "AIS TIMELAPSE"),
        eyebrow=data.get("eyebrow", "VESSEL MOVEMENT — AIS TIMELAPSE"),
        watermark=data.get("watermark", "neptune"),
        css=_CSS,
        layout=layout,
        n_panels=str(n_panels),
        panel_cells="\n".join(cells_html),
        js=js_body,
    )
