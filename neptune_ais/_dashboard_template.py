"""Dashboard HTML template — self-contained maritime intelligence UI.

Assembled from named ``string.Template`` sections and rendered by
:func:`viz.generate_dashboard`.  Uses ``$$var`` substitution so that
JavaScript braces (``{`` / ``}``) are literal — no double-brace
escaping needed.

This module is internal (``_`` prefix).  Import only via
``viz.generate_dashboard()``.
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
  background: #0a0a1a; color: #e0e0e0; overflow: hidden;
}
#map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }

/* ── panels ─────────────────────────────────── */
.panel {
  position: absolute; z-index: 10;
  background: rgba(10, 10, 30, 0.92);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px; padding: 16px;
  backdrop-filter: blur(14px);
  box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
#left-panel {
  top: 16px; left: 16px; width: 320px;
  max-height: calc(100vh - 100px); overflow-y: auto;
}
#top-center {
  top: 16px; left: 50%; transform: translateX(-50%);
  text-align: center; min-width: 260px;
}
#right-panel {
  top: 16px; right: 16px; width: 280px;
  max-height: calc(100vh - 100px); overflow-y: auto;
}
#bottom-bar {
  bottom: 0; left: 0; right: 0;
  border-radius: 0; display: flex; align-items: center;
  gap: 12px; padding: 10px 20px;
}

/* ── typography ─────────────────────────────── */
h1 { font-size: 18px; font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; }
h2 { font-size: 13px; font-weight: 600; text-transform: uppercase;
     letter-spacing: 1px; color: rgba(255,255,255,0.5); margin-bottom: 8px; }
.desc { font-size: 12px; color: rgba(255,255,255,0.55); line-height: 1.5;
        margin-bottom: 12px; font-family: -apple-system, sans-serif; }
.big-num { font-size: 48px; font-weight: 700; line-height: 1;
           font-variant-numeric: tabular-nums; }
.metric { text-align: center; }
.metric-val { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
                color: rgba(255,255,255,0.45); margin-top: 2px; }
.metrics-row { display: flex; gap: 16px; margin: 10px 0; }
.cyan { color: #00c8ff; }
.orange { color: #ff6432; }
.green { color: #32ff82; }
.red { color: #ff6464; }

/* ── buttons ────────────────────────────────── */
.btn {
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
  color: #ccc; padding: 5px 12px; border-radius: 5px; cursor: pointer;
  font-size: 12px; font-family: inherit; transition: all 0.15s;
}
.btn:hover { background: rgba(255,255,255,0.15); color: #fff; }
.btn.active { background: rgba(0,200,255,0.25); border-color: rgba(0,200,255,0.5);
              color: #00c8ff; }
.btn-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }

/* ── chart canvas ───────────────────────────── */
.chart-container { position: relative; width: 100%; height: 100px;
                   margin: 8px 0; }
.chart-container canvas { width: 100%; height: 100%; }

/* ── mode tabs ──────────────────────────────── */
.tabs { display: flex; gap: 0; margin: 10px 0; border-radius: 6px; overflow: hidden; }
.tab {
  flex: 1; padding: 6px 8px; text-align: center; font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
  color: rgba(255,255,255,0.5); transition: all 0.15s;
}
.tab:first-child { border-radius: 6px 0 0 6px; }
.tab:last-child { border-radius: 0 6px 6px 0; }
.tab.active { background: rgba(0,200,255,0.2); border-color: rgba(0,200,255,0.4);
              color: #00c8ff; }

/* ── vessel detail card ─────────────────────── */
#vessel-card { display: none; margin-top: 12px; padding-top: 12px;
               border-top: 1px solid rgba(255,255,255,0.1); }
#vessel-card.visible { display: block; }
#vessel-card .v-name { font-size: 16px; font-weight: 700; margin-bottom: 2px; }
#vessel-card .v-type { font-size: 11px; color: rgba(255,255,255,0.5);
                       text-transform: uppercase; margin-bottom: 8px; }
#vessel-card .v-row { display: flex; justify-content: space-between;
                      font-size: 12px; padding: 3px 0;
                      border-bottom: 1px solid rgba(255,255,255,0.05); }
#vessel-card .v-label { color: rgba(255,255,255,0.4); }

/* ── search ─────────────────────────────────── */
#search-input {
  width: 100%; padding: 6px 10px; border-radius: 5px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.15);
  color: #e0e0e0; font-size: 12px; font-family: inherit; outline: none;
  margin: 6px 0;
}
#search-input::placeholder { color: rgba(255,255,255,0.3); }
#search-input:focus { border-color: rgba(0,200,255,0.5); }

/* ── timeline ───────────────────────────────── */
#timeline-slider { flex: 1; accent-color: #00c8ff; height: 4px; }
#clock { font-variant-numeric: tabular-nums; font-size: 13px;
         color: #00c8ff; font-weight: 600; min-width: 170px; text-align: right; }
#active-count { font-size: 11px; color: rgba(255,255,255,0.4); min-width: 140px; }
.sparkline-container { width: 100%; height: 30px; position: relative; }
.sparkline-container canvas { width: 100%; height: 100%; }

/* ── layer toggles ──────────────────────────── */
.layer-toggle { display: flex; align-items: center; gap: 6px; margin: 3px 0;
                font-size: 12px; cursor: pointer; }
.layer-toggle input { accent-color: #00c8ff; }

/* ── no-gate mode ───────────────────────────── */
body.no-gate .gate-only { display: none !important; }

/* ── scrollbar ──────────────────────────────── */
.panel::-webkit-scrollbar { width: 4px; }
.panel::-webkit-scrollbar-track { background: transparent; }
.panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 2px; }
"""

# ---------------------------------------------------------------------------
# HTML structure
# ---------------------------------------------------------------------------

_HEAD = Template("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — $title</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/deck.gl@9.1.4/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
""" + _CSS + """
</style>
</head>
<body class="$body_class">
<div id="map"></div>
""")

_LEFT_PANEL = Template("""\
<div id="left-panel" class="panel">
  <h1>$title</h1>
  <p class="desc">$description</p>

  <div class="tabs gate-only">
    <div class="tab active" data-mode="FULL_PERIOD">Full Period</div>
    <div class="tab" data-mode="OVERVIEW">Overview</div>
    <div class="tab" data-mode="PLAYBACK_LENS">Playback Lens</div>
  </div>

  <div class="chart-container gate-only">
    <canvas id="chart-canvas"></canvas>
  </div>

  <div class="metrics-row gate-only">
    <div class="metric">
      <div class="metric-val" id="stat-avg">—</div>
      <div class="metric-label">Avg / Day</div>
    </div>
    <div class="metric">
      <div class="metric-val" id="stat-transit">—</div>
      <div class="metric-label">Transit Vessels</div>
    </div>
    <div class="metric">
      <div class="metric-val" id="stat-delta">—</div>
      <div class="metric-label">Delta vs Pre</div>
    </div>
  </div>

  <div class="metrics-row">
    <div class="metric">
      <div class="metric-val cyan" id="stat-tracked">—</div>
      <div class="metric-label">Tracked Vessels</div>
    </div>
    <div class="metric">
      <div class="metric-val" id="stat-crossings">—</div>
      <div class="metric-label">Crossing Events</div>
    </div>
  </div>

  <div style="margin-top:12px">
    <button class="btn" id="zoom-in" title="Zoom in">+</button>
    <button class="btn" id="zoom-out" title="Zoom out">&minus;</button>
    <button class="btn" id="reset-view">Reset View</button>
  </div>

  <div id="subset-note" style="margin-top:8px;font-size:11px;color:rgba(255,255,255,0.35);display:none;"></div>
</div>
""")

_TOP_CENTER = Template("""\
<div id="top-center" class="panel gate-only">
  <div style="font-size:12px;color:rgba(255,255,255,0.5)" id="current-date">—</div>
  <div class="big-num cyan" id="live-count">0</div>
  <div style="font-size:11px;color:rgba(255,255,255,0.5);margin:4px 0">Unique Vessels Crossed</div>
  <div class="metrics-row" style="justify-content:center">
    <div class="metric">
      <div class="metric-val green" id="live-inbound">0</div>
      <div class="metric-label">&uarr; Inbound</div>
    </div>
    <div class="metric">
      <div class="metric-val orange" id="live-outbound">0</div>
      <div class="metric-label">&darr; Outbound</div>
    </div>
  </div>
  <div style="font-size:11px;color:rgba(255,255,255,0.4)" id="live-reversals">0 same-hull reversals today</div>
</div>
""")

_RIGHT_PANEL = Template("""\
<div id="right-panel" class="panel">
  <h2>Filters</h2>

  <div class="gate-only">
    <div class="btn-row" id="period-btns">
      <button class="btn active" data-period="FULL">Full Period</button>
      <button class="btn" data-period="BEFORE">Before</button>
      <button class="btn" data-period="AFTER">After</button>
    </div>
  </div>

  <input type="text" id="search-input" placeholder="Name, MMSI, IMO...">

  <div class="gate-only">
    <div class="btn-row" id="transit-btns">
      <button class="btn" data-filter="transitOnly">Transit Only</button>
    </div>
  </div>

  <h2 style="margin-top:10px">Flags</h2>
  <div class="btn-row" id="flag-btns"></div>

  <h2 style="margin-top:10px">Type</h2>
  <div class="btn-row" id="type-btns"></div>

  <h2 style="margin-top:10px">Layers</h2>
  <div id="layer-toggles">
    <label class="layer-toggle"><input type="checkbox" data-layer="trips" checked> Trips</label>
    <label class="layer-toggle"><input type="checkbox" data-layer="heads" checked> Heads</label>
    <label class="layer-toggle"><input type="checkbox" data-layer="tracks"> Tracks</label>
    <label class="layer-toggle gate-only"><input type="checkbox" data-layer="gates" checked> Gates</label>
    <label class="layer-toggle"><input type="checkbox" data-layer="density"> Density</label>
    <label class="layer-toggle"><input type="checkbox" data-layer="infrastructure" checked> Infrastructure</label>
  </div>

  <div id="vessel-card">
    <div class="v-name" id="vc-name"></div>
    <div class="v-type" id="vc-type"></div>
    <div id="vc-details"></div>
  </div>
</div>
""")

_BOTTOM_BAR = Template("""\
<div id="bottom-bar" class="panel">
  <button class="btn" id="play-btn" title="Play / Pause (Space)">&#9654;</button>
  <div class="btn-row" id="speed-btns" style="margin:0"></div>
  <input type="range" id="timeline-slider" min="0" max="10000" value="0">
  <span id="clock">—</span>
  <span id="active-count"></span>
</div>
<div class="panel" style="bottom:50px;left:20px;right:20px;padding:4px 12px;border-radius:6px;">
  <div class="sparkline-container"><canvas id="sparkline-canvas"></canvas></div>
</div>
""")

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS_DATA = Template("""\
<script>
// ── Embedded data ──────────────────────────────
const TRIPS = $trips_json;
const ANALYTICS = $analytics_json;
const VESSEL_INDEX = $vessel_index_json;
const DENSITY_DATA = $density_json;
const EVENT_DATA = $events_json;
const INFRA_DATA = $infra_json;
const GATE_COORDS = $gate_coords;
const HAS_GATE = $has_gate;
const MAX_TIME = $max_time;
const TRAIL_LENGTH = $trail_length;
const GLOBAL_START_MS = $global_start_ms;
const EVENT_DATE = '$event_date';
const DATE_FROM = '$date_from';
const DATE_TO = '$date_to';
""")

_JS_STATE = Template("""\
// ── State ──────────────────────────────────────
const SPEEDS = [
  {value: 7200, label: '2H/S'},
  {value: 21600, label: '6H/S'},
  {value: 86400, label: '1D/S'},
];
const state = {
  playing: false,
  currentTime: 0,
  speed: $default_speed,
  selectedMmsi: null,
  filters: {
    flags: new Set(),
    shipTypes: new Set(),
    transitOnly: false,
    period: 'FULL',
    search: '',
  },
  layers: {
    trips: true, heads: true, tracks: false, density: false,
    infrastructure: true, gates: true,
  },
  mode: 'FULL_PERIOD',
};

// ── Filter logic ───────────────────────────────
function getVisibleIndices() {
  const vis = [];
  const s = state.filters;
  const searchLower = s.search.toLowerCase();
  for (let i = 0; i < TRIPS.length; i++) {
    const t = TRIPS[i];
    if (s.transitOnly && !t.isTransit) continue;
    if (s.flags.size > 0 && !s.flags.has(t.flag)) continue;
    if (s.shipTypes.size > 0 && !s.shipTypes.has(t.shipType)) continue;
    if (searchLower) {
      const mmsiStr = String(t.mmsi);
      const name = (t.name || '').toLowerCase();
      const imo = (VESSEL_INDEX[mmsiStr] || {}).imo || '';
      if (!mmsiStr.includes(searchLower) &&
          !name.includes(searchLower) &&
          !String(imo).toLowerCase().includes(searchLower)) continue;
    }
    vis.push(i);
  }
  return vis;
}

let visibleSet = new Set(TRIPS.map((_, i) => i));
function applyFilters() {
  visibleSet = new Set(getVisibleIndices());
  updateLayers();
  drawChart();
}
""")

_JS_LAYERS = Template("""\
// ── Deck.gl ────────────────────────────────────
const INITIAL_VIEW = {
  longitude: $center_lon,
  latitude: $center_lat,
  zoom: $zoom,
  pitch: $pitch,
  bearing: $bearing,
};

// Arrow icon: draw a small upward-pointing triangle on a canvas, use as icon.
const ARROW_SIZE = 32;
const _arrowCanvas = document.createElement('canvas');
_arrowCanvas.width = ARROW_SIZE;
_arrowCanvas.height = ARROW_SIZE;
const _actx = _arrowCanvas.getContext('2d');
_actx.clearRect(0, 0, ARROW_SIZE, ARROW_SIZE);
_actx.fillStyle = '#ffffff';
_actx.beginPath();
_actx.moveTo(ARROW_SIZE / 2, 2);
_actx.lineTo(ARROW_SIZE - 4, ARROW_SIZE - 4);
_actx.lineTo(ARROW_SIZE / 2, ARROW_SIZE - 10);
_actx.lineTo(4, ARROW_SIZE - 4);
_actx.closePath();
_actx.fill();
const ARROW_ICON_MAPPING = {arrow: {x: 0, y: 0, width: ARROW_SIZE, height: ARROW_SIZE, anchorY: ARROW_SIZE / 2}};

const deckgl = new deck.DeckGL({
  container: 'map',
  mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  initialViewState: {...INITIAL_VIEW},
  controller: true,
  onClick: (info) => {
    if (info.object && info.object.mmsi != null) {
      selectVessel(info.object.mmsi);
    } else {
      deselectVessel();
    }
  },
  layers: [],
});

function getFilteredTrips() {
  return TRIPS.filter((_, i) => visibleSet.has(i));
}

function updateLayers() {
  const ct = state.currentTime;
  const filtered = getFilteredTrips();
  const layers = [];

  // 1. Trips layer
  if (state.layers.trips) {
    layers.push(new deck.TripsLayer({
      id: 'trips',
      data: filtered,
      getPath: d => d.path,
      getTimestamps: d => d.timestamps,
      getColor: d => {
        if (state.selectedMmsi != null) {
          return d.mmsi === state.selectedMmsi ? d.color : [d.color[0], d.color[1], d.color[2], 60];
        }
        return d.color;
      },
      currentTime: ct,
      trailLength: TRAIL_LENGTH,
      widthMinPixels: 3,
      widthMaxPixels: 8,
      capRounded: true,
      jointRounded: true,
      opacity: 0.9,
    }));
  }

  // 2. Vessel heads (arrow markers with bearing)
  if (state.layers.heads) {
    const heads = filtered.filter(d => {
      const ts = d.timestamps;
      return ts[0] <= ct && ct <= ts[ts.length - 1];
    }).map(d => {
      const ts = d.timestamps;
      let idx = 0;
      for (let i = 0; i < ts.length - 1; i++) {
        if (ts[i + 1] >= ct) { idx = i; break; }
      }
      const frac = ts[idx + 1] !== ts[idx]
        ? (ct - ts[idx]) / (ts[idx + 1] - ts[idx]) : 0;
      const p0 = d.path[idx], p1 = d.path[Math.min(idx + 1, d.path.length - 1)];
      const lon = p0[0] + (p1[0] - p0[0]) * frac;
      const lat = p0[1] + (p1[1] - p0[1]) * frac;
      // Bearing: degrees clockwise from north (atan2 of delta-lon, delta-lat).
      const dlat = p1[1] - p0[1];
      const dlon = p1[0] - p0[0];
      const bearing = (Math.atan2(dlon, dlat) * 180 / Math.PI + 360) % 360;
      return {
        position: [lon, lat],
        color: d.color,
        mmsi: d.mmsi,
        bearing: bearing,
      };
    });
    layers.push(new deck.IconLayer({
      id: 'heads',
      data: heads,
      iconAtlas: _arrowCanvas,
      iconMapping: ARROW_ICON_MAPPING,
      getIcon: () => 'arrow',
      getPosition: d => d.position,
      getAngle: d => 360 - d.bearing,
      getSize: 24,
      getColor: d => {
        if (state.selectedMmsi != null && d.mmsi !== state.selectedMmsi) {
          return [...d.color, 60];
        }
        return [...d.color, 255];
      },
      sizeUnits: 'pixels',
      pickable: true,
    }));
  }

  // 3. Static track lines
  if (state.layers.tracks) {
    layers.push(new deck.PathLayer({
      id: 'tracks',
      data: filtered,
      getPath: d => d.path,
      getColor: d => {
        if (state.selectedMmsi != null) {
          return d.mmsi === state.selectedMmsi
            ? [...d.color, 200] : [...d.color, 30];
        }
        return [...d.color, 120];
      },
      widthMinPixels: 1,
      widthMaxPixels: 3,
      capRounded: true,
      jointRounded: true,
    }));
  }

  // 4. Gate line
  if (state.layers.gates && HAS_GATE && GATE_COORDS) {
    layers.push(new deck.LineLayer({
      id: 'gate',
      data: [{sourcePosition: GATE_COORDS[0], targetPosition: GATE_COORDS[1]}],
      getSourcePosition: d => d.sourcePosition,
      getTargetPosition: d => d.targetPosition,
      getColor: [0, 200, 255, 200],
      widthMinPixels: 3,
    }));
  }

  // 5. Infrastructure
  if (state.layers.infrastructure && INFRA_DATA.length > 0) {
    layers.push(new deck.ScatterplotLayer({
      id: 'infra',
      data: INFRA_DATA,
      getPosition: d => [d.lon, d.lat],
      getFillColor: [255, 200, 50, 200],
      getLineColor: [255, 255, 255, 150],
      radiusMinPixels: 6,
      radiusMaxPixels: 10,
      lineWidthMinPixels: 1,
      stroked: true,
      pickable: true,
    }));
  }

  // 6. Density heatmap
  if (state.layers.density && DENSITY_DATA.length > 0) {
    layers.push(new deck.HeatmapLayer({
      id: 'density',
      data: DENSITY_DATA,
      getPosition: d => [d.center_lon, d.center_lat],
      getWeight: d => d.count,
      radiusPixels: 40,
      intensity: 1,
      threshold: 0.05,
      colorRange: [
        [255, 255, 178], [254, 204, 92], [253, 141, 60],
        [240, 59, 32], [189, 0, 38],
      ],
    }));
  }

  deckgl.setProps({layers});
}
""")

_JS_CHART = """\
// ── Chart (Canvas2D) ───────────────────────────
const chartCanvas = document.getElementById('chart-canvas');
const chartCtx = chartCanvas ? chartCanvas.getContext('2d') : null;

function drawChart() {
  if (!chartCtx || !HAS_GATE) return;
  const dc = ANALYTICS.daily_crossings;
  if (!dc || dc.length < 2) return;

  const rect = chartCanvas.parentElement.getBoundingClientRect();
  chartCanvas.width = rect.width * 2;
  chartCanvas.height = rect.height * 2;
  const ctx = chartCtx;
  ctx.scale(2, 2);
  const W = rect.width, H = rect.height;

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(1, ...dc.map(d => Math.max(d.inbound, d.outbound)));
  const pad = {l: 5, r: 5, t: 8, b: 16};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  function drawLine(key, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < dc.length; i++) {
      const x = pad.l + (i / (dc.length - 1)) * cw;
      const y = pad.t + ch - (dc[i][key] / maxVal) * ch;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawLine('inbound', '#32ff82');
  drawLine('outbound', '#ff6432');

  // "Now" indicator
  if (MAX_TIME > 0) {
    const frac = state.currentTime / MAX_TIME;
    const x = pad.l + frac * cw;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0,200,255,0.6)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, pad.t + ch);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Date labels
  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(dc[0].date, pad.l, H - 2);
  ctx.textAlign = 'right';
  ctx.fillText(dc[dc.length - 1].date, W - pad.r, H - 2);
}
"""

_JS_SPARKLINE = """\
// ── Sparkline ──────────────────────────────────
const sparkCanvas = document.getElementById('sparkline-canvas');
const sparkCtx = sparkCanvas ? sparkCanvas.getContext('2d') : null;

function drawSparkline() {
  if (!sparkCtx) return;
  const sp = ANALYTICS.sparkline;
  if (!sp || sp.length < 2) return;

  const rect = sparkCanvas.parentElement.getBoundingClientRect();
  sparkCanvas.width = rect.width * 2;
  sparkCanvas.height = rect.height * 2;
  const ctx = sparkCtx;
  ctx.scale(2, 2);
  const W = rect.width, H = rect.height;

  ctx.clearRect(0, 0, W, H);

  const maxVal = Math.max(1, ...sp);
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(0,200,255,0.5)';
  ctx.lineWidth = 1;
  for (let i = 0; i < sp.length; i++) {
    const x = (i / (sp.length - 1)) * W;
    const y = H - (sp[i] / maxVal) * (H - 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // "Now" indicator
  if (MAX_TIME > 0) {
    const frac = state.currentTime / MAX_TIME;
    const x = frac * W;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0,200,255,0.8)';
    ctx.lineWidth = 1;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
}
"""

_JS_CONTROLS = """\
// ── Playback & Controls ────────────────────────
let lastFrame = performance.now();

function fmtDate(timeS) {
  const ms = GLOBAL_START_MS + timeS * 1000;
  const d = new Date(ms);
  return d.toISOString().replace('T', ' ').replace(/\\.\\d+Z/, ' UTC');
}

function fmtShortDate(timeS) {
  const ms = GLOBAL_START_MS + timeS * 1000;
  const d = new Date(ms);
  const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
  return months[d.getUTCMonth()] + ' ' + String(d.getUTCDate()).padStart(2,'0') + ', ' + d.getUTCFullYear();
}

function currentDayStr() {
  const ms = GLOBAL_START_MS + state.currentTime * 1000;
  const d = new Date(ms);
  return d.toISOString().slice(0, 10);
}

// Update live counter
function updateLiveCounter() {
  if (!HAS_GATE) return;
  const day = currentDayStr();
  const dc = ANALYTICS.daily_crossings;
  const rec = dc.find(d => d.date === day);
  document.getElementById('live-count').textContent = rec ? rec.unique_vessels : 0;
  document.getElementById('live-inbound').textContent = rec ? rec.inbound : 0;
  document.getElementById('live-outbound').textContent = rec ? rec.outbound : 0;
  document.getElementById('current-date').textContent = fmtShortDate(state.currentTime);

  // Reversals for current day
  const dayStart = new Date(day + 'T00:00:00Z').getTime();
  const dayEnd = dayStart + 86400000;
  const epochS = GLOBAL_START_MS / 1000;
  let revCount = 0;
  for (const r of ANALYTICS.reversals) {
    const rMs = (epochS + r.reversal_s) * 1000;
    if (rMs >= dayStart && rMs < dayEnd) revCount++;
  }
  document.getElementById('live-reversals').textContent = revCount + ' same-hull reversals today';
}

function countActive() {
  const ct = state.currentTime;
  let n = 0;
  for (const t of getFilteredTrips()) {
    const ts = t.timestamps;
    if (ts[0] <= ct && ct <= ts[ts.length - 1]) n++;
  }
  return n;
}

function animate(now) {
  if (state.playing) {
    const dt = (now - lastFrame) / 1000;
    state.currentTime += dt * state.speed;
    if (state.currentTime > MAX_TIME) state.currentTime = 0;
    updateLayers();
  }
  lastFrame = now;

  // Update UI (throttled to avoid excessive redraws)
  const slider = document.getElementById('timeline-slider');
  slider.value = (state.currentTime / MAX_TIME * 10000) | 0;
  document.getElementById('clock').textContent = fmtDate(state.currentTime);
  document.getElementById('active-count').textContent = countActive() + ' active vessels';
  updateLiveCounter();

  // Redraw sparkline "now" indicator (lightweight)
  drawSparkline();
  drawChart();

  requestAnimationFrame(animate);
}

// ── Init controls ──────────────────────────────

// Play/pause
document.getElementById('play-btn').onclick = () => {
  state.playing = !state.playing;
  const btn = document.getElementById('play-btn');
  btn.innerHTML = state.playing ? '&#9646;&#9646;' : '&#9654;';
  btn.classList.toggle('active', state.playing);
};

// Speed buttons
const speedRow = document.getElementById('speed-btns');
SPEEDS.forEach((s, i) => {
  const b = document.createElement('button');
  b.className = 'btn' + (s.value === state.speed ? ' active' : '');
  b.textContent = s.label;
  b.onclick = () => {
    state.speed = s.value;
    speedRow.querySelectorAll('.btn').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
  };
  speedRow.appendChild(b);
});

// Timeline slider
document.getElementById('timeline-slider').oninput = (e) => {
  state.currentTime = (e.target.value / 10000) * MAX_TIME;
  updateLayers();
};

// Zoom controls
document.getElementById('zoom-in').onclick = () => {
  const vs = deckgl.getViewports()[0];
  deckgl.setProps({initialViewState: {...vs, zoom: vs.zoom + 1, transitionDuration: 300}});
};
document.getElementById('zoom-out').onclick = () => {
  const vs = deckgl.getViewports()[0];
  deckgl.setProps({initialViewState: {...vs, zoom: vs.zoom - 1, transitionDuration: 300}});
};
document.getElementById('reset-view').onclick = () => {
  deckgl.setProps({initialViewState: {...INITIAL_VIEW, transitionDuration: 500}});
};

// Mode tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.mode = tab.dataset.mode;
    drawChart();
  };
});
"""

_JS_FILTERS = """\
// ── Filter UI ──────────────────────────────────

// Period buttons
document.querySelectorAll('#period-btns .btn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('#period-btns .btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.filters.period = btn.dataset.period;
    applyFilters();
  };
});

// Transit toggle
document.querySelectorAll('#transit-btns .btn').forEach(btn => {
  btn.onclick = () => {
    btn.classList.toggle('active');
    state.filters.transitOnly = btn.classList.contains('active');
    applyFilters();
  };
});

// Search
document.getElementById('search-input').oninput = (e) => {
  state.filters.search = e.target.value;
  applyFilters();
};

// Flag pills
const flagRow = document.getElementById('flag-btns');
ANALYTICS.flag_counts.forEach(fc => {
  const b = document.createElement('button');
  b.className = 'btn';
  b.textContent = fc.flag;
  b.onclick = () => {
    if (state.filters.flags.has(fc.flag)) {
      state.filters.flags.delete(fc.flag);
      b.classList.remove('active');
    } else {
      state.filters.flags.add(fc.flag);
      b.classList.add('active');
    }
    applyFilters();
  };
  flagRow.appendChild(b);
});

// Type pills
const typeRow = document.getElementById('type-btns');
ANALYTICS.type_counts.forEach(tc => {
  const b = document.createElement('button');
  b.className = 'btn';
  b.textContent = tc.type;
  b.onclick = () => {
    if (state.filters.shipTypes.has(tc.type)) {
      state.filters.shipTypes.delete(tc.type);
      b.classList.remove('active');
    } else {
      state.filters.shipTypes.add(tc.type);
      b.classList.add('active');
    }
    applyFilters();
  };
  typeRow.appendChild(b);
});

// Layer toggles
document.querySelectorAll('#layer-toggles input').forEach(cb => {
  cb.onchange = () => {
    state.layers[cb.dataset.layer] = cb.checked;
    updateLayers();
  };
});
"""

_JS_VESSEL_DETAIL = """\
// ── Vessel selection ───────────────────────────
function selectVessel(mmsi) {
  state.selectedMmsi = mmsi;
  const info = VESSEL_INDEX[String(mmsi)] || {};
  const card = document.getElementById('vessel-card');
  card.classList.add('visible');
  document.getElementById('vc-name').textContent =
    (info.flag ? info.flag + ' ' : '') + (info.name || 'MMSI ' + mmsi);
  document.getElementById('vc-type').textContent = info.type || 'Unknown type';

  const details = document.getElementById('vc-details');
  details.innerHTML = '';
  const rows = [
    ['MMSI', mmsi],
    ['IMO', info.imo || '—'],
    ['Flag', info.flag || '—'],
    ['Length', info.length ? info.length + ' m' : '—'],
    ['Beam', info.beam ? info.beam + ' m' : '—'],
  ];
  rows.forEach(([label, val]) => {
    const row = document.createElement('div');
    row.className = 'v-row';
    row.innerHTML = '<span class="v-label">' + label + '</span><span>' + val + '</span>';
    details.appendChild(row);
  });
  updateLayers();
}

function deselectVessel() {
  state.selectedMmsi = null;
  document.getElementById('vessel-card').classList.remove('visible');
  updateLayers();
}
"""

_JS_KEYBOARD = """\
// ── Keyboard shortcuts ─────────────────────────
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.code) {
    case 'Space':
      e.preventDefault();
      document.getElementById('play-btn').click();
      break;
    case 'ArrowRight':
      state.currentTime = Math.min(state.currentTime + MAX_TIME / 200, MAX_TIME);
      updateLayers();
      break;
    case 'ArrowLeft':
      state.currentTime = Math.max(state.currentTime - MAX_TIME / 200, 0);
      updateLayers();
      break;
    case 'Escape':
      deselectVessel();
      break;
    case 'Digit1': state.speed = SPEEDS[0].value; break;
    case 'Digit2': state.speed = SPEEDS[1].value; break;
    case 'Digit3': state.speed = SPEEDS[2].value; break;
  }
});
"""

_JS_INIT = Template("""\
// ── Init ───────────────────────────────────────
(function init() {
  // Populate summary stats
  const S = ANALYTICS.summary;
  document.getElementById('stat-tracked').textContent = S.total_tracked;
  document.getElementById('stat-transit').textContent = S.total_transit;
  document.getElementById('stat-crossings').textContent = S.total_crossings;
  document.getElementById('stat-avg').textContent = S.avg_per_day;
  if (S.delta_pct != null) {
    const el = document.getElementById('stat-delta');
    el.textContent = (S.delta_pct > 0 ? '+' : '') + S.delta_pct + '%';
    el.className = 'metric-val ' + (S.delta_pct < 0 ? 'red' : 'green');
  }

  // Subset note
  if ($showing_subset) {
    const note = document.getElementById('subset-note');
    note.style.display = 'block';
    note.textContent = 'Showing ' + $n_tracks + ' of ' + $total_track_count + ' tracks';
  }

  drawChart();
  drawSparkline();
  updateLayers();
  requestAnimationFrame(animate);
})();
""")

_CLOSE = """\
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


def render_dashboard(data: dict) -> str:
    """Assemble template sections into a complete HTML dashboard.

    Args:
        data: Dict of template variables produced by
            :func:`viz.generate_dashboard`.

    Returns:
        Complete HTML string.
    """
    body_class = "" if data["has_gate"] else "no-gate"
    head_data = {**data, "body_class": body_class}

    # For JS templates that use $-substitution, we need both
    # Template-safe and raw-embed values.
    js_data = {
        "trips_json": data["trips_json"],
        "analytics_json": data["analytics_json"],
        "vessel_index_json": data["vessel_index_json"],
        "density_json": data["density_json"],
        "events_json": data["events_json"],
        "infra_json": data["infra_json"],
        "gate_coords": data["gate_coords"],
        "has_gate": "true" if data["has_gate"] else "false",
        "max_time": data["max_time"],
        "trail_length": data["trail_length"],
        "global_start_ms": data["global_start_ms"],
        "event_date": data.get("event_date", ""),
        "date_from": data.get("date_from", ""),
        "date_to": data.get("date_to", ""),
    }

    js_state_data = {
        "default_speed": data["default_speed"],
    }

    js_layers_data = {
        "center_lon": data["center_lon"],
        "center_lat": data["center_lat"],
        "zoom": data["zoom"],
        "pitch": data["pitch"],
        "bearing": data["bearing"],
    }

    js_init_data = {
        "showing_subset": "true" if data["showing_subset"] else "false",
        "n_tracks": data["n_tracks"],
        "total_track_count": data["total_track_count"],
    }

    parts = [
        _HEAD.safe_substitute(head_data),
        _LEFT_PANEL.safe_substitute(data),
        _TOP_CENTER.safe_substitute(data),
        _RIGHT_PANEL.safe_substitute(data),
        _BOTTOM_BAR.safe_substitute(data),
        _JS_DATA.safe_substitute(js_data),
        _JS_STATE.safe_substitute(js_state_data),
        _JS_LAYERS.safe_substitute(js_layers_data),
        _JS_CHART,
        _JS_SPARKLINE,
        _JS_CONTROLS,
        _JS_FILTERS,
        _JS_VESSEL_DETAIL,
        _JS_KEYBOARD,
        _JS_INIT.safe_substitute(js_init_data),
        _CLOSE,
    ]

    return "\n".join(parts)
