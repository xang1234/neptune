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
  background: #060708; color: #dfe7ec; overflow: hidden;
}
#map { position: absolute; top: 0; left: 0; right: 0; bottom: 0; }

/* ── panels ─────────────────────────────────── */
.panel {
  position: absolute; z-index: 10;
  background: rgba(8, 10, 14, 0.92);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: 8px; padding: 16px;
  backdrop-filter: blur(14px);
  box-shadow: 0 6px 30px rgba(0,0,0,0.6);
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
  background: rgba(6, 8, 10, 0.96);
  border: none; border-top: 1px solid rgba(255,255,255,0.08);
}

/* ── typography ─────────────────────────────── */
h1 { font-size: 17px; font-weight: 700; letter-spacing: 2px;
     text-transform: uppercase; margin-bottom: 4px; }
h2 { font-size: 9px; font-weight: 600; text-transform: uppercase;
     letter-spacing: 1.5px; color: rgba(255,255,255,0.35); margin-bottom: 8px; }
.desc { font-size: 12px; color: rgba(255,255,255,0.55); line-height: 1.5;
        margin-bottom: 12px; font-family: -apple-system, sans-serif; }
.big-num { font-size: 48px; font-weight: 700; line-height: 1;
           color: #fff; font-variant-numeric: tabular-nums; }
.metric { text-align: center; flex: 1;
          border: 1px solid rgba(255,255,255,0.08); border-radius: 6px;
          padding: 8px 6px; background: rgba(255,255,255,0.02); }
.metric-val { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.metric-label { font-size: 8px; text-transform: uppercase; letter-spacing: 1px;
                color: rgba(255,255,255,0.4); margin-top: 3px; }
.metrics-row { display: flex; gap: 8px; margin: 10px 0; }
.cyan { color: #43d3ff; }
.orange { color: #ffa03c; }
.green { color: #32ff82; }
.red { color: #ff6464; }

/* ── buttons ────────────────────────────────── */
.btn {
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.16);
  color: rgba(255,255,255,0.6); padding: 5px 10px; border-radius: 4px;
  cursor: pointer; font-size: 10px; font-family: inherit;
  text-transform: uppercase; letter-spacing: 0.8px; transition: all 0.15s;
}
.btn:hover { border-color: rgba(255,255,255,0.4); color: #fff; }
.btn.active { background: rgba(67,211,255,0.10); border-color: rgba(67,211,255,0.7);
              color: #43d3ff; }
.btn-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }

/* ── chart canvas ───────────────────────────── */
.chart-container { position: relative; width: 100%; height: 100px;
                   margin: 8px 0; }
.chart-container canvas { width: 100%; height: 100%; }

/* ── mode tabs ──────────────────────────────── */
.tabs { display: flex; gap: 0; margin: 10px 0; border-radius: 6px; overflow: hidden; }
.tab {
  flex: 1; padding: 6px 8px; text-align: center; font-size: 9px;
  text-transform: uppercase; letter-spacing: 0.8px; cursor: pointer;
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.12);
  color: rgba(255,255,255,0.45); transition: all 0.15s;
}
.tab:first-child { border-radius: 4px 0 0 4px; }
.tab:last-child { border-radius: 0 4px 4px 0; }
.tab.active { background: rgba(67,211,255,0.10); border-color: rgba(67,211,255,0.7);
              color: #43d3ff; }

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
#search-input:focus { border-color: rgba(67,211,255,0.5); }

/* ── timeline ───────────────────────────────── */
#timeline-slider {
  flex: 1; -webkit-appearance: none; appearance: none; height: 3px;
  background: rgba(255,255,255,0.15); border-radius: 2px; outline: none;
}
#timeline-slider::-webkit-slider-thumb {
  -webkit-appearance: none; appearance: none; width: 12px; height: 12px;
  border-radius: 50%; background: #43d3ff; cursor: pointer;
  box-shadow: 0 0 10px rgba(67,211,255,0.9);
}
#timeline-slider::-moz-range-thumb {
  width: 12px; height: 12px; border: none; border-radius: 50%;
  background: #43d3ff; cursor: pointer;
}
#clock { font-variant-numeric: tabular-nums; font-size: 13px;
         color: #ffa03c; font-weight: 600; min-width: 170px; text-align: right; }
#active-count { font-size: 10px; color: rgba(255,255,255,0.4); min-width: 140px;
                text-transform: uppercase; letter-spacing: 0.5px; }
.sparkline-container { width: 100%; height: 36px; position: relative; }
.sparkline-container canvas { width: 100%; height: 100%; }

/* ── playback badge ─────────────────────────── */
#playback-badge {
  display: none; font-size: 9px; text-transform: uppercase;
  letter-spacing: 1.5px; color: #43d3ff; margin-left: 8px;
  vertical-align: middle;
}
#playback-badge.visible { display: inline; }
#playback-badge::before {
  content: ''; display: inline-block; width: 6px; height: 6px;
  border-radius: 50%; background: #43d3ff; margin-right: 5px;
  vertical-align: middle; animation: pulse-dot 1.5s infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── timeline dates ────────────────────────────── */
.timeline-date { font-size: 10px; color: rgba(255,255,255,0.35);
                 font-variant-numeric: tabular-nums; white-space: nowrap; }

/* ── position counts ───────────────────────────── */
.pos-counts { display: flex; gap: 12px; margin-bottom: 10px;
              padding-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.pos-count-val { font-size: 14px; font-weight: 700; font-variant-numeric: tabular-nums; }
.pos-count-label { font-size: 9px; text-transform: uppercase;
                   color: rgba(255,255,255,0.4); letter-spacing: 0.3px; }

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
  <h1>$title <span id="playback-badge">Playback</span></h1>
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
  <div style="display:flex;justify-content:space-between;align-items:baseline;gap:16px">
    <span style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:rgba(255,255,255,0.35)">Live UTC<br>Day</span>
    <span style="font-size:13px;font-weight:700;letter-spacing:1px" id="current-date">—</span>
  </div>
  <div class="big-num" style="text-align:left" id="live-count">0</div>
  <div style="font-size:10px;letter-spacing:1px;text-transform:uppercase;color:rgba(255,255,255,0.45);margin:4px 0;text-align:left">Unique Vessels Crossed</div>
  <div class="metrics-row">
    <div class="metric" style="text-align:left;padding:8px 10px">
      <div class="metric-label" style="margin:0 0 4px"><span class="cyan">&uarr;</span> Inbound</div>
      <div class="metric-val" id="live-inbound">0</div>
    </div>
    <div class="metric" style="text-align:left;padding:8px 10px">
      <div class="metric-label" style="margin:0 0 4px"><span class="orange">&darr;</span> Outbound</div>
      <div class="metric-val" id="live-outbound">0</div>
    </div>
  </div>
  <div style="font-size:10px;letter-spacing:0.5px;text-transform:uppercase;color:rgba(255,255,255,0.4);text-align:left" id="live-reversals">0 same-hull reversals today</div>
</div>
""")

_RIGHT_PANEL = Template("""\
<div id="right-panel" class="panel">
  <div class="pos-counts gate-only">
    <div class="metric">
      <div class="pos-count-val cyan" id="pos-total">—</div>
      <div class="pos-count-label">Total Positions</div>
    </div>
    <div class="metric">
      <div class="pos-count-val" id="pos-before">—</div>
      <div class="pos-count-label">Before</div>
    </div>
    <div class="metric">
      <div class="pos-count-val" id="pos-after">—</div>
      <div class="pos-count-label">After</div>
    </div>
  </div>

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
  <div class="btn-row" id="layer-toggles">
    <button class="btn active" data-layer="trips">Trips</button>
    <button class="btn active" data-layer="heads">Heads</button>
    <button class="btn" data-layer="tracks">Tracks</button>
    <button class="btn active gate-only" data-layer="gates">Gates</button>
    <button class="btn" data-layer="density">Density</button>
    <button class="btn active" data-layer="infrastructure">Infrastructure</button>
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
  <button class="btn" id="play-btn" title="Play / Pause (Space)">Play</button>
  <div class="btn-row" id="speed-btns" style="margin:0"></div>
  <span class="timeline-date" id="date-start"></span>
  <input type="range" id="timeline-slider" min="0" max="10000" value="0">
  <span class="timeline-date" id="date-end"></span>
  <span id="clock">—</span>
  <span id="active-count"></span>
</div>
<div class="panel" style="bottom:53px;left:0;right:0;padding:4px 0;border-radius:0;border:none;border-top:1px solid rgba(255,255,255,0.06);background:rgba(6,8,10,0.96);">
  <div class="sparkline-container" style="position:relative">
    <canvas id="sparkline-canvas"></canvas>
    <canvas id="event-dots-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;"></canvas>
  </div>
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
const CROSSING_TIMES = $crossing_times_json;
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
  {value: 1800, label: '30M/S'},
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

// Uniform palette: amber long-exposure trails, ice-blue live heads.
const TRAIL_COLOR = [255, 168, 60];
const HEAD_COLOR = [205, 235, 255];
const SELECTED_COLOR = [80, 220, 255];

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

  // 1a. Trail glow layer (warm amber bloom underneath)
  if (state.layers.trips) {
    layers.push(new deck.TripsLayer({
      id: 'trips-glow',
      data: filtered,
      getPath: d => d.path,
      getTimestamps: d => d.timestamps,
      getColor: [255, 140, 20],
      currentTime: ct,
      trailLength: TRAIL_LENGTH,
      widthMinPixels: 8,
      widthMaxPixels: 20,
      capRounded: true,
      jointRounded: true,
      opacity: 0.15,
    }));
  }

  // 1b. Trips layer (uniform amber "long-exposure" corridors; selection pops cyan)
  if (state.layers.trips) {
    layers.push(new deck.TripsLayer({
      id: 'trips',
      data: filtered,
      getPath: d => d.path,
      getTimestamps: d => d.timestamps,
      getColor: d => {
        if (state.selectedMmsi != null) {
          return d.mmsi === state.selectedMmsi ? SELECTED_COLOR : [...TRAIL_COLOR, 50];
        }
        return TRAIL_COLOR;
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
        if (state.selectedMmsi != null) {
          return d.mmsi === state.selectedMmsi
            ? [...SELECTED_COLOR, 255] : [...HEAD_COLOR, 60];
        }
        return [...HEAD_COLOR, 255];
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
            ? [...SELECTED_COLOR, 200] : [...TRAIL_COLOR, 25];
        }
        return [...TRAIL_COLOR, 90];
      },
      widthMinPixels: 1,
      widthMaxPixels: 3,
      capRounded: true,
      jointRounded: true,
    }));
  }

  // 4. Gate line + crossing pulse
  if (state.layers.gates && HAS_GATE && GATE_COORDS) {
    layers.push(new deck.LineLayer({
      id: 'gate',
      data: [{sourcePosition: GATE_COORDS[0], targetPosition: GATE_COORDS[1]}],
      getSourcePosition: d => d.sourcePosition,
      getTargetPosition: d => d.targetPosition,
      getColor: [0, 200, 255, 200],
      widthMinPixels: 3,
    }));
    // Pulsing ring at gate midpoint when crossings happen nearby in time
    const gateMid = [(GATE_COORDS[0][0]+GATE_COORDS[1][0])/2, (GATE_COORDS[0][1]+GATE_COORDS[1][1])/2];
    let recentCrossings = 0;
    for (const t of CROSSING_TIMES) {
      if (Math.abs(t - ct) < 300) recentCrossings++;
    }
    if (recentCrossings > 0) {
      const pulseR = 8 + 6 * Math.sin(ct * 3);
      layers.push(new deck.ScatterplotLayer({
        id: 'gate-pulse',
        data: [{position: gateMid}],
        getPosition: d => d.position,
        getFillColor: [0, 200, 255, 0],
        getLineColor: [0, 200, 255, Math.min(200, recentCrossings * 40)],
        radiusMinPixels: Math.max(4, pulseR),
        radiusMaxPixels: 30,
        lineWidthMinPixels: 2,
        stroked: true,
        filled: false,
      }));
    }
  }

  // 5. Infrastructure
  if (state.layers.infrastructure && INFRA_DATA.length > 0) {
    layers.push(new deck.ScatterplotLayer({
      id: 'infra',
      data: INFRA_DATA,
      getPosition: d => [d.lon, d.lat],
      getFillColor: [255, 200, 50, 190],
      getLineColor: [255, 255, 255, 120],
      radiusMinPixels: 4,
      radiusMaxPixels: 7,
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

  drawLine('inbound', '#43d3ff');
  drawLine('outbound', '#ffa03c');

  // "Now" indicator
  if (MAX_TIME > 0) {
    const frac = state.currentTime / MAX_TIME;
    const x = pad.l + frac * cw;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(67,211,255,0.6)';
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
// ── Sparkline (dual inbound/outbound lines) ────
const sparkCanvas = document.getElementById('sparkline-canvas');
const sparkCtx = sparkCanvas ? sparkCanvas.getContext('2d') : null;
const dotsCanvas = document.getElementById('event-dots-canvas');
const dotsCtx = dotsCanvas ? dotsCanvas.getContext('2d') : null;

function drawSparkline() {
  if (!sparkCtx) return;
  const spIn = ANALYTICS.sparkline_inbound;
  const spOut = ANALYTICS.sparkline_outbound;
  const sp = ANALYTICS.sparkline;
  const hasDual = spIn && spOut && spIn.length >= 2 && spOut.length >= 2;
  if (!sp || sp.length < 2) return;

  const rect = sparkCanvas.parentElement.getBoundingClientRect();
  sparkCanvas.width = rect.width * 2;
  sparkCanvas.height = rect.height * 2;
  const ctx = sparkCtx;
  ctx.scale(2, 2);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  // Rolling mean so the band reads as flowing curves, not a spike field
  function smooth(data, w) {
    const out = new Array(data.length);
    for (let i = 0; i < data.length; i++) {
      let s = 0, n = 0;
      for (let j = Math.max(0, i - w); j <= Math.min(data.length - 1, i + w); j++) {
        s += data[j]; n++;
      }
      out[i] = s / n;
    }
    return out;
  }

  function drawLine(data, color, fill) {
    const maxVal = Math.max(1, ...data);
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * W;
      const y = H - (data[i] / maxVal) * (H - 4) - 1;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    if (fill) {
      ctx.lineTo(W, H);
      ctx.lineTo(0, H);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
  }

  if (hasDual) {
    drawLine(smooth(spIn, 4), 'rgba(67,211,255,0.8)', 'rgba(67,211,255,0.08)');
    drawLine(smooth(spOut, 4), 'rgba(255,160,60,0.8)', 'rgba(255,160,60,0.08)');
  } else {
    drawLine(smooth(sp, 4), 'rgba(67,211,255,0.6)', 'rgba(67,211,255,0.08)');
  }

  // "Now" indicator
  if (MAX_TIME > 0) {
    const frac = state.currentTime / MAX_TIME;
    const x = frac * W;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(67,211,255,0.8)';
    ctx.lineWidth = 1;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }

  // Event dots on overlay canvas (#2)
  drawEventDots(W, H);
}

function drawEventDots(W, H) {
  if (!dotsCtx || !HAS_GATE || MAX_TIME <= 0) return;
  const rect = dotsCanvas.parentElement.getBoundingClientRect();
  dotsCanvas.width = rect.width * 2;
  dotsCanvas.height = rect.height * 2;
  const ctx = dotsCtx;
  ctx.scale(2, 2);
  ctx.clearRect(0, 0, W, H);

  const dc = ANALYTICS.daily_crossings;
  if (!dc || dc.length === 0) return;

  // One dot per day, colored by balance
  for (const day of dc) {
    const dayDate = new Date(day.date + 'T12:00:00Z');
    const dayS = (dayDate.getTime() - GLOBAL_START_MS) / 1000;
    const frac = dayS / MAX_TIME;
    if (frac < 0 || frac > 1) continue;
    const x = frac * W;
    const total = day.inbound + day.outbound;
    const r = Math.min(5, 2 + total * 0.15);
    // Yellow = balanced flow, red = strongly one-directional
    const inRatio = total > 0 ? day.inbound / total : 0.5;
    const skew = Math.abs(inRatio - 0.5) * 2;
    ctx.beginPath();
    ctx.arc(x, H / 2, r, 0, Math.PI * 2);
    ctx.fillStyle = skew > 0.5 ? 'rgba(255,80,80,0.85)' : 'rgba(255,210,60,0.85)';
    ctx.fill();
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
  const nActive = countActive();
  const posTotal = ANALYTICS.summary.total_positions;
  document.getElementById('active-count').textContent =
    nActive + ' active vessels' + (posTotal ? ' \u00b7 ' + posTotal.toLocaleString() + ' positions' : '');
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
  btn.textContent = state.playing ? 'Pause' : 'Play';
  btn.classList.toggle('active', state.playing);
  document.getElementById('playback-badge').classList.toggle('visible', state.playing);
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

// Layer toggles (chip buttons)
document.querySelectorAll('#layer-toggles .btn').forEach(btn => {
  btn.onclick = () => {
    btn.classList.toggle('active');
    state.layers[btn.dataset.layer] = btn.classList.contains('active');
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
    ['IMO', info.imo],
    ['Callsign', info.callsign],
    ['Flag', info.flag],
    ['Destination', info.destination],
    ['Length', info.length ? info.length + ' m' : null],
    ['Beam', info.beam ? info.beam + ' m' : null],
    ['Draught', info.draught ? info.draught + ' m' : null],
  ];
  rows.forEach(([label, val]) => {
    if (!val) return;
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

  // Position counts (#5)
  if (S.total_positions > 0) {
    document.getElementById('pos-total').textContent = S.total_positions.toLocaleString();
    document.getElementById('pos-before').textContent = S.positions_before ? S.positions_before.toLocaleString() : '—';
    document.getElementById('pos-after').textContent = S.positions_after ? S.positions_after.toLocaleString() : '—';
  }

  // Date range labels (#7)
  if (DATE_FROM) document.getElementById('date-start').textContent = DATE_FROM;
  if (DATE_TO) document.getElementById('date-end').textContent = DATE_TO;

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
        "crossing_times_json": data["crossing_times_json"],
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
