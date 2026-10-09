# AIS visualization review: 30 candidates, five recommendations

Reviewed 2026-10-09 against the current working tree. This is a product and visualization design review; it does not implement the proposals.

Neptune's strongest next step is to make its attractive vessel maps explainable and analytically consistent. Preserve the cinematic timelapses, while making it easier to select a pattern, identify its vessels, understand its events, and assess what the observations support.

## Evidence and scope

Reviewed `viz.py`, all three visualization templates, track/crossing/density/port derivation, position and event schemas, `HEURISTICS.md`, notebooks 08–11, the existing replay HTML, and frames extracted from the dashboard, corridor, and Florida-crossing GIFs. Loaded the existing D3 corridor HTML and exercised a dashboard generated from current source. That dashboard used existing replay geometry with **synthetic vessel IDs, types, and date anchor** solely to inspect UI behavior; its counts are not real-world findings.

The collaborative browser could navigate and evaluate page JavaScript, but its screenshot and click tools failed repeatedly. Visual assessment therefore uses the repository's rendered assets; interaction checks used the same browser's DOM evaluation. No end-user study, large-data benchmark, or full test-suite run was performed.

Important existing strengths and gaps:

- Replay, vessel selection, metadata filters, density, crossing charts, playback, and multi-panel timelapses already exist. Recommending those capabilities from scratch would miss the actual opportunity.
- The dashboard's amber paths and white directional heads create a strong visual identity. Dense arrows, road labels, and peripheral panels compete with the traffic in its rendered example. The D3 corridors are striking, but saturated white areas hide categorical distinctions and brightness has no quantitative legend.
- `getVisibleIndices()` filters vessel metadata; charts and daily counters still read fixed `ANALYTICS`, and density still reads fixed `DENSITY_DATA`. In the inspection fixture, a cargo selection reduced nine tracks to four while the crossing total remained ten.
- The Before/After buttons set `state.filters.period`, but the filter logic does not consume it. The fixture retained tracks extending beyond the Before cutoff. The three mode tabs set `state.mode`, which the chart implementation does not use.
- `generate_dashboard()` samples tracks before computing crossings. Its subset notice is useful, but analytical totals can still reflect a rendering sample. Density independently samples up to 50,000 position rows.
- `prepare_density()` counts reports in H3 cells, then the dashboard renders a heatmap of cell centers. `derive/density.py` separately calculates unique vessels and observation counts on a rounded latitude/longitude grid. These are different products and geometries, despite similar names.
- The dashboard defaults to density resolution 4, which `prepare_density()` documents as an overview scale. The local-port inspection fixture collapsed 6,339 reports into one density cell. That is evidence of a poor default for this fixture, not a benchmark of all datasets.
- `EVENT_DATA` is embedded but otherwise unused by the dashboard. Its timeline dots summarize daily crossing balance rather than the supplied maritime events. The existing vessel card is mainly identity and dimensions.
- Before/after percentages use raw counts. Average crossings per day divides by dates containing crossings; the seven-day calculation advances over those records. Directional sparkline series are each scaled to their own maximum. These choices can distort comparisons.
- Track derivation already splits at gaps and jumps, and the D3 preparation has a gap threshold. The opportunity is to expose these decisions and their meaning, not to claim gaps are currently always connected.
- Neptune already stores timestamps, source/provenance, QC fields, heuristic event confidence, and port boundary metadata. Much useful information is lost in the compact trip payload.

Code anchors: `neptune_ais/viz.py` (`prepare_density`, `_build_trips`, `_compute_dashboard_analytics`, `generate_dashboard`, `prepare_timelapse_tracks`, `prepare_ports`); `neptune_ais/_dashboard_template.py` (`getVisibleIndices`, `drawChart`, `drawSparkline`, `selectVessel`); `neptune_ais/derive/density.py`; `neptune_ais/derive/tracks.py`; `neptune_ais/datasets/events.py`; `HEURISTICS.md`.

## Selection criteria

Assumed primary use: Python analysts exploring archival AIS and sharing standalone visualizations. The ranking would change for a broadcast-only portfolio or a live navigation product.

Each candidate was assessed for the question it answers, likely user interpretation, implementation fit, data requirements, failure modes, and benefit relative to effort. The judgments below are design assessments, not measured user outcomes. Small/medium/large describe relative scope, not delivery estimates.

## The 30 candidates

| # | Idea and likely user value | Implementation and principal tradeoff | Decision |
|---|---|---|---|
| 1 | **Coordinated map, time range, charts, and vessel list.** Selecting an interesting interval or vessel class yields one understandable subset everywhere. | Shared selection state and common aggregation semantics in the existing dashboard; medium scope. Distinguish time cursor, selected interval, and spatial filter so ordinary panning does not unexpectedly change totals. | **Winner: rank 1.** Broad benefit and directly fixes observed inconsistency. |
| 2 | **A traffic atlas that changes detail with zoom.** Overview density resolves into corridors and individual vessels; users can understand both busy regions and local behavior. | Multi-resolution spatial aggregates, explicit measures, and density-to-track transitions; medium–large scope. Unique-vessel counts are not additive across time or cells, and visual sampling must not determine analytical totals. | **Winner: rank 2.** Largest improvement to both legibility and quantitative meaning. |
| 3 | **Vessel journey inspector.** Selecting a ship reveals its speed history, stops, events, and gaps along one timeline. | Extend the existing detail card, retain observation timestamps and event IDs, and link chart hover to map time; medium scope. Event labels must remain heuristic, and identity/destination metadata must respect observation time. | **Winner: rank 3.** Makes existing Neptune analytics visible and inspectable. |
| 4 | **Synchronized before/after comparison and difference maps.** Users can locate and quantify a change without remembering a previous animation. | Shared camera, cohort, scales, comparable windows, and coverage-aware aggregates; medium–large scope. Unequal duration, missing data, and zero baselines can create false changes. | **Winner: rank 4.** High analytical payoff, but requires representative historical data and careful denominators. |
| 5 | **Visible data gaps, observation support, and provenance.** Users can tell what was seen from what was interpolated or inferred. | Preserve QC and source metadata, annotate track breaks, and expose evidence details; medium scope. Too many warning colors overwhelm the map; confidence tiers must not be presented as calibrated probabilities. | **Winner: rank 5.** Essential trust foundation; more modest immediate visual payoff than the other four. |
| 6 | **Directional corridor bands.** Opposing traffic streams become distinct, reducing ambiguity in dense lanes. | Aggregate local segment headings into directional bins; medium scope. Global east/west coloring, already partly available, is inadequate for curved routes. Parallel offset bands can imply different physical lanes unless labeled. | Defer until the atlas supports reliable counts and direction semantics. |
| 7 | **Speed-colored tracks.** Acceleration and slow zones become visible immediately. | Retain SOG per vertex and add a sequential legend; small–medium scope. Color cannot simultaneously encode speed, type, and certainty. Missing SOG must not become zero. | Incorporate as an optional lens in the journey inspector, rather than a separate flagship feature. |
| 8 | **Consistent, accessible vessel symbols and legends.** Users learn one color/shape vocabulary across replay, dashboard, and exports. | Centralize palettes and category names; add text, shape, focus states, and contrast checks; small–medium scope. Eight simultaneous categories still overload a dense overview. | Baseline requirement across winners; especially valuable in the atlas. |
| 9 | **Port occupancy and dwell-time view.** Helps users see concentrations of stopped vessels and long visits. | Join port zones to time-resolved vessel presence and display distributions; large scope. Slow presence does not by itself identify a queue, available berth, or congestion; the current boundary helper keeps one preferred zone per port. | Strong follow-on after journey inspection and better zone handling. |
| 10 | **Port-to-port flow map.** Makes connections and changing vessel flows easy to explain. | Sequence reliable port-call events and aggregate observed transitions; large scope. Regional archives censor voyages; stated AIS destinations are not confirmed arrivals, and vessel counts are not cargo volume. | Defer: attractive but easy to overstate with incomplete trajectories. |
| 11 | **Hour-by-day traffic calendar.** Exposes recurring activity peaks more efficiently than repeated playback. | Aggregate unique vessels or crossings by weekday/hour with observed-time denominators; medium scope. Short archives and changing coverage create apparent seasonality. | Useful later, once comparison windows and observation support exist. |
| 12 | **Paired encounter replay.** Lets users inspect two nearby vessels together with distance over time. | Reuse encounter events and synchronize both tracks; medium scope. Time buckets, positioning error, and ordinary port proximity can resemble meaningful meetings. | Add as a focused journey-inspector extension; do not imply cargo transfer. |
| 13 | **Ranked anomaly inbox.** Directs attention toward unusual routes or behavior. | Build a baseline, explainable features, thresholds, and feedback loop; large scope. Receiver gaps, fleet differences, and routine operations can dominate alerts; confidence in a red warning is easily excessive. | Defer until baselines and validation data justify an anomaly model. |
| 14 | **Live operational map with freshness.** Useful for watching an active feed and seeing stale contacts. | Incremental position state, expiry rules, reconnection handling, and live chart updates; large scope. The current event detectors are batch-based, and replay labels must not imply live surveillance. | Valuable for a different deployment goal; lower fit with the current archival/shareable workflow. |
| 15 | **Predicted vessel positions and uncertainty fans.** Could help users anticipate motion. | Short-horizon motion model with evaluation by vessel/operating context; large scope. Constant velocity fails around ports; an attractive fan is not a calibrated uncertainty region. | Defer. Displaying observations faithfully has much stronger support today. |
| 16 | **Cinematic 3D globe and camera flights.** Memorable overview for presentations. | Globe projection, horizon behavior, camera paths, and broad coverage; medium–large scope. Occlusion and projection make local traffic comparison harder, while current samples emphasize regional waters. | Keep as a presentation experiment, below the five analytical improvements. |
| 17 | **Extruded density towers.** Creates an immediate impression of intensity. | Enable 3D cell extrusion; small rendering change. Nearby towers occlude one another, height exaggerates small differences, and camera angle becomes part of the measurement. | Reject as the default; flat quantitative density is clearer. |
| 18 | **Space-time cube.** Shows trajectories with time as a third axis for specialists. | Encode longitude/latitude/time as 3D paths with slicing and picking; large scope. Navigation and overlapping paths impose a substantial learning burden. | Defer to a specialist notebook, not the primary interface. |
| 19 | **Route bundling.** Reduces a mass of overlapping tracks into recognizable connections. | Cluster similar routes and render bundled representative lines; large scope. Artificial curves can cross land or conceal rare but important deviations; temporal behavior disappears. | Prefer spatial aggregation first; revisit for explicit origin/destination analysis. |
| 20 | **Lasso and polygon selection.** Makes local investigations possible without writing a bbox. | Draw/edit geometry, select intersecting observations, and propagate the cohort; medium scope. Users need to know whether touching the region once selects an entire voyage or only observations inside it. | Extend winner 1 after simple map-cell and time selections work. |
| 21 | **Draw a gate interactively.** Lets analysts investigate a chokepoint without editing Python coordinates. | Two endpoints, direction labels, debounce crossing recomputation, and persist the gate; medium scope. Endpoint order determines inbound/outbound; jitter can produce repeated crossings. | Strong follow-on to coordinated exploration, not a substitute for reliable totals. |
| 22 | **Synchronized fleet or port small multiples.** Makes class/region differences visible without overlay clutter. | Reuse existing multi-panel support but share metric definitions and scales; medium scope. Different geography and independently normalized brightness undermine comparisons. | Fold into winner 4; multi-panel rendering already exists. |
| 23 | **Saved, annotated story scenes.** Makes discoveries reproducible and easy to share. | Save camera, interval, filters, selection, and text annotations in an export; medium scope. Annotation ownership and stale data need clear treatment, and storytelling does not fix exploration itself. | Best sixth feature after the five; unusually good fit with standalone HTML. |
| 24 | **Shareable investigation state.** A colleague opens the same filtered view. | Versioned URL hash or embedded state plus dataset identity; small–medium scope. A state link cannot supply a missing local dataset; arbitrary client state needs validation. | Include as a finishing step for winner 1 and annotated exports. |
| 25 | **Report-ready static figures.** Supports papers, briefings, and comparisons outside the interactive view. | Deterministic rendering, readable legends, scale bars, UTC interval, sources, and captions; medium scope. A screenshot alone cannot explain sampling, uncertainty, or the selected metric. | High-value follow-on; use the same semantics established by winners 2, 4, and 5. |
| 26 | **Weather/current and environmental overlays.** Adds context for routes and behavior. | Acquire, cache, time-align, and reproject external grids; large scope. Resolution mismatch and unavailable layers create apparent explanations that the AIS data alone cannot establish. | Defer pending a concrete user question and data source. |
| 27 | **EEZ, protected-area, and regulatory context.** Makes proximity and boundary transitions easier to interpret. | Layer maintained boundary geometry with attribution and dates; medium–large scope. Jurisdictional disputes, stale geometry, and ambiguous activity make labels consequential. | Selective context in the inspector; do not turn spatial overlap into a legal conclusion. |
| 28 | **Natural-language visualization controls.** Helps users formulate a query without AIS vocabulary. | Map requests into explicit filter/query state with confirmation of interpretation; large scope. The proposed chatbot plan is not implemented, and semantic errors can be concealed behind fluent explanations. | Defer until deterministic interactions produce trustworthy views that an assistant can operate. |
| 29 | **More bloom, particles, and cinematic presets.** Improves visual spectacle in videos. | Tune the existing D3/Canvas/WebGL effects; small–medium scope. Brighter overlaps erase vessel-type color, and exposure/decay can change apparent importance without changing the data. | Preserve tasteful presentation presets; not a top-five functional improvement. |
| 30 | **Automatic highlight reel.** Summarizes interesting movement without manual editing. | Rank candidate scenes, choose camera paths, and render clips; large scope. Scene selection can sensationalize ordinary events and requires a reliable definition of interesting. | Defer until events, comparisons, and saved scenes provide defensible inputs. |

## 1. Make every view respond to the same selection

**User experience.** An analyst notices a spike in crossings, drags over that interval, and sees the relevant tracks, vessels, events, and updated totals. Clicking “Tankers” narrows every view. Clicking a chart mark highlights its corresponding vessels; clicking a vessel finds its row and context. A visible sentence or compact chip row states the selected UTC interval, region, cohort, and metric scope.

This should feel predictable. A user should never have to work out whether a number describes the whole archive, the displayed sample, the current animation frame, or a selected range. Preserve an explicitly labeled full-dataset benchmark where useful.

**Why this is first.** It benefits nearly every investigation and corrects a demonstrated weakness in the current controls. A visually polished filter that changes only part of the display can lead to an incorrect conclusion. Coordinated interaction both reduces that risk and changes the map into a useful way to ask questions.

**Implementation.** Extend the existing dashboard rather than introducing a separate application framework. Maintain one state object for time interval, playback cursor, selected vessel, vessel cohort, and an optional explicit spatial selection. Keep cursor and interval separate: a playhead is an instant; an analysis range is a population over time. Panning should not silently filter the analysis.

Retain crossing records with MMSI, time, direction, and location instead of relying exclusively on precomputed daily totals and an anonymous timestamp list. Compute lightweight cohort aggregates when selection changes, not every animation frame. Use indexed observations/events or precomputed per-vessel/time contributions for the exported dataset. A compact index can support small exports; large-data query services should only be introduced when demonstrated payload limits require them.

Calculate analytical results from the full requested data before any rendering reduction. Displayed track sampling should have its own count and an explanation. Arbitrary range filtering must consider timestamp overlap and clip visible paths; testing a trip's start time alone is insufficient.

**First release.** A range brush, vessel-type selection, linked vessel list, shared map/chart counts, explicit scope labels, and a reset action. Make the existing mode/period controls functional or remove redundant choices. Add polygon selection and saved state later.

**How to validate it.** Use a small fixture with known crossings inside and outside each time/cohort boundary. Verify agreement among the selected list, map, chart, and exported counts, including zero matches. Give users a task such as “find the vessels responsible for the busiest hour” and measure correctness, completion time, and whether they can explain the count's scope.

**Confidence: very high.** The need is supported by code and browser behavior. The exact time saved is unmeasured. The main risk is adding interaction ambiguity; explicit selection state and a small first release control that risk.

## 2. Build a traffic atlas with explicit measures and appropriate detail at each zoom

**User experience.** At a regional scale, users see readable concentrations and principal corridors. Zooming into a port reveals individual vessel tracks and direction markers. Selecting a cell explains its quantity: for example, “42 distinct observed vessels in this cell, selected UTC interval.” A metric selector distinguishes observed vessels, observed vessel-hours, and AIS reports.

Brightness should have a stable meaning. A narrow, visually spectacular corridor and an anchorage full of repeated reports should not be interchangeable definitions of “busy.” Cinematic accumulation remains available as a presentation mode with clear labeling.

**Why this is second.** It improves the appearance and interpretation of almost every map, including the main promotional examples. Scale-dependent detail reduces clutter without deleting all context. Explicit units turn density into a measurable result and eliminate a major source of misinterpretation.

**Implementation.** Extend the density derivation to produce a consistent cell schema and explicit grid type. The existing stored rounded-grid counts and presentation-time H3 counts should not be treated as equivalent. Precompute a limited number of spatial resolutions for the selected dataset, then switch layers by zoom with a short crossfade and stable thresholds.

Render actual cell boundaries with an H3 layer or precomputed polygons, rather than blurring only the centers of large cells. This is supported by deck.gl's existing [H3HexagonLayer](https://deck.gl/docs/api-reference/geo-layers/h3-hexagon-layer); using its standalone bundle also requires the documented h3-js dependency. Keep the map flat in analytical mode, with subdued geography, a quantitative legend, consistent type colors, and restrained head markers.

Compute distinct MMSIs over the entire selected interval per cell. Do not sum hourly distinct counts or child-cell distinct counts: the same vessel can appear repeatedly. Start with direct Polars grouping over the scoped data; retain membership information or use a properly mergeable approximation only if scale requires it, labeling any approximation.

Observed vessel-hours need a declared model. Integrate only acceptable observation intervals, cap gaps, allocate time across traversed cells or use a disclosed approximation, and avoid double-counting overlapping source observations. Label this “observed vessel-hours,” not complete maritime occupancy. AIS reporting rates vary with vessel class, speed, maneuvering, and status, so raw message volume is not a neutral traffic measure. [USCG AIS system types](https://navcen.uscg.gov/types-of-ais).

**First release.** Exact distinct observed vessels and report counts, two or three spatial resolutions, cell tooltips, a fixed quantitative color scale for a chosen metric/resolution, and individual tracks at close zoom. Add vessel-hours after its observation-gap semantics are validated. Preserve a selected ship while zooming.

**How to validate it.** Compare a heavily reporting stationary ship with many less frequently reporting moving ships. Duplicating messages must not change distinct-vessel results. Verify exact counts against source queries and assess whether users can identify the busiest cell and explain the legend. Benchmark transitions on representative datasets rather than promising a specific frame rate in advance.

**Confidence: high.** The overplotting and metric mismatch are concrete. Automatic detail transitions need user testing because unexpected representation changes can be disorienting. Do not autoscale colors independently in ways that imply unchanged density across different quantities.

## 3. Turn vessel selection into a journey inspector

**User experience.** Clicking a ship opens a compact panel containing its identity, speed-over-time chart, a strip of events and observation gaps, and its highlighted path. Moving over the chart positions a marker at the corresponding map time; selecting a port call or crossing focuses that interval. The rest of the fleet remains visible but subdued.

For example, a user could investigate a slowdown near a port by comparing observed speed, a detected low-speed interval, the port footprint, and the reports supporting the detector. An encounter opens the other vessel beside it. The UI describes the evidence without inventing a story about the ship's intent.

**Why this is third.** Neptune has already done substantial work on event detection, provenance, vessel identity, and port intelligence. Most of that value is absent from the standalone dashboard. The inspector makes those existing capabilities tangible and gives a user somewhere useful to go after an overview reveals something interesting.

**Implementation.** Extend `selectVessel()` and the current detail card. Index events by both `mmsi` and `other_mmsi`, so either member of an encounter can find the event. Keep stable IDs and original timestamps across Python output and browser selections. Retain speed observations and source/QC context alongside the compact geometry payload; a synthetic segment bearing should not be mislabeled as transmitted heading.

Use event intervals on the timeline and appropriately located map markers. Expose detector name, threshold settings, provenance, and supporting observations in expandable details. Present “detected port call” or “low-speed interval,” not unverified anchoring, fishing, cargo transfer, or berth assignment. Confidence bands are heuristic support, not probability of guilt or intent.

Use time-aware vessel metadata: the destination last observed anywhere in the archive may be later than the current playback instant. If an as-of join is unavailable, label the card as last-known dataset metadata. Distinguish reported SOG from speed calculated between observations.

Port geometry is available, but `prepare_ports()` currently prefers one derived zone per port. Showing all port activity zones requires preserving zone identities and confidence rather than assuming the existing helper already exports a complete terminal/anchorage model.

**First release.** Identity, a speed chart, supplied event intervals, explicit observation gaps, and map/time linking for one selected vessel. Add paired encounters and richer zone context next. Avoid an additional dashboard of unrelated metrics.

**How to validate it.** Ask users to identify when a vessel slowed, which observations support a detected event, and whether a gap prevents a conclusion. Check encounter lookup from both vessels, sparse tracks, missing speed, repeated events, and metadata changing during playback.

**Confidence: high.** The data and selection mechanism already exist, and the current unused event payload is direct evidence of unrealized value. Payload size and heuristic overinterpretation are the main risks; progressive detail and explicit provenance address them.

## 4. Add synchronized comparisons that distinguish change from observation differences

**User experience.** Choose two intervals or cohorts and compare two maps with linked cameras and identical legends. A difference view highlights increases and decreases. A small chart reports the metric, both values, their units, the absolute change, and a percentage only when the baseline supports one.

Users should be able to ask “where did observed traffic move?” without remembering a bright trail from the previous playback. This is particularly valuable for ports, chokepoints, disruptions, and changes in vessel mix. The interface should say “observed change,” with causal explanations left to evidence beyond the picture.

**Why this is fourth.** It turns a collection of impressive scenes into a way to compare scenarios. It also corrects existing statistical pitfalls. Its position below the inspector reflects the need for suitable history and comparable data, not a lack of potential value.

**Implementation.** Reuse multi-panel layout experience, but add shared camera state, fixed scales, matched metric definitions, and explicit A/B selections. Use quantitative aggregates rather than differences between glowing rendered images; additive bloom and exposure are unsuitable measurement functions.

For a rate comparison, divide by an explicitly defined duration and disclose source availability. Fill observed zero-crossing bins with zero; leave unobserved or unknown bins visibly missing. A day without crossings cannot be classified as an outage solely because its crossing count is zero. Archive completeness can identify some missing intervals, but does not establish complete receiver coverage.

The current average and rolling-window calculations need a real calendar axis. Before/after raw totals must not be compared as rates when the durations differ. Inbound/outbound curves need one y-scale. A zero baseline should show an absolute change or “no baseline,” not an infinite or arbitrary percentage. Use a diverging difference palette centered on zero and flag low-support cells.

Keep definitions of counts separate: gate crossings, distinct vessels, and vessel-days answer different questions. For example, one vessel crossing repeatedly increases crossings without increasing distinct vessels. Likewise, cell distinct counts cannot be summed into a regional fleet count.

**First release.** Two user-selected intervals with matched duration, a common vessel cohort/source selection, synchronized flat maps, one rate metric, and a small coverage/support panel. Add richer cohort comparisons after this version is understandable.

**How to validate it.** Test identical traffic with unequal window lengths, an observed zero day, a missing day, repeated crossings by one vessel, and a zero baseline. Identical underlying rates must not appear to change merely because one window is longer. Ask users to locate the largest observed change and explain whether the data establishes its cause.

**Confidence: high for analytical users; medium for presentation-only use.** Correct comparisons have obvious value, but coverage corrections must not promise to reconstruct unobserved traffic. Side-by-side maps with explicit limitations are preferable to a precise-looking but unsupported correction factor.

## 5. Make observation gaps, interpolation, and confidence visible

**User experience.** A selected vessel's journey explicitly marks gaps. Its last observed position can have a hollow marker with “last observation 38 minutes earlier”; a later observation starts a new segment. Interpolated playback positions have a distinct state from reported positions. An evidence drawer shows sources, QC results, and event support without covering the map in alerts.

A small observation-support strip helps distinguish a quiet observed interval from missing or poorly understood data. Sparse data should prompt questions about observation, not an automatic “dark activity” label.

**Why this makes the top five.** Neptune's multi-source normalization and QC are central strengths, but polished continuous motion can conceal their limits. Showing those limits preserves the value of the analysis and reduces the chance that an exported visual will suggest more certainty than the data warrants.

**Implementation.** Carry source IDs, observation times, QC flags, and segment break reasons into an optional evidence payload. Use existing track gap/jump logic as the basis, while distinguishing original reporting gaps from gaps introduced by clipping or display decimation. Determine source continuity before thinning and preserve boundary points and reasons.

Do not draw an invented route through a gap. Use endpoint markers and a gap interval; an optional dashed connector must be explicitly described as linking observations rather than locating the missing journey. A growing circle is not a statistical uncertainty bound unless a calibrated model supports that interpretation.

Show heuristic event confidence as low/medium/high support, with an explanation of the detector and settings. Keep quality symbols separate from vessel-type color and selection emphasis. Mark empirical port footprints separately from reference boundaries. An observation-support overlay should be named for what it measures; AIS positions alone cannot establish complete receiver coverage.

**First release.** Gap intervals, an observation-versus-interpolation label on the selected vessel, a last-observation timestamp, source/QC details, and exact-versus-sampled count labeling. These are useful without building a probabilistic model or a new backend.

**How to validate it.** Give users a path with a long outage, a rejected jump, and sparse observations. They should correctly identify what is unknown and avoid inferring a continuously observed route or intentional AIS shutdown. Verify that display downsampling cannot manufacture an unlabeled original-data gap.

**Confidence: high on value, moderate on the best visual treatment.** This ranks fifth as a standalone product feature because it has less immediate visual impact and can add clutter. Basic accuracy labels and gap honesty belong in the first releases of all the other recommendations.

## Implementation order and follow-up findings

Product-value ranking: **1 coordinated exploration; 2 quantitative traffic atlas; 3 journey inspector; 4 valid comparisons; 5 visible observation limits**.

Build order can differ: first fix selection/count semantics and basic quality labels; then add the atlas and journey inspector; then add comparisons. All five can start within the current Python-to-standalone-HTML architecture. No evidence from this review justifies a frontend-framework migration or a new mandatory server.

Preserve the D3 cinematic renderer as a presentation path. Analytical views should share consistent selection, units, and evidence semantics with it where applicable, without requiring every renderer to implement every investigative interaction.

Two concrete bug follow-ups discovered during this review:

1. **P1: Dashboard selection consistency.** Make period controls functional and align filtered tracks, crossing records, charts, counters, and density. Clearly label any intentional full-dataset reference. Acceptance: known cohort/range fixtures agree across views, including empty selections and tracks spanning boundaries.
2. **P1: Comparison denominators and chart scales.** Use explicit calendar windows, distinguish zero from missing bins, normalize comparable rates, handle zero baselines, and share directional scales. Acceptance: unequal-duration windows with identical traffic rates do not imply change; missing data remains distinguishable from observed zero.

Beads workflow was attempted. `bd onboard` and `bd ready` ran, but `bd create` failed because the current database lacks `issue_prefix`. The installed CLI also has no `bd sync` command and reports no configured Dolt remote. No issue IDs were created, and the existing Beads migration/deletion in the working tree was preserved. The two findings are recorded here so they remain actionable without reinitializing the user's issue database.

Confidence in this ranking comes from concrete implementation gaps, existing data support, and low dependence on speculative new models. Whether users prefer the proposed interaction details still requires task-based testing; no adoption, speed, or accuracy improvement is claimed as measured.
