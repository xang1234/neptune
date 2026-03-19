"""Norway — Norwegian Coastal Administration adapter.

**Status: DEFERRED** — Not implemented in the current release.

Mixed delivery: TCP NMEA streams and bulk CSV archives for the
Norwegian EEZ.

Deferral rationale
------------------
Norway was assessed during Phase 6 planning and intentionally deferred
for the following reasons:

1. **Unique wire format**: Norway delivers AIS via raw TCP NMEA sentence
   streams, requiring a dedicated NMEA parser. No other adapter needs
   this — all current sources deliver JSON, CSV, or Parquet. The NMEA
   parser would be a significant new dependency with its own test surface.

2. **Overlap with Finland**: For Baltic coverage, Finland (Digitraffic)
   already provides open-data AIS with both archival and streaming modes.
   Adding Norway primarily extends coverage into the Norwegian EEZ,
   which is lower priority than proving the adapter model generalizes
   across delivery styles (already demonstrated by GFW, Finland, AISHub).

3. **Bulk CSV archive access**: Norway's historical data requires
   registration and a data-sharing agreement with the Norwegian Coastal
   Administration (Kystverket). This gating condition means the adapter
   cannot be fully tested without institutional access.

4. **Architecture readiness**: The ``SourceAdapter`` protocol and
   ``SourceCapabilities`` model are proven to generalize across REST
   JSON (GFW, AISHub), REST JSON with scaling (Finland), and WebSocket
   JSON (AISStream). Adding NMEA TCP would validate yet another delivery
   style, but the marginal architectural insight is low compared to the
   implementation cost.

Criteria for reopening
----------------------
Revisit this deferral if any of the following change:

- A user or stakeholder requests Norwegian EEZ coverage specifically.
- A well-tested Python NMEA parser becomes available as a lightweight
  dependency (e.g. ``pyais`` or ``libais``).
- The Norwegian Coastal Administration opens a REST API or provides
  pre-parsed data in JSON/CSV format.
- Neptune needs to prove its adapter model works with raw sentence-level
  AIS data (NMEA 0183 / IEC 62320).

When reopening, the adapter should:

- Implement ``SourceAdapter`` protocol with ``supports_backfill=True``
  and ``supports_streaming=True`` (mixed delivery).
- Map NMEA sentence types to canonical positions schema.
- Handle NMEA-specific quirks: multi-sentence messages, sentence
  checksums, talker IDs, proprietary sentences.
- Declare the institutional access requirement in ``SourceCapabilities``.
"""

from __future__ import annotations
