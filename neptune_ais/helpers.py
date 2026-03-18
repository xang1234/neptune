"""Helpers — high-level maritime primitives.

Convenience functions for common maritime questions: latest positions,
port calls, encounters, loitering, EEZ crossings, density, vessel history,
and point-in-time snapshots.

Module role — convenience API
-----------------------------
**Owns:**
- Implementations of ``Neptune.latest_positions()``, ``port_calls()``,
  ``encounters()``, ``loitering()``, ``eez_crossings()``, ``density()``,
  ``vessel_history()``, and ``snapshot()``.
- These compose ``derive`` pipelines and ``datasets`` queries into
  user-friendly one-call methods.

**Does not own:**
- The derivation algorithms themselves — those live in ``derive``.
- Schema definitions — those live in ``datasets``.
- Spatial lookups — those live in ``geometry``.

**Import rule:** Helpers may import from ``datasets``, ``derive``, and
``geometry``. It must not import from ``adapters``, ``storage``,
``catalog``, or ``cli``.
"""

from __future__ import annotations
