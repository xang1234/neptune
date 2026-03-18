"""SQL — DuckDB integration layer.

Provides a DuckDB connection bound to the Neptune catalog so users can run
SQL directly over cataloged Parquet datasets.

Module role — query engine bridge
---------------------------------
**Owns:**
- DuckDB connection lifecycle and configuration.
- Registering cataloged Parquet paths as DuckDB views/tables.
- The ``Neptune.duckdb()`` and ``Neptune.sql()`` implementations.

**Does not own:**
- Catalog lookups — delegates to ``catalog`` for file discovery.
- Schema definitions — uses ``datasets`` for table registration.

**Import rule:** SQL may import from ``catalog`` and ``datasets``. It must
not import from ``adapters``, ``derive``, ``geometry``, or ``cli``.

Note: The Phase 1 DuckDB integration is implemented directly in
``api.Neptune.duckdb()`` and ``api.Neptune.sql()`` for simplicity.
This module will host shared DuckDB utilities (connection pooling,
spatial extension loading, custom functions) as the integration matures.
"""

from __future__ import annotations
