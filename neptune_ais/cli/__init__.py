"""CLI — command-line interface for Neptune.

Provides ``neptune`` commands for downloading, querying, exporting,
deriving analytics, and generating map visualizations.

Subsystem boundary
------------------
**Owns:**
- Argument parsing, output formatting, and user interaction for the
  ``neptune`` console script.
- Command registration and help text.
- Progress bars, table rendering, and terminal output.

**Delegates to:**
- ``neptune_ais.api`` — all data operations. CLI commands construct a
  ``Neptune`` instance and call its methods; they do not orchestrate
  adapters, storage, or derivation directly.
- ``neptune_ais.sql`` — SQL query execution for the ``neptune sql`` command.
- ``neptune_ais.viz`` — map generation for the ``neptune map`` command.

**Rule:** CLI is a thin presentation layer. It must not contain business
logic, schema definitions, or storage operations. If a CLI command needs
new functionality, that functionality belongs in ``api`` or another core
module — the CLI just wires it to arguments and output.

**Install extra:** ``pip install neptune-ais[cli]`` (click, rich).
"""

from __future__ import annotations

__all__ = [
    "main",
]
