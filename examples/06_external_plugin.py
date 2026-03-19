"""Example 6: External plugin — add a custom AIS source to Neptune.

Shows how to create an external adapter that plugs into Neptune's
source model without modifying the core package.

Step 1: Implement the SourceAdapter protocol.
Step 2: Register via Python entry points in your pyproject.toml.
Step 3: Users discover your source via `neptune sources`.
"""

# --- Step 1: Implement the adapter ---

# from neptune_ais.adapters.base import (
#     FetchSpec,
#     RawArtifact,
#     SourceAdapter,
#     SourceCapabilities,
# )
# import polars as pl
#
# class MyCompanyAdapter:
#     SOURCE_ID = "mycompany"
#
#     @property
#     def source_id(self) -> str:
#         return self.SOURCE_ID
#
#     @property
#     def capabilities(self) -> SourceCapabilities:
#         return SourceCapabilities(
#             source_id=self.SOURCE_ID,
#             provider="My Company AIS",
#             description="Internal AIS data from our receiver network.",
#             supports_backfill=True,
#             supports_streaming=False,
#             auth_scheme="api_key",
#             coverage="North Sea",
#             history_start="2023-01-01",
#             datasets_provided=["positions"],
#             delivery_format="CSV files",
#         )
#
#     def available_dates(self):
#         return None  # or (date(2023, 1, 1), date.today())
#
#     def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
#         # Download from your internal API/file server
#         ...
#
#     def normalize_positions(self, artifacts: list[RawArtifact]) -> pl.DataFrame:
#         # Parse your format → canonical positions schema
#         ...
#
#     def normalize_vessels(self, artifacts):
#         return None  # or extract vessels
#
#     def qc_rules(self):
#         return []

# --- Step 2: Register in your pyproject.toml ---
#
# [project.entry-points."neptune_ais.adapters"]
# mycompany = "my_package.adapter:MyCompanyAdapter"

# --- Step 3: After pip install, your source appears automatically ---
#
# $ neptune sources
# Source       Provider                       Coverage
# -------------------------------------------------------------------------
# ...
# mycompany    My Company AIS                 North Sea
#
# $ neptune sources mycompany
# mycompany — My Company AIS
#   Internal AIS data from our receiver network.
#   ...

print("Example 6: External plugin adapter.")
print()
print("To create a plugin:")
print("  1. Implement the SourceAdapter protocol (see SourceAdapter in base.py)")
print("  2. Add entry point to pyproject.toml:")
print('     [project.entry-points."neptune_ais.adapters"]')
print('     my_source = "my_package:MyAdapter"')
print("  3. pip install your package — Neptune discovers it automatically")
