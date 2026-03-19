"""Core performance benchmarks for Neptune AIS.

Measures baseline performance for the three main workflows:
1. Ingestion: DataFrame creation, normalization, Parquet write
2. Query: Polars lazy scan, filter, group_by, DuckDB SQL
3. Derivation: track segmentation, event detection, compaction

Results are printed as a structured report and optionally saved to
JSON for CI comparison against baselines.

Usage:
    python benchmarks/bench_core.py [--save baseline.json] [--rows 100000]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def generate_positions(n_rows: int, n_vessels: int = 100) -> pl.DataFrame:
    """Generate synthetic AIS position data for benchmarking."""
    import random

    random.seed(42)

    rows_per_vessel = n_rows // n_vessels
    records: list[dict[str, Any]] = []

    for v in range(n_vessels):
        mmsi = 200000000 + v
        base_lat = 40.0 + random.uniform(-10, 10)
        base_lon = -74.0 + random.uniform(-20, 20)

        for i in range(rows_per_vessel):
            records.append({
                "mmsi": mmsi,
                "timestamp": f"2024-06-15T{(i // 3600) % 24:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}",
                "lat": base_lat + random.uniform(-0.1, 0.1),
                "lon": base_lon + random.uniform(-0.1, 0.1),
                "sog": random.uniform(0, 20),
                "cog": random.uniform(0, 360),
                "heading": random.choice([random.uniform(0, 360), 511.0]),
                "vessel_name": f"VESSEL_{v:04d}",
                "ship_type": str(random.choice([30, 60, 70, 80])),
                "source": "benchmark",
            })

    return pl.DataFrame(records)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkResult:
    """Stores timing results for a single benchmark."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.timings: dict[str, float] = {}

    def time(self, label: str):
        """Context manager that times a block."""
        return _Timer(self, label)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "timings_ms": {
            k: round(v * 1000, 2) for k, v in self.timings.items()
        }}


class _Timer:
    def __init__(self, result: BenchmarkResult, label: str) -> None:
        self._result = result
        self._label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self._result.timings[self._label] = elapsed


# ---------------------------------------------------------------------------
# Benchmark: Ingestion
# ---------------------------------------------------------------------------


def bench_ingestion(df: pl.DataFrame, tmp_dir: Path) -> BenchmarkResult:
    """Benchmark ingestion pipeline: sort + write Parquet."""
    result = BenchmarkResult("ingestion")

    with result.time("sort_mmsi_timestamp"):
        sorted_df = df.sort(["mmsi", "timestamp"])

    with result.time("write_parquet"):
        sorted_df.write_parquet(
            tmp_dir / "bench_positions.parquet",
            compression="zstd",
            compression_level=3,
            row_group_size=500_000,
            statistics=True,
        )

    with result.time("read_parquet"):
        _ = pl.read_parquet(tmp_dir / "bench_positions.parquet")

    return result


# ---------------------------------------------------------------------------
# Benchmark: Query
# ---------------------------------------------------------------------------


def bench_query(df: pl.DataFrame) -> BenchmarkResult:
    """Benchmark query patterns: filter, group_by, latest positions."""
    result = BenchmarkResult("query")

    lf = df.lazy()

    with result.time("filter_single_vessel"):
        _ = lf.filter(pl.col("mmsi") == 200000050).collect()

    with result.time("filter_bbox"):
        _ = lf.filter(
            (pl.col("lat").is_between(35.0, 45.0))
            & (pl.col("lon").is_between(-80.0, -70.0))
        ).collect()

    with result.time("group_by_vessel_count"):
        _ = lf.group_by("mmsi").agg(pl.len().alias("count")).collect()

    with result.time("latest_positions"):
        _ = (
            lf.group_by("mmsi")
            .agg(pl.all().sort_by("timestamp").last())
            .collect()
        )

    with result.time("top_speed_vessels"):
        _ = (
            lf.group_by("mmsi")
            .agg(pl.col("sog").max().alias("max_sog"))
            .sort("max_sog", descending=True)
            .head(10)
            .collect()
        )

    return result


# ---------------------------------------------------------------------------
# Benchmark: DuckDB SQL
# ---------------------------------------------------------------------------


def bench_duckdb(df: pl.DataFrame, tmp_dir: Path) -> BenchmarkResult:
    """Benchmark DuckDB SQL queries over positions data."""
    import duckdb

    result = BenchmarkResult("duckdb_sql")

    con = duckdb.connect()

    # Write to temp Parquet for DuckDB to read (avoids pyarrow dependency).
    tmp = tmp_dir / "bench_duckdb.parquet"
    df.write_parquet(tmp)
    con.execute(f"CREATE TABLE positions AS SELECT * FROM read_parquet('{tmp}')")

    with result.time("count_all"):
        con.execute("SELECT count(*) FROM positions").fetchone()

    with result.time("group_by_vessel"):
        con.execute(
            "SELECT mmsi, count(*) as n FROM positions GROUP BY mmsi ORDER BY n DESC LIMIT 10"
        ).fetchall()

    with result.time("filter_bbox"):
        con.execute(
            "SELECT * FROM positions WHERE lat BETWEEN 35 AND 45 AND lon BETWEEN -80 AND -70"
        ).fetchall()

    with result.time("avg_speed_by_type"):
        con.execute(
            "SELECT ship_type, avg(sog) as avg_sog FROM positions GROUP BY ship_type"
        ).fetchall()

    con.close()
    return result


# ---------------------------------------------------------------------------
# Benchmark: Derivation (compaction + track-style operations)
# ---------------------------------------------------------------------------


def bench_derivation(df: pl.DataFrame) -> BenchmarkResult:
    """Benchmark derivation operations: compaction, dedup, time gaps."""
    from neptune_ais.stream import compact_batch, DEDUP_KEY_FIELDS

    result = BenchmarkResult("derivation")

    # Compaction benchmark: convert to dicts, compact, convert back.
    records = df.head(10_000).to_dicts()

    with result.time("compact_10k_messages"):
        _ = compact_batch(records)

    # Dedup via Polars unique.
    with result.time("polars_unique_dedup"):
        _ = df.unique(subset=["mmsi", "timestamp", "lat", "lon"], keep="any")

    # Time-gap computation (track segmentation proxy).
    with result.time("time_gap_computation"):
        _ = (
            df.lazy()
            .sort(["mmsi", "timestamp"])
            .with_columns(
                pl.col("timestamp")
                .shift(1)
                .over("mmsi")
                .alias("prev_ts")
            )
            .collect()
        )

    # Per-vessel statistics (track summary proxy).
    with result.time("per_vessel_stats"):
        _ = (
            df.lazy()
            .group_by("mmsi")
            .agg([
                pl.len().alias("n_points"),
                pl.col("sog").mean().alias("avg_sog"),
                pl.col("sog").max().alias("max_sog"),
                pl.col("lat").min().alias("min_lat"),
                pl.col("lat").max().alias("max_lat"),
            ])
            .collect()
        )

    return result


# ---------------------------------------------------------------------------
# Benchmark: Streaming operations
# ---------------------------------------------------------------------------


def bench_streaming() -> BenchmarkResult:
    """Benchmark streaming operations: ingest, dedup, backpressure."""
    import asyncio
    from neptune_ais.stream import (
        NeptuneStream, StreamConfig, BackpressurePolicy,
    )

    result = BenchmarkResult("streaming")

    async def _bench_ingest(n: int):
        config = StreamConfig(
            max_queue_size=10_000,
            backpressure=BackpressurePolicy.DROP_OLDEST,
        )
        async with NeptuneStream(config=config) as stream:
            for i in range(n):
                await stream.ingest({
                    "mmsi": i % 100,
                    "timestamp": f"2024-01-01T{(i // 3600) % 24:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}",
                    "lat": 40.0,
                    "lon": -74.0,
                })

    with result.time("ingest_10k_messages"):
        asyncio.run(_bench_ingest(10_000))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_all(n_rows: int = 100_000) -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    import tempfile

    print(f"Generating {n_rows:,} synthetic positions ({n_rows // 100} per vessel)...")
    df = generate_positions(n_rows)
    print(f"  DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory: ~{df.estimated_size() / 1024 / 1024:.1f} MB")

    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("\n--- Ingestion benchmarks ---")
        r = bench_ingestion(df, tmp_dir)
        results.append(r)
        for k, v in r.timings.items():
            print(f"  {k}: {v * 1000:.1f} ms")

        print("\n--- Query benchmarks ---")
        r = bench_query(df)
        results.append(r)
        for k, v in r.timings.items():
            print(f"  {k}: {v * 1000:.1f} ms")

        print("\n--- DuckDB SQL benchmarks ---")
        try:
            r = bench_duckdb(df, tmp_dir)
            results.append(r)
            for k, v in r.timings.items():
                print(f"  {k}: {v * 1000:.1f} ms")
        except ImportError:
            print("  (skipped — duckdb not installed)")

        print("\n--- Derivation benchmarks ---")
        r = bench_derivation(df)
        results.append(r)
        for k, v in r.timings.items():
            print(f"  {k}: {v * 1000:.1f} ms")

        print("\n--- Streaming benchmarks ---")
        r = bench_streaming()
        results.append(r)
        for k, v in r.timings.items():
            print(f"  {k}: {v * 1000:.1f} ms")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neptune AIS performance benchmarks")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of rows to generate")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_all(args.rows)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rows": args.rows,
        "benchmarks": [r.to_dict() for r in results],
    }

    if args.save:
        Path(args.save).write_text(json.dumps(report, indent=2))
        print(f"\nResults saved to {args.save}")

    print(f"\nTotal benchmarks: {sum(len(r.timings) for r in results)}")


if __name__ == "__main__":
    main()
