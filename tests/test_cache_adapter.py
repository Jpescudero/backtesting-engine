from __future__ import annotations

import time
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from src.data.cache_adapter import MarketDataAdapter


@pytest.fixture()
def sample_parquet(tmp_path: Path) -> Path:
    root = tmp_path / "parquet"
    root.mkdir()

    timestamps = pd.date_range("2023-01-01", periods=120, freq="T", tz="UTC")
    df = pd.DataFrame(
        {
            "bid": pd.Series(range(120), dtype="float64"),
            "ask": pd.Series(range(120), dtype="float64") + 0.5,
        },
        index=timestamps,
    )

    symbol_dir = root / "TEST"
    year_dir = symbol_dir / "2023"
    year_dir.mkdir(parents=True)
    df.to_parquet(year_dir / "TEST_2023-01-01_00.parquet")
    return root


def test_load_normalizes_and_caches(sample_parquet: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    adapter = MarketDataAdapter(parquet_root=sample_parquet, cache_dir=cache_dir, memory_cache_size=2)

    df = adapter.load("TEST", "2023-01-01", "2023-01-01 01:00", ["bid", "ask"], "1m")

    assert df.index.tz is not None
    assert df.dtypes.to_list() == ["float64", "float64"]
    assert any(cache_dir.glob("*.parquet"))

    df_cached = adapter.load("TEST", "2023-01-01", "2023-01-01 01:00", ["bid", "ask"], "1m")
    pd.testing.assert_frame_equal(df, df_cached)


def test_memory_cache_eviction(sample_parquet: Path, tmp_path: Path) -> None:
    adapter = MarketDataAdapter(parquet_root=sample_parquet, cache_dir=tmp_path / "cache2", memory_cache_size=2)

    adapter.load("TEST", "2023-01-01", "2023-01-01 00:30", ["bid"], "1m")
    first_key = next(iter(adapter._memory_cache._store))

    adapter.load("TEST", "2023-01-01 00:30", "2023-01-01 01:00", ["ask"], "1m")
    adapter.load("TEST", "2023-01-01", "2023-01-01 00:15", ["bid", "ask"], "1m")

    assert first_key not in adapter._memory_cache._store


def test_warm_load_is_faster_and_lighter(sample_parquet: Path, tmp_path: Path) -> None:
    adapter = MarketDataAdapter(parquet_root=sample_parquet, cache_dir=tmp_path / "cache3", memory_cache_size=2)

    start = "2023-01-01"
    end = "2023-01-01 01:00"
    fields = ["bid", "ask"]

    cold_start = time.perf_counter()
    adapter.load("TEST", start, end, fields, "1m")
    cold_elapsed = time.perf_counter() - cold_start

    warm_start = time.perf_counter()
    adapter.load("TEST", start, end, fields, "1m")
    warm_elapsed = time.perf_counter() - warm_start

    assert warm_elapsed <= cold_elapsed

    import tracemalloc

    tracemalloc.start()
    adapter.load("TEST", start, end, fields, "1m")
    cold_current, cold_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    adapter.load("TEST", start, end, fields, "1m")
    warm_current, warm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert warm_peak <= cold_peak
    assert warm_current <= cold_current
