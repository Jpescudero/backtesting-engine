"""Tests for data_to_parquet helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

from src import config
from src.data import data_to_parquet


def _patch_path_constants(monkeypatch: pytest.MonkeyPatch, base_dir: Path, mirror_dir: Path) -> None:
    """Patch path constants so data processing happens in temp dirs."""

    data_dir = base_dir / "data"
    parquet_dir = data_dir / "parquet"
    npz_dir = data_dir / "npz"

    monkeypatch.setattr(config.paths, "DATA_DIR", data_dir)
    monkeypatch.setattr(config.paths, "DATA_MIRRORS", [mirror_dir / "data"])
    monkeypatch.setattr(config.paths, "RAW_DATA_DIR", data_dir / "raw")
    monkeypatch.setattr(config.paths, "DARWINEX_RAW_DIR", data_dir / "raw" / "darwinex")
    monkeypatch.setattr(config.paths, "PARQUET_DIR", parquet_dir)
    monkeypatch.setattr(config.paths, "PARQUET_TICKS_DIR", parquet_dir / "ticks")
    monkeypatch.setattr(config.paths, "PARQUET_BARS_1M_DIR", parquet_dir / "bars_1m")
    monkeypatch.setattr(config.paths, "NPZ_DIR", npz_dir)
    monkeypatch.setattr(config.paths, "NPZ_NDXM_DIR", npz_dir / "NDXm")
    monkeypatch.setattr(config.paths, "OTHER_DATA_DIR", data_dir / "other")
    monkeypatch.setattr(config.paths, "NOTEBOOKS_DIR", base_dir / "notebooks")
    monkeypatch.setattr(config.paths, "REPORTS_DIR", base_dir / "reports")
    monkeypatch.setattr(config.paths, "MODELS_DIR", base_dir / "models")
    monkeypatch.setattr(config.paths, "RESEARCH_DIR", base_dir / "research")

    # Keep data_to_parquet module aligned with patched paths
    monkeypatch.setattr(data_to_parquet, "DARWINEX_RAW_DIR", config.paths.DARWINEX_RAW_DIR)
    monkeypatch.setattr(data_to_parquet, "PARQUET_TICKS_DIR", config.paths.PARQUET_TICKS_DIR)
    monkeypatch.setattr(data_to_parquet, "ensure_directories_exist", config.paths.ensure_directories_exist)
    monkeypatch.setattr(data_to_parquet, "resolve_data_dir_with_pattern", config.paths.resolve_data_dir_with_pattern)


def test_data_to_parquet_prefers_mirror_when_primary_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should read raw data from a mirror hub when the primary has no files."""

    primary_root = tmp_path / "primary"
    mirror_root = tmp_path / "mirror"
    mirror_data = mirror_root / "data"
    mirror_raw_symbol = mirror_data / "raw" / "darwinex" / "NDXm"
    mirror_raw_symbol.mkdir(parents=True, exist_ok=True)

    source_file = mirror_raw_symbol / "NDXm_2021-01-01_00_bid.log.gz"
    source_file.write_bytes(b"dummy")

    _patch_path_constants(monkeypatch, base_dir=primary_root, mirror_dir=mirror_root)

    calls: list[Tuple[Path | None, Path | None, Dict[str, Any]]] = []

    def fake_parse_filename(path: Path) -> Dict[str, str]:
        return {"symbol": "NDXm", "date": "2021-01-01", "hour": "00", "side": "bid"}

    def fake_merge_bid_ask_to_parquet(
        bid_path: Path | None,
        ask_path: Path | None,
        symbol: str,
        date: str,
        hour: str,
        out_root: Path | None = None,
    ) -> Path:
        calls.append((bid_path, ask_path, {"symbol": symbol, "date": date, "hour": hour, "out_root": out_root}))
        output_dir = data_to_parquet.PARQUET_TICKS_DIR / symbol
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "dummy.parquet"

    monkeypatch.setattr(data_to_parquet, "parse_filename", fake_parse_filename)
    monkeypatch.setattr(data_to_parquet, "merge_bid_ask_to_parquet", fake_merge_bid_ask_to_parquet)

    data_to_parquet.data_to_parquet(symbol="NDXm")

    assert len(calls) == 1
    bid_path, ask_path, meta = calls[0]
    assert ask_path is None
    assert bid_path == source_file
    assert meta["out_root"] == config.paths.PARQUET_TICKS_DIR


def test_data_to_parquet_creates_missing_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When no raw data exists, the function should create the expected directory."""

    primary_root = tmp_path / "primary"
    mirror_root = tmp_path / "mirror"
    _patch_path_constants(monkeypatch, base_dir=primary_root, mirror_dir=mirror_root)

    symbol_dir = config.paths.DARWINEX_RAW_DIR / "NDXm"
    assert not symbol_dir.exists()

    data_to_parquet.data_to_parquet(symbol="NDXm")

    assert symbol_dir.exists()
    assert list(symbol_dir.glob("*.log.gz")) == []
