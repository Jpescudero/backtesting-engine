"""Tests for intraday mean reversion data loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from research.intraday_mean_reversion.utils.data_loader import load_intraday_data


@pytest.fixture()
def base_params(tmp_path: Path) -> Dict[str, Any]:
    """Return base parameters with a temporary data path."""

    return {
        "DATA_PATH": str(tmp_path),
        "DATA_FILE_PATTERN": "{symbol}_1m.parquet",
        "START_YEAR": 2020,
        "END_YEAR": 2020,
    }


def _create_npz(path: Path, start: str) -> None:
    """Create a minimal NPZ file containing OHLCV data."""

    timestamps = (
        pd.date_range(start=start, periods=3, freq="min", tz="UTC").view("int64")
    )
    np.savez(
        path,
        ts=timestamps,
        o=np.array([1.0, 2.0, 3.0]),
        h=np.array([1.5, 2.5, 3.5]),
        l=np.array([0.5, 1.5, 2.5]),
        c=np.array([1.2, 2.2, 3.2]),
        v=np.array([10.0, 20.0, 30.0]),
    )


@pytest.mark.usefixtures("base_params")
def test_load_intraday_data_fallbacks_to_supported_format(tmp_path, base_params):
    symbol = "TEST"
    data_dir = tmp_path / symbol
    data_dir.mkdir()
    npz_path = data_dir / f"{symbol}_1m.npz"
    _create_npz(npz_path, start="2020-01-01")

    df = load_intraday_data(symbol, 2020, 2020, base_params)

    assert not df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0].tz == pd.Timestamp.now(tz="UTC").tz


@pytest.mark.usefixtures("base_params")
def test_load_intraday_data_ignores_non_matching_stems(tmp_path, base_params):
    symbol = "TEST"
    data_dir = tmp_path / symbol
    data_dir.mkdir()

    csv_path = data_dir / "OTHER_1m.csv"
    csv_path.write_text("timestamp,open,high,low,close,volume\n2020-01-01 00:00:00,1,1,1,1,1\n")

    with pytest.raises(FileNotFoundError):
        load_intraday_data(symbol, 2020, 2020, base_params)
