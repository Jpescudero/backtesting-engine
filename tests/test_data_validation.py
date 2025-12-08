import pandas as pd
import pytest

from src.data.parquet_to_npz import bars_df_to_npz_arrays


def test_bars_df_to_npz_arrays_rejects_nan_ohlc():
    index = pd.date_range("2024-01-01 00:00", periods=3, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.1, float("nan"), 3.1],
            "volume": [10, 11, 12],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="OHLC NaN"):
        bars_df_to_npz_arrays(df)


def test_bars_df_to_npz_arrays_detects_unordered_index():
    index = pd.to_datetime([
        "2024-01-01 00:02",
        "2024-01-01 00:01",
        "2024-01-01 00:03",
    ]).tz_localize("UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.1, 2.1, 3.1],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="ordenado"):
        bars_df_to_npz_arrays(df)


def test_bars_df_to_npz_arrays_detects_critical_gap():
    index = pd.date_range("2024-01-01 00:00", periods=3, freq="1min", tz="UTC")
    index = index.insert(2, index[1] + pd.Timedelta(minutes=10))
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.5, 2.5, 3.5, 4.5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.1, 2.1, 3.1, 4.1],
        },
        index=index,
    )

    with pytest.raises(ValueError, match="gaps críticos"):
        bars_df_to_npz_arrays(df)
