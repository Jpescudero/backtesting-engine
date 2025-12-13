from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_mean_reversion.study_1.feature_engineering import parse_allowed_time_windows
from research.intraday_mean_reversion.study_1.regime_filters import evaluate_filters
from research.intraday_mean_reversion.study_1.runner import run_study, _REQUIRED_OUTPUT_COLUMNS


def test_parse_allowed_time_windows_parses_ranges() -> None:
    windows = parse_allowed_time_windows("09:45-11:30,13:30-15:30")
    assert len(windows) == 2
    assert windows[0][0].hour == 9 and windows[0][0].minute == 45
    assert windows[1][1].hour == 15 and windows[1][1].minute == 30


def test_evaluate_filters_alignment() -> None:
    idx = pd.date_range("2022-01-03 09:30", periods=5, freq="min", tz="UTC")
    features = pd.DataFrame(
        {
            "realized_vol": np.linspace(0.1, 0.5, len(idx)),
            "trend_strength": np.linspace(0.0, 0.4, len(idx)),
            "shock_active": [False, False, True, False, False],
        },
        index=idx,
    )
    events_index = idx[::2]
    params = {
        "vol_threshold_type": "absolute",
        "vol_threshold_value": 0.3,
        "vol_regime_mode": "avoid_high_vol",
        "trend_threshold": 0.2,
        "use_time_filter": False,
    }
    filters = evaluate_filters(features, events_index, params)
    assert list(filters.index) == list(events_index)
    assert all(filters.dtypes == bool)


def test_run_study_produces_outputs(tmp_path: Path) -> None:
    idx = pd.date_range("2022-01-03 15:35", periods=120, freq="min", tz="UTC")
    base = np.linspace(100.0, 102.0, len(idx)) + np.sin(np.arange(len(idx)) / 10)
    close = pd.Series(base, index=idx)
    df = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1_000,
        },
        index=idx,
    )

    params = {
        "SYMBOL": "SP500",
        "START_YEAR": 2022,
        "END_YEAR": 2022,
        "DATA_PATH": "unused",
        "DATA_FILE_PATTERN": "unused",
        "LOOKBACK_MINUTES": 5,
        "ZSCORE_ENTRY": 0.5,
        "HOLD_TIME_BARS": 3,
        "SESSION_START_TIME": "15:30",
        "SESSION_END_TIME": "18:30",
        "vol_window_min": 5,
        "trend_window_min": 5,
        "shock_window_min": 5,
    }

    run_dir = run_study(params, tmp_path, df=df, run_id="test_run")
    trade_log = pd.read_csv(run_dir / "trade_log.csv")
    regime_summary = pd.read_csv(run_dir / "regime_summary.csv")

    assert set(_REQUIRED_OUTPUT_COLUMNS).issubset(trade_log.columns)
    assert {"bucket_id", "n_trades"}.issubset(regime_summary.columns)
