"""Unit tests for OpeningSweepV4 parameter handling."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.data.feeds import OHLCVArrays
from src.strategies.opening_sweep_v4 import OpeningSweepV4, OpeningSweepV4Params


def test_opening_sweep_accepts_dataclass_config() -> None:
    """Passing a dataclass config should initialize without errors and keep values."""

    params = OpeningSweepV4Params(wick_factor=2.1)
    strategy = OpeningSweepV4(config=params)

    assert strategy.params.wick_factor == 2.1
    assert strategy.config == asdict(params)


def test_opening_sweep_merges_partial_dict_config() -> None:
    """Dict configs should merge with defaults preserving unspecified fields."""

    strategy = OpeningSweepV4(config={"wick_factor": 1.9})

    assert strategy.params.wick_factor == 1.9
    assert strategy.params.max_horizon == OpeningSweepV4Params().max_horizon
    assert strategy.config["wick_factor"] == 1.9


def test_opening_sweep_session_filter_blocks_out_of_window() -> None:
    ts = pd.date_range("2024-01-01 09:00", periods=5, freq="1min", tz="UTC").view(np.int64)
    o = np.full(5, 10.0)
    h = np.full(5, 11.0)
    low = np.full(5, 8.0)
    c = np.array([9.0, 9.0, 10.5, 10.4, 10.6])
    v = np.full(5, 100.0)
    data = OHLCVArrays(ts=ts, o=o, h=h, low=low, c=c, v=v)

    params = OpeningSweepV4Params(
        wick_factor=0.0,
        atr_percentile=0.0,
        volume_percentile=0.0,
        session_windows=((9 * 60, 9 * 60 + 1),),
        trading_timezone="UTC",
    )
    strategy = OpeningSweepV4(config=params)

    result = strategy.generate_strategy_result(data)

    assert result.signals.sum() == 0


def test_opening_sweep_limits_trades_per_day() -> None:
    ts = pd.date_range("2024-01-01 09:00", periods=6, freq="1min", tz="UTC").view(np.int64)
    o = np.full(6, 10.0)
    h = np.full(6, 11.0)
    low = np.full(6, 8.0)
    c = np.array([9.0, 9.0, 10.5, 10.6, 10.7, 10.8])
    v = np.full(6, 100.0)
    data = OHLCVArrays(ts=ts, o=o, h=h, low=low, c=c, v=v)

    params = OpeningSweepV4Params(
        wick_factor=0.0,
        atr_percentile=0.0,
        volume_percentile=0.0,
        session_windows=((0, 24 * 60),),
        trading_timezone="UTC",
        max_trades_per_day=1,
        max_horizon=2,
    )
    strategy = OpeningSweepV4(config=params)

    result = strategy.generate_strategy_result(data)

    assert int(result.signals.sum()) == 1
    assert strategy.daily_entry_counts == {"2024-01-01": 1}
