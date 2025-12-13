from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_mean_reversion.study_2.half_life import (
    TimeBucket,
    compute_half_life_log,
    compute_reference_mean,
    parse_time_buckets,
    reversion_probability_by_horizon,
    summarize_by_bucket,
)
from research.intraday_mean_reversion.study_2.runner import run_study


def test_parse_time_buckets() -> None:
    buckets = parse_time_buckets("OPEN:09:35-10:30,PM:14:00-15:45")
    assert len(buckets) == 2
    assert buckets[0].label == "OPEN"
    assert buckets[0].start.hour == 9 and buckets[0].end.minute == 30


def test_compute_half_life_log_basic() -> None:
    idx = pd.date_range("2022-01-03 09:35", periods=6, freq="min", tz="UTC")
    close = pd.Series([100.0, 101.0, 100.5, 100.25, 100.1, 100.05], index=idx)
    reference_mean = compute_reference_mean(close, lookback_min=2)
    events = pd.DataFrame({"side": 1}, index=idx[1:2])
    buckets = [TimeBucket("OPEN", idx[0].time(), idx[-1].time())]
    params = {"HALF_LIFE_THRESHOLD": 0.5, "MAX_LOOKAHEAD_MIN": 5}

    log = compute_half_life_log(events, close, reference_mean, params, buckets)
    assert len(log) == 1
    assert log.loc[0, "reverted"]
    assert log.loc[0, "half_life_min"] == 1.0


def test_reversion_probability_and_summary() -> None:
    log = pd.DataFrame(
        {
            "time_bucket": ["OPEN", "OPEN", "PM"],
            "reverted": [True, False, True],
            "half_life_min": [5.0, np.nan, 3.0],
        }
    )
    summary = summarize_by_bucket(log)
    probs = reversion_probability_by_horizon(log, horizons=[5, 10])

    open_row = summary[summary["time_bucket"] == "OPEN"].iloc[0]
    assert open_row["n_events"] == 2
    assert open_row["%_no_reversion"] == 0.5
    open_prob = probs[probs["time_bucket"] == "OPEN"].iloc[0]
    assert open_prob["P_rev_5m"] == 0.5


def test_run_study_produces_outputs(tmp_path: Path) -> None:
    idx = pd.date_range("2022-01-03 15:35", periods=90, freq="min", tz="UTC")
    close = pd.Series(100.0 + np.linspace(0, 1, len(idx)) + np.sin(np.arange(len(idx)) / 10), index=idx)
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
        "HALF_LIFE_THRESHOLD": 0.5,
        "MAX_LOOKAHEAD_MIN": 10,
        "TIME_BUCKETS": "PM:15:30-16:30",
        "vol_window_min": 5,
        "trend_window_min": 5,
        "shock_window_min": 5,
    }

    run_dir = run_study(params, tmp_path, df=df, run_id="test_run_s2")
    assert (run_dir / "event_half_life_log.csv").exists()
    assert (run_dir / "half_life_by_time_bucket.csv").exists()
    assert (run_dir / "reversion_probability_by_horizon.csv").exists()
