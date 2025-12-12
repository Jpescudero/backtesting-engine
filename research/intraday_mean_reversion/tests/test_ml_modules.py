"""Tests for ML meta-labeling utilities."""

from __future__ import annotations

from datetime import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.intraday_mean_reversion.utils.ml_cv import generate_walk_forward_splits
from research.intraday_mean_reversion.utils.ml_features import build_feature_matrix
from research.intraday_mean_reversion.utils.ml_reporting import save_predictions, summarize_uplift


def _sample_price_data() -> pd.DataFrame:
    index = pd.date_range("2021-01-04 09:30", periods=300, freq="min")
    close = pd.Series(np.linspace(100, 101.5, len(index)), index=index)
    return pd.DataFrame({"close": close})


def _sample_events(index: pd.DatetimeIndex) -> pd.DataFrame:
    events_idx = index[50:150:10]
    events = pd.DataFrame(
        {
            "z_score": np.linspace(-2.0, 2.0, len(events_idx)),
            "vol_lookback": np.linspace(0.01, 0.02, len(events_idx)),
            "ret_lookback": np.linspace(-0.001, 0.001, len(events_idx)),
            "r_H_net": np.linspace(-0.0005, 0.001, len(events_idx)),
        },
        index=events_idx,
    )
    events["is_r_H_net_positive"] = events["r_H_net"] > 0
    events["entry_timestamp"] = events.index
    return events


def test_build_feature_matrix_no_nan():
    df = _sample_price_data()
    events = _sample_events(df.index)
    params = {"ML_LABEL": "is_r_H_net_positive"}

    X, y = build_feature_matrix(df, events, params)

    assert not X.empty
    assert y.index.equals(X.index)
    assert X.isna().sum().sum() == 0
    assert {"z_score", "vol_lookback"}.issubset(X.columns)


def test_walk_forward_split_generation_respects_bounds():
    dates = pd.date_range("2018-01-02", "2022-12-31", freq="90D")
    events = pd.DataFrame({"dummy": range(len(dates))}, index=dates)
    splits = generate_walk_forward_splits(
        events,
        train_start_year=2018,
        train_end_year=2020,
        test_start_year=2021,
        test_end_year=2022,
        fold_years=1,
        min_train_days=1,
        embargo_days=0,
    )

    assert splits, "At least one fold should be created"
    assert splits[0].train_idx.min().year == 2018
    assert splits[0].test_idx.min().year == 2021
    assert all(split.test_idx.max().year <= 2022 for split in splits)


def test_summarize_uplift_and_save_predictions(tmp_path):
    index = pd.to_datetime([
        datetime(2021, 1, 4, 9, 30),
        datetime(2021, 1, 4, 9, 40),
        datetime(2021, 1, 5, 9, 30),
    ])
    labeled = pd.DataFrame(
        {
            "r_H_net": [0.001, -0.0005, 0.002],
            "is_r_H_net_positive": [True, False, True],
            "entry_timestamp": index,
            "side": [1, -1, 1],
            "z_score": [1.5, -1.2, 2.0],
        },
        index=index,
    )
    probs = pd.Series([0.6, 0.4, 0.8], index=index)

    summary, baseline_daily, ml_daily = summarize_uplift(labeled, probs, threshold=0.55)

    assert summary.loc[summary["variant"] == "baseline", "n_trades"].iloc[0] == 3
    assert summary.loc[summary["variant"] == "ml_filtered", "n_trades"].iloc[0] == 2
    assert not baseline_daily.empty

    save_predictions(tmp_path, labeled, probs, threshold=0.55)
    assert (tmp_path / "ml_predictions.csv").exists()
    assert (tmp_path / "ml_uplift_summary.csv").exists()
    assert (tmp_path / "daily_pnl_ml.csv").exists()
    assert (tmp_path / "plots" / "uplift_thresholds.png").exists()
