"""Unit tests for threshold recommendation utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.intraday_mean_reversion.utils.thresholding import recommend_thresholds_from_bins


def _sample_bin_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z_bin_left": [-4.0, -2.0, 2.0, 3.0],
            "z_bin_right": [-3.0, -1.0, 2.5, 3.5],
            "n": [60, 30, 80, 55],
            "p_hat": [0.55, 0.35, 0.6, 0.7],
            "ci_low": [0.45, 0.20, 0.5, 0.6],
            "ci_high": [0.65, 0.50, 0.7, 0.8],
            "E_r_H_net": [0.0005, -0.0003, 0.001, 0.002],
            "median_r_H_net": [0.0004, -0.0002, 0.0009, 0.0015],
            "q05": [-0.001, -0.002, -0.0015, -0.001],
            "q95": [0.002, 0.001, 0.0025, 0.003],
            "p_loss_below_x": [0.1, 0.5, 0.2, 0.15],
            "z_bin_center": [-3.5, -1.5, 2.25, 3.25],
        }
    )


def test_recommendation_prefers_first_positive_bin() -> None:
    params = {
        "MODE": "fade_up_only",
        "MIN_EVENTS_PER_BIN": 50,
        "MIN_CI_LOW": 0.4,
        "MIN_EXPECTANCY_NET": 0.0,
        "MAX_TAIL_LOSS": 0.4,
    }

    recommendation = recommend_thresholds_from_bins(_sample_bin_stats(), params)

    assert recommendation.recommended_z_min_short == 2.25
    assert recommendation.accepted_bins["direction"].tolist() == ["fade_up", "fade_up"]
    assert recommendation.accepted_bins["is_frontier"].any()


def test_recommendation_handles_fade_down_mode() -> None:
    params = {
        "MODE": "fade_down_only",
        "MIN_EVENTS_PER_BIN": 50,
        "MIN_CI_LOW": 0.4,
        "MIN_EXPECTANCY_NET": 0.0,
        "MAX_TAIL_LOSS": 0.4,
    }

    recommendation = recommend_thresholds_from_bins(_sample_bin_stats(), params)

    assert recommendation.recommended_z_min_long == 3.5
    assert all(recommendation.accepted_bins["direction"] == "fade_down")


def test_recommendation_returns_none_when_no_bins_pass() -> None:
    params = {
        "MODE": "fade_up_only",
        "MIN_EVENTS_PER_BIN": 100,
        "MIN_CI_LOW": 0.6,
        "MIN_EXPECTANCY_NET": 0.01,
        "MAX_TAIL_LOSS": 0.1,
    }

    recommendation = recommend_thresholds_from_bins(_sample_bin_stats(), params)

    assert recommendation.recommended_z_min_short is None
    assert recommendation.accepted_bins.empty
