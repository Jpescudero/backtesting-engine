"""Regime filters for avoiding mean reversion trades."""
from __future__ import annotations

from datetime import time
from typing import Any, Iterable

import pandas as pd

from research.intraday_mean_reversion.study_1.feature_engineering import parse_allowed_time_windows


FilterResult = pd.DataFrame


def _time_in_windows(index: pd.DatetimeIndex, windows: Iterable[tuple[time, time]]) -> pd.Series:
    if not windows:
        return pd.Series(True, index=index)
    mask = pd.Series(False, index=index)
    for start, end in windows:
        mask |= (index.time >= start) & (index.time <= end)
    return mask


def _apply_volatility_filter(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    vol_series = features["realized_vol"]
    threshold_type = str(params.get("vol_threshold_type", "percentile")).lower()
    threshold_value = float(params.get("vol_threshold_value", 0.7))
    mode = str(params.get("vol_regime_mode", "avoid_high_vol")).lower()

    if threshold_type == "percentile":
        cutoff = vol_series.quantile(threshold_value)
    elif threshold_type == "absolute":
        cutoff = threshold_value
    else:
        raise ValueError(f"Invalid vol_threshold_type='{threshold_type}'")

    if mode == "avoid_high_vol":
        return vol_series >= cutoff
    if mode == "avoid_low_vol":
        return vol_series <= cutoff
    raise ValueError(f"Invalid vol_regime_mode='{mode}'")


def _apply_trend_filter(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    trend_strength = features["trend_strength"].abs()
    threshold = float(params.get("trend_threshold", 0.0))
    return trend_strength >= threshold


def _apply_time_filter(events_index: pd.DatetimeIndex, params: dict[str, Any]) -> pd.Series:
    use_filter = bool(params.get("use_time_filter", False))
    if not use_filter:
        return pd.Series(False, index=events_index)

    windows = parse_allowed_time_windows(params.get("allowed_time_windows"))
    in_window = _time_in_windows(events_index, windows)
    return ~in_window


def _apply_shock_filter(features: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    return features["shock_active"].astype(bool)


def evaluate_filters(features: pd.DataFrame, events_index: pd.DatetimeIndex, params: dict[str, Any]) -> FilterResult:
    """Evaluate all filters and return a boolean DataFrame aligned to events."""

    vol_filter = _apply_volatility_filter(features, params)
    trend_filter = _apply_trend_filter(features, params)
    shock_filter = _apply_shock_filter(features, params)
    time_filter = _apply_time_filter(events_index, params)

    aligned = pd.DataFrame(
        {
            "vol_filter": vol_filter.reindex(events_index).fillna(False),
            "trend_filter": trend_filter.reindex(events_index).fillna(False),
            "shock_filter": shock_filter.reindex(events_index).fillna(False),
            "time_filter": time_filter,
        },
        index=events_index,
    )
    return aligned.astype(bool)
