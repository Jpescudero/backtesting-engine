"""Event detection for intraday mean reversion research."""

from __future__ import annotations

import logging
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_DEF_EPS = 1e-12


def _within_session(index: pd.DatetimeIndex, start_time: time, end_time: time) -> pd.Series:
    intraday_times = index.time
    return (intraday_times >= start_time) & (intraday_times <= end_time)


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def detect_mean_reversion_events(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Detect mean reversion events using z-score of lookback returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Price DataFrame indexed by datetime with at least a ``close`` column.
    params : dict[str, Any]
        Parameters containing ``LOOKBACK_MINUTES``, ``ZSCORE_ENTRY``, ``SESSION_START_TIME``, and ``SESSION_END_TIME``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing detected events with metadata and engineered features.
    """

    lookback = int(params["LOOKBACK_MINUTES"])
    z_entry = float(params["ZSCORE_ENTRY"])
    session_start = _parse_time(str(params["SESSION_START_TIME"]))
    session_end = _parse_time(str(params["SESSION_END_TIME"]))

    close = df["close"].astype(float)
    returns_1m = close.pct_change()
    ret_lookback = close / close.shift(lookback) - 1.0
    vol_lookback = returns_1m.rolling(window=lookback, min_periods=lookback).std()
    vol_lookback = vol_lookback.replace(0, np.nan)

    z_score = ret_lookback / (vol_lookback + _DEF_EPS)

    session_mask = _within_session(df.index, session_start, session_end)

    long_mask = z_score <= -z_entry
    short_mask = z_score >= z_entry
    event_mask = (long_mask | short_mask) & session_mask

    events = pd.DataFrame(index=df.index[event_mask])
    events["side"] = np.where(long_mask.loc[event_mask], 1, -1)
    events["z_score"] = z_score.loc[event_mask]
    events["ret_lookback"] = ret_lookback.loc[event_mask]
    events["vol_lookback"] = vol_lookback.loc[event_mask]
    events["hour"] = events.index.hour
    events["day_of_week"] = events.index.dayofweek

    events = events.dropna(subset=["z_score", "ret_lookback", "vol_lookback"])

    logger.debug("Detected %s events", len(events))
    return events
