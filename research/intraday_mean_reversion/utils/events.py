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


def _apply_cooldown_and_reset(
    events: pd.DataFrame, z_score: pd.Series, cooldown_bars: int, z_reset: float, full_index: pd.Index
) -> pd.DataFrame:
    """Filter events to avoid overlapping triggers via cooldown and z-score reset.

    Parameters
    ----------
    events : pd.DataFrame
        Candidate events with index aligned to ``z_score``.
    z_score : pd.Series
        Full z-score series to evaluate reset conditions between events.
    cooldown_bars : int
        Number of bars to wait after an accepted event before allowing another.
    z_reset : float
        Absolute z-score threshold that must be crossed before allowing another event.
    full_index : pd.Index
        Index of the original price DataFrame used to map positions to bar offsets.

    Returns
    -------
    pd.DataFrame
        Filtered events DataFrame respecting cooldown/reset constraints.
    """

    if events.empty:
        return events

    event_positions = pd.Index(full_index).get_indexer(events.index)
    reset_positions = np.flatnonzero(np.abs(z_score.values) < z_reset) if z_reset > 0 else np.array([], dtype=int)

    accepted_indices: list[int] = []
    last_accept_pos: int | None = None
    waiting_for_reset = False

    for pos, idx in zip(event_positions, events.index):
        if last_accept_pos is not None and cooldown_bars > 0:
            if pos - last_accept_pos < cooldown_bars:
                continue

        if waiting_for_reset and last_accept_pos is not None:
            if reset_positions.size == 0:
                continue
            search_idx = np.searchsorted(reset_positions, last_accept_pos + 1)
            if search_idx >= reset_positions.size:
                continue
            next_reset_pos = reset_positions[search_idx]
            if next_reset_pos >= pos:
                continue
            waiting_for_reset = False

        accepted_indices.append(idx)
        last_accept_pos = pos
        waiting_for_reset = z_reset > 0

    filtered = events.loc[accepted_indices]
    logger.debug(
        "Applied cooldown/reset: kept %s of %s events", len(filtered), len(events)
    )
    return filtered


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
    cooldown_bars = int(params.get("COOLDOWN_BARS", 0))
    z_reset = float(params.get("Z_RESET", 0.0))
    eps = float(params.get("Z_EPS", _DEF_EPS))

    close = df["close"].astype(float)
    logret_1m = np.log(close / close.shift(1))
    ret_lookback = np.log(close / close.shift(lookback))
    vol_lookback = (
        logret_1m.rolling(window=lookback, min_periods=lookback).std().mul(np.sqrt(lookback))
    )
    vol_lookback = vol_lookback.replace(0, np.nan)

    z_score = ret_lookback / (vol_lookback + eps)

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

    mode = str(params.get("MODE", "both")).lower()
    z_min_short = float(params.get("Z_MIN_SHORT", params.get("ZSCORE_ENTRY", z_entry)))
    z_min_long = float(params.get("Z_MIN_LONG", params.get("ZSCORE_ENTRY", z_entry)))

    if mode == "fade_up_only":
        events = events[(events["side"] == -1) & (events["z_score"] >= z_min_short)]
    elif mode == "fade_down_only":
        events = events[(events["side"] == 1) & (events["z_score"] <= -z_min_long)]
    elif mode != "both":
        raise ValueError(f"Unsupported MODE='{mode}'")

    events = _apply_cooldown_and_reset(events, z_score, cooldown_bars, z_reset, df.index)

    logger.debug("Detected %s events", len(events))
    return events
