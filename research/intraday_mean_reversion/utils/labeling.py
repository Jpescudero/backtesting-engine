"""Labeling utilities for intraday mean reversion events."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .costs import compute_trade_cost_breakdown


def label_events(df: pd.DataFrame, events: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Attach outcome labels and costs to detected events.

    Parameters
    ----------
    df : pandas.DataFrame
        Price DataFrame with ``close`` column indexed by datetime. ``open`` is
        used when available to model t+1 entry prices realistically.
    events : pandas.DataFrame
        Events DataFrame returned by ``detect_mean_reversion_events``.
    params : dict[str, Any]
        Parameter dictionary containing ``HOLD_TIME_BARS`` and cost fields.

    Returns
    -------
    pandas.DataFrame
        Labeled events with raw and net returns plus helper fields.
    """

    if events.empty:
        return events.copy()

    hold_bars = int(params.get("HOLD_TIME_BARS", 1))
    close = df["close"].astype(float)
    open_price = df.get("open")
    open_series = open_price.astype(float) if open_price is not None else None

    entry_prices_close = close.shift(-1).reindex(events.index)
    entry_prices_open = open_series.shift(-1).reindex(events.index) if open_series is not None else None
    entry_prices = entry_prices_open.combine_first(entry_prices_close) if entry_prices_open is not None else entry_prices_close

    exit_shift = hold_bars + 1
    exit_prices_raw = close.shift(-exit_shift).reindex(events.index)

    next_close_after_entry = close.shift(-2).reindex(events.index)
    r_next = (next_close_after_entry / entry_prices - 1.0) * events["side"]
    r_H_raw = (exit_prices_raw / entry_prices - 1.0) * events["side"]

    cost_breakdowns = [compute_trade_cost_breakdown(params, e, x) for e, x in zip(entry_prices, exit_prices_raw)]
    cost_df = pd.DataFrame(cost_breakdowns, index=events.index)
    r_H_net = r_H_raw - cost_df["cost_total_return"]

    labeled = events.copy()
    labeled["entry_timestamp"] = df.index.to_series().shift(-1).reindex(events.index)
    labeled["entry_price"] = entry_prices
    labeled["exit_price_raw"] = exit_prices_raw
    labeled["r_next"] = r_next
    labeled["is_next_bar_positive"] = r_next > 0
    labeled["r_H_raw"] = r_H_raw
    labeled["is_r_H_positive"] = r_H_raw > 0
    labeled["r_H_net"] = r_H_net
    labeled["is_r_H_net_positive"] = r_H_net > 0
    labeled = pd.concat([labeled, cost_df], axis=1)

    return labeled.dropna(subset=["r_next", "r_H_raw", "r_H_net"])
