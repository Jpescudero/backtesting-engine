"""Labeling utilities for intraday mean reversion events."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .costs import compute_trade_costs


def label_events(df: pd.DataFrame, events: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Attach outcome labels and costs to detected events.

    Parameters
    ----------
    df : pandas.DataFrame
        Price DataFrame with ``close`` column indexed by datetime.
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

    entry_prices = close.reindex(events.index)
    exit_prices_raw = close.shift(-hold_bars).reindex(events.index)

    r_next = (close.shift(-1) / close - 1.0).reindex(events.index) * events["side"]
    r_H_raw = (exit_prices_raw / entry_prices - 1.0) * events["side"]

    costs = np.vectorize(compute_trade_costs)(params, entry_prices, exit_prices_raw)
    r_H_net = r_H_raw - costs

    labeled = events.copy()
    labeled["entry_price"] = entry_prices
    labeled["exit_price_raw"] = exit_prices_raw
    labeled["r_next"] = r_next
    labeled["is_next_bar_positive"] = r_next > 0
    labeled["r_H_raw"] = r_H_raw
    labeled["is_r_H_positive"] = r_H_raw > 0
    labeled["r_H_net"] = r_H_net
    labeled["is_r_H_net_positive"] = r_H_net > 0

    return labeled.dropna(subset=["r_next", "r_H_raw", "r_H_net"])
