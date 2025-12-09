"""Tests for trade level statistics reporting."""

from __future__ import annotations

import pandas as pd

from src.analytics.metrics import trade_level_stats


def test_trade_level_stats_compute_distances_and_breakeven() -> None:
    trades = pd.DataFrame(
        {
            "entry_price": [100.0, 120.0, 150.0],
            "stop_loss": [95.0, 130.0, 150.0],
            "take_profit": [110.0, 110.0, 165.0],
            "qty": [1.0, -1.0, 1.0],
            "pnl": [0.0, -2.0, 30.0],
        }
    )

    stats = trade_level_stats(trades)

    assert stats["breakeven_count"] == 1
    assert stats["breakeven_rate"] == 1 / 3

    sl_pct = stats["sl_distance_pct"]
    tp_pct = stats["tp_distance_pct"]

    assert sl_pct["mean"] > 0
    assert sl_pct["count"] == 3
    assert tp_pct["count"] == 3

    # Long and short distances should be positive after normalization
    assert stats["sl_distance_abs"]["max"] > 0
    assert stats["tp_distance_abs"]["min"] > 0
