"""Tests for intraday mean reversion metrics utilities."""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from research.intraday_mean_reversion.utils.metrics import compute_daily_pnl


def test_compute_daily_pnl_normalizes_entry_timestamp_column() -> None:
    """Ensure entry timestamps in a column are normalized per date."""

    labeled_events = pd.DataFrame(
        {
            "entry_timestamp": pd.to_datetime(
                ["2023-01-01 10:00", "2023-01-01 11:00", "2023-01-02 09:30"]
            ),
            "r_H_net": [0.1, -0.05, 0.2],
        }
    )

    result = compute_daily_pnl(labeled_events)

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "daily_pnl": [0.05, 0.2],
            "n_trades": [2, 1],
        }
    )

    pdt.assert_frame_equal(result.reset_index(drop=True), expected)


def test_compute_daily_pnl_uses_index_when_column_missing() -> None:
    """Fall back to the index when ``entry_timestamp`` column is absent."""

    labeled_events = pd.DataFrame(
        {"r_H_net": [0.3, -0.1]},
        index=pd.to_datetime(["2023-02-01 09:30", "2023-02-01 15:55"]),
    )

    result = compute_daily_pnl(labeled_events)

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-02-01"]),
            "daily_pnl": [0.2],
            "n_trades": [2],
        }
    )

    pdt.assert_frame_equal(result.reset_index(drop=True), expected)
