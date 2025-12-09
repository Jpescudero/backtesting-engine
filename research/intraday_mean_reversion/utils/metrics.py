"""Metrics computation for labeled intraday mean reversion events."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _probability(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else float("nan")


def compute_metrics(labeled_events: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate metrics for labeled events.

    Parameters
    ----------
    labeled_events : pandas.DataFrame
        Output of ``label_events`` containing return columns and side.

    Returns
    -------
    dict[str, Any]
        Dictionary of aggregated metrics.
    """

    if labeled_events.empty:
        return {
            "n_events": 0,
            "n_longs": 0,
            "n_shorts": 0,
            "p_next_bar_pos": float("nan"),
            "p_H_raw_pos": float("nan"),
            "p_H_net_pos": float("nan"),
            "E_r_H_raw": float("nan"),
            "E_r_H_net": float("nan"),
            "median_r_H_net": float("nan"),
            "std_r_H_net": float("nan"),
            "pct5_r_H_net": float("nan"),
            "pct25_r_H_net": float("nan"),
            "pct75_r_H_net": float("nan"),
            "pct95_r_H_net": float("nan"),
            "sharpe_net": float("nan"),
        }

    n_events = len(labeled_events)
    n_longs = int((labeled_events["side"] == 1).sum())
    n_shorts = int((labeled_events["side"] == -1).sum())

    metrics: dict[str, Any] = {
        "n_events": n_events,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
    }

    metrics["p_next_bar_pos"] = _probability(labeled_events["is_next_bar_positive"])
    metrics["p_H_raw_pos"] = _probability(labeled_events["is_r_H_positive"])
    metrics["p_H_net_pos"] = _probability(labeled_events["is_r_H_net_positive"])

    metrics["E_r_H_raw"] = float(labeled_events["r_H_raw"].mean())
    metrics["E_r_H_net"] = float(labeled_events["r_H_net"].mean())
    metrics["median_r_H_net"] = float(labeled_events["r_H_net"].median())
    metrics["std_r_H_net"] = float(labeled_events["r_H_net"].std(ddof=0))

    percentiles = labeled_events["r_H_net"].quantile([0.05, 0.25, 0.75, 0.95])
    metrics["pct5_r_H_net"] = float(percentiles.loc[0.05])
    metrics["pct25_r_H_net"] = float(percentiles.loc[0.25])
    metrics["pct75_r_H_net"] = float(percentiles.loc[0.75])
    metrics["pct95_r_H_net"] = float(percentiles.loc[0.95])

    avg_cost = float((labeled_events["r_H_raw"] - labeled_events["r_H_net"]).mean())
    metrics["E_r_H_raw_over_cost"] = (
        metrics["E_r_H_raw"] / avg_cost if avg_cost not in {0.0, float("nan")} else float("nan")
    )

    std_net = metrics["std_r_H_net"]
    metrics["sharpe_net"] = (
        metrics["E_r_H_net"] / std_net * math.sqrt(n_events)
        if std_net not in {0.0, float("nan")} else float("nan")
    )

    return metrics
