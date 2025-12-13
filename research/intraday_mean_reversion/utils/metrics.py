"""Metrics computation for labeled intraday mean reversion events."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def _probability(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else float("nan")


def proportion_ci_wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute a Wilson score interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    alpha : float, optional
        Significance level; ``alpha=0.05`` returns a 95% confidence interval.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the Wilson interval.
    """

    if n == 0:
        return (float("nan"), float("nan"))

    z = NormalDist().inv_cdf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = max(0.0, min(1.0, (centre - margin) / denom))
    upper = max(0.0, min(1.0, (centre + margin) / denom))
    return (lower, upper)


def _sharpe_ratio(series: pd.Series, periods_per_year: int) -> float:
    if series.empty:
        return float("nan")
    std = series.std(ddof=0)
    if std in {0.0, float("nan")}:
        return float("nan")
    return float(series.mean() / std * math.sqrt(periods_per_year))


def compute_daily_pnl(labeled_events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate net returns by entry date for interpretable Sharpe ratio."""

    if labeled_events.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "n_trades"])

    entry_timestamps = labeled_events.get("entry_timestamp", labeled_events.index)
    entry_dates = pd.DatetimeIndex(pd.to_datetime(entry_timestamps)).normalize()
    daily = labeled_events.assign(entry_date=entry_dates).groupby("entry_date", as_index=False).agg(
        daily_pnl=("r_H_net", "sum"),
        n_trades=("r_H_net", "size"),
    )
    daily["date"] = pd.to_datetime(daily["entry_date"])
    return daily[["date", "daily_pnl", "n_trades"]]


def compute_zscore_bin_stats(
    labeled_events: pd.DataFrame, bins: int, loss_tail_x: float, alpha: float = 0.05
) -> pd.DataFrame:
    """Compute performance statistics by z-score bin.

    Parameters
    ----------
    labeled_events : pd.DataFrame
        Labeled events containing ``z_score`` and return columns.
    bins : int
        Number of bins to create along the z-score axis.
    loss_tail_x : float
        Threshold used to compute the probability of losses below ``-x``.
    alpha : float, optional
        Significance level for the Wilson confidence interval. ``0.05`` yields a 95% CI.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per z-score bin including success probabilities and
        return distribution statistics.
    """

    if labeled_events.empty:
        return pd.DataFrame(
            columns=
            [
                "z_bin_left",
                "z_bin_right",
                "n",
                "p_hat",
                "ci_low",
                "ci_high",
                "E_r_H_net",
                "median_r_H_net",
                "q05",
                "q95",
                "p_loss_below_x",
            ]
        )

    z_scores = labeled_events["z_score"]
    z_min, z_max = float(z_scores.min()), float(z_scores.max())
    if z_min == z_max:
        z_min -= 1e-6
        z_max += 1e-6
    bin_edges = np.linspace(z_min, z_max, bins + 1)
    categories = pd.cut(z_scores, bins=bin_edges, include_lowest=True)
    grouped = labeled_events.groupby(categories, observed=False)

    records = []
    for interval, group in grouped:
        if interval is None:
            continue
        n = len(group)
        k = int(group["is_r_H_net_positive"].sum())
        p_hat = k / n if n else float("nan")
        ci_low, ci_high = proportion_ci_wilson(k, n, alpha)
        loss_prob = float((group["r_H_net"] < -loss_tail_x).mean()) if n else float("nan")
        quantiles = (
            group["r_H_net"].quantile([0.05, 0.5, 0.95])
            if n
            else pd.Series([float("nan")] * 3, index=[0.05, 0.5, 0.95])
        )
        records.append(
            {
                "z_bin_left": interval.left,
                "z_bin_right": interval.right,
                "n": n,
                "p_hat": p_hat,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "E_r_H_net": float(group["r_H_net"].mean()),
                "E_r_H_raw": float(group["r_H_raw"].mean()),
                "median_r_H_net": float(quantiles.loc[0.5]),
                "q05": float(quantiles.loc[0.05]),
                "q95": float(quantiles.loc[0.95]),
                "p_loss_below_x": loss_prob,
            }
        )

    bin_stats = pd.DataFrame(records).sort_values("z_bin_left").reset_index(drop=True)
    if not bin_stats.empty:
        bin_stats["z_bin_center"] = (bin_stats["z_bin_left"] + bin_stats["z_bin_right"]) / 2

    return bin_stats


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
            "n_days": 0,
            "p_next_bar_pos": float("nan"),
            "p_H_raw_pos": float("nan"),
            "p_H_net_pos": float("nan"),
            "E_r_H_raw": float("nan"),
            "E_r_H_gross": float("nan"),
            "E_r_H_net": float("nan"),
            "E_pnl_gross": float("nan"),
            "E_pnl_net": float("nan"),
            "median_r_H_net": float("nan"),
            "std_r_H_net": float("nan"),
            "pct5_r_H_net": float("nan"),
            "pct25_r_H_net": float("nan"),
            "pct75_r_H_net": float("nan"),
            "pct95_r_H_net": float("nan"),
            "avg_cost_total_return": float("nan"),
            "E_r_H_raw_over_cost": float("nan"),
            "sharpe_per_trade": float("nan"),
            "sharpe_net": float("nan"),
            "sharpe_daily": float("nan"),
        }

    n_events = len(labeled_events)
    n_longs = int((labeled_events["side"] == 1).sum())
    n_shorts = int((labeled_events["side"] == -1).sum())

    daily_pnl = compute_daily_pnl(labeled_events)

    metrics: dict[str, Any] = {
        "n_events": n_events,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "n_days": int(len(daily_pnl)),
    }

    metrics["p_next_bar_pos"] = _probability(labeled_events["is_next_bar_positive"])
    metrics["p_H_raw_pos"] = _probability(labeled_events["is_r_H_positive"])
    metrics["p_H_net_pos"] = _probability(labeled_events["is_r_H_net_positive"])

    metrics["E_r_H_raw"] = float(labeled_events["r_H_raw"].mean())
    metrics["E_r_H_gross"] = metrics["E_r_H_raw"]
    metrics["E_r_H_net"] = float(labeled_events["r_H_net"].mean())
    metrics["E_pnl_gross"] = float(labeled_events.get("pnl_gross", pd.Series(dtype=float)).mean())
    metrics["E_pnl_net"] = float(labeled_events.get("pnl_net", pd.Series(dtype=float)).mean())
    metrics["median_r_H_net"] = float(labeled_events["r_H_net"].median())
    metrics["std_r_H_net"] = float(labeled_events["r_H_net"].std(ddof=0))

    percentiles = labeled_events["r_H_net"].quantile([0.05, 0.25, 0.75, 0.95])
    metrics["pct5_r_H_net"] = float(percentiles.loc[0.05])
    metrics["pct25_r_H_net"] = float(percentiles.loc[0.25])
    metrics["pct75_r_H_net"] = float(percentiles.loc[0.75])
    metrics["pct95_r_H_net"] = float(percentiles.loc[0.95])

    cost_total = labeled_events.get("cost_return")
    avg_cost_return = float(cost_total.mean()) if cost_total is not None else float("nan")
    metrics["avg_cost_total_return"] = avg_cost_return
    metrics["E_r_H_raw_over_cost"] = (
        metrics["E_r_H_raw"] / avg_cost_return if avg_cost_return not in {0.0, float("nan")} else float("nan")
    )

    std_net = metrics["std_r_H_net"]
    sharpe_per_trade = (
        metrics["E_r_H_net"] / std_net * math.sqrt(n_events)
        if std_net not in {0.0, float("nan")} else float("nan")
    )
    metrics["sharpe_per_trade"] = sharpe_per_trade
    metrics["sharpe_net"] = sharpe_per_trade
    metrics["sharpe_daily"] = _sharpe_ratio(daily_pnl["daily_pnl"], periods_per_year=252)

    return metrics
