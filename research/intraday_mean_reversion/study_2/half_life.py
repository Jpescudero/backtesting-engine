"""Half-life computation for intraday mean reversion events."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeBucket:
    """Represents a named intraday time window."""

    label: str
    start: time
    end: time


def parse_time_buckets(value: str | Iterable[str] | Iterable[tuple[str, str, str]]) -> list[TimeBucket]:
    """Parse time bucket definitions into ``TimeBucket`` objects.

    Parameters
    ----------
    value : str | Iterable[str] | Iterable[tuple[str, str, str]]
        Bucket definitions. Strings should use the format ``NAME:HH:MM-HH:MM``
        separated by commas. Alternatively, an iterable of strings in the same
        format or tuples ``(name, start, end)`` can be provided.

    Returns
    -------
    list[TimeBucket]
        Parsed time buckets preserving input order.

    Raises
    ------
    ValueError
        If any bucket definition is malformed.
    """

    if isinstance(value, str):
        raw_tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        raw_tokens = list(value)

    buckets: list[TimeBucket] = []
    for token in raw_tokens:
        if isinstance(token, tuple) and len(token) == 3:
            label, start_raw, end_raw = token
        elif isinstance(token, str):
            if ":" not in token or "-" not in token:
                raise ValueError(
                    "Invalid bucket definition '{token}'. Expected NAME:HH:MM-HH:MM"
                )
            label, range_token = token.split(":", maxsplit=1)
            start_raw, end_raw = range_token.split("-", maxsplit=1)
        else:
            raise ValueError(f"Unsupported time bucket token: {token!r}")

        start = _parse_hhmm(str(start_raw).strip())
        end = _parse_hhmm(str(end_raw).strip())
        buckets.append(TimeBucket(label=label.strip(), start=start, end=end))

    return buckets


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def assign_time_bucket(index: pd.DatetimeIndex, buckets: list[TimeBucket]) -> pd.Series:
    """Assign each timestamp to a configured time bucket.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to classify.
    buckets : list[TimeBucket]
        Parsed time buckets.

    Returns
    -------
    pd.Series
        Series of bucket labels indexed by ``index``. Timestamps that do not
        fall into any bucket are labeled ``"UNBUCKETED"``.
    """

    if not buckets:
        return pd.Series(["UNBUCKETED"] * len(index), index=index)

    times = pd.Series(index.time, index=index)
    labels = pd.Series("UNBUCKETED", index=index, dtype=object)
    for bucket in buckets:
        in_bucket = (times >= bucket.start) & (times < bucket.end)
        labels.loc[in_bucket] = bucket.label
    return labels


def compute_reference_mean(close: pd.Series, lookback_min: int) -> pd.Series:
    """Compute the rolling mean used as reference for half-life deviations."""

    return close.astype(float).rolling(window=lookback_min, min_periods=lookback_min).mean()


def _first_reversion(
    close: pd.Series, ref_value: float, start_pos: int, max_lookahead: int, threshold: float
) -> tuple[float, bool]:
    initial_price = float(close.iloc[start_pos])
    initial_deviation = initial_price - ref_value
    if not np.isfinite(initial_deviation) or initial_deviation == 0.0:
        return np.nan, False

    target = abs(initial_deviation) * threshold
    for offset in range(1, max_lookahead + 1):
        if start_pos + offset >= len(close):
            break
        deviation = float(close.iloc[start_pos + offset]) - ref_value
        if abs(deviation) <= target:
            return float(offset), True
    return np.nan, False


def compute_half_life_log(
    events: pd.DataFrame,
    close: pd.Series,
    reference_mean: pd.Series,
    params: dict[str, float | int | str],
    buckets: list[TimeBucket],
) -> pd.DataFrame:
    """Compute half-life metrics for each event.

    Parameters
    ----------
    events : pd.DataFrame
        Mean reversion events with an index aligned to ``close`` and a ``side``
        column.
    close : pd.Series
        Close price series.
    reference_mean : pd.Series
        Pre-computed reference mean series aligned to ``close``.
    params : dict[str, float | int | str]
        Study parameters containing ``HALF_LIFE_THRESHOLD`` and
        ``MAX_LOOKAHEAD_MIN``.
    buckets : list[TimeBucket]
        Parsed time buckets used to segment events.

    Returns
    -------
    pd.DataFrame
        Event-level log containing timestamp, bucket, side, initial deviation,
        half-life in minutes and whether reversion occurred within the
        lookahead window.
    """

    if events.empty:
        return pd.DataFrame(
            columns=["timestamp", "time_bucket", "side", "initial_deviation", "half_life_min", "reverted"]
        )

    threshold = float(params.get("HALF_LIFE_THRESHOLD", 0.5))
    max_lookahead = int(params.get("MAX_LOOKAHEAD_MIN", 60))

    close = close.astype(float)
    reference_mean = reference_mean.astype(float)

    timestamps = events.index
    positions = close.index.get_indexer(timestamps)
    bucket_labels = assign_time_bucket(timestamps, buckets)

    records: list[dict[str, object]] = []
    for ts, pos in zip(timestamps, positions):
        if pos < 0:
            ref_value = np.nan
            initial_dev = np.nan
            half_life, reverted = np.nan, False
        else:
            ref_value = float(reference_mean.iloc[pos])
            initial_dev = float(close.iloc[pos] - ref_value)
            half_life, reverted = _first_reversion(close, ref_value, pos, max_lookahead, threshold)
        records.append(
            {
                "timestamp": ts,
                "time_bucket": bucket_labels.loc[ts],
                "side": events.loc[ts, "side"],
                "initial_deviation": initial_dev,
                "half_life_min": half_life,
                "reverted": bool(reverted),
            }
        )

    return pd.DataFrame(records)


def summarize_by_bucket(log: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-life statistics by time bucket."""

    if log.empty:
        return pd.DataFrame(
            columns=[
                "time_bucket",
                "n_events",
                "mean_half_life_min",
                "median_half_life_min",
                "p75_half_life_min",
                "p90_half_life_min",
                "%_no_reversion",
            ]
        )

    grouped = log.groupby("time_bucket", dropna=False)
    records = []
    for bucket, group in grouped:
        half_life = group.loc[group["reverted"], "half_life_min"].dropna()
        record = {
            "time_bucket": bucket,
            "n_events": int(len(group)),
            "mean_half_life_min": float(half_life.mean()) if not half_life.empty else float("nan"),
            "median_half_life_min": float(half_life.median()) if not half_life.empty else float("nan"),
            "p75_half_life_min": float(half_life.quantile(0.75)) if not half_life.empty else float("nan"),
            "p90_half_life_min": float(half_life.quantile(0.90)) if not half_life.empty else float("nan"),
            "%_no_reversion": float((~group["reverted"]).mean()) if len(group) else float("nan"),
        }
        records.append(record)
    return pd.DataFrame(records)


def reversion_probability_by_horizon(log: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Compute reversion probabilities by time bucket and fixed horizons."""

    horizons_list = list(horizons)
    if log.empty:
        columns = [f"P_rev_{h}m" for h in horizons_list]
        return pd.DataFrame(columns=["time_bucket", *columns])

    grouped = log.groupby("time_bucket", dropna=False)
    records = []
    for bucket, group in grouped:
        entry = {"time_bucket": bucket}
        for horizon in horizons_list:
            prob = float((group["half_life_min"] <= horizon).fillna(False).mean())
            entry[f"P_rev_{horizon}m"] = prob
        records.append(entry)
    return pd.DataFrame(records)
