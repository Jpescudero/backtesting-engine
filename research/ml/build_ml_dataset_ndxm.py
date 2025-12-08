"""Build ML-ready dataset for NDXm 1m bars."""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd

from src.config.paths import DATA_DIR, ensure_directories_exist
from src.data.feeds import NPZOHLCVFeed


# -----------------------------
# 1. Feature engineering
# -----------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bar features for a 1m OHLCV DataFrame.

    The input DataFrame must have columns ["open", "high", "low", "close", "volume"]
    and a timezone-aware datetime index.
    """

    features = pd.DataFrame(index=df.index)

    close = df["close"]
    ret1 = np.log(close).diff(1)
    ret5 = np.log(close).diff(5)
    ret15 = np.log(close).diff(15)

    features["ret_1"] = ret1
    features["ret_5"] = ret5
    features["ret_15"] = ret15

    # Rolling volatility of 1-bar log returns
    features["vol_5"] = ret1.rolling(5, min_periods=5).std()
    features["vol_15"] = ret1.rolling(15, min_periods=15).std()

    # Relative volume (vs. same-day mean)
    daily_mean_volume = df["volume"].groupby(df.index.date).transform("mean")
    features["vol_rel"] = df["volume"] / daily_mean_volume.replace(0, np.nan)

    # Time-of-day cyclical encoding
    minutes_in_day = df.index.hour * 60 + df.index.minute
    max_minutes = 24 * 60
    features["sin_time"] = np.sin(2 * np.pi * minutes_in_day / max_minutes)
    features["cos_time"] = np.cos(2 * np.pi * minutes_in_day / max_minutes)

    return features


# -----------------------------
# 2. Labels
# -----------------------------
def build_labels(df: pd.DataFrame, horizon: int = 10, thr: float = 0.0) -> pd.Series:
    """Directional label: -1, 0, +1 based on forward cumulative log-return.

    Args:
        df: Input OHLCV DataFrame with a "close" column.
        horizon: Number of bars to look ahead.
        thr: Absolute return threshold for the neutral class.
    """

    close = df["close"]
    fwd_ret = np.log(close).shift(-horizon) - np.log(close)
    y = pd.Series(index=df.index, dtype=np.int8)

    y[fwd_ret > thr] = 1
    y[fwd_ret < -thr] = -1
    y[(fwd_ret >= -thr) & (fwd_ret <= thr)] = 0
    return y


# -----------------------------
# 3. Load OHLCV bars from NPZ feed
# -----------------------------
def load_ndxm_bars_1m(years: Iterable[int]) -> pd.DataFrame:
    """Load 1m bars for NDXm from NPZ storage and return a pandas DataFrame."""

    feed = NPZOHLCVFeed(symbol="NDXm", timeframe="1m")
    data = feed.load_years(list(years))

    dt_index = pd.to_datetime(data.ts, unit="ns", utc=True)
    df = pd.DataFrame(
        {
            "open": data.o,
            "high": data.h,
            "low": data.low,
            "close": data.c,
            "volume": data.v,
        },
        index=dt_index,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML dataset for NDXm 1m bars")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--thr", type=float, default=0.0)
    args = parser.parse_args()

    ensure_directories_exist()

    years = list(range(args.start_year, args.end_year + 1))

    df = load_ndxm_bars_1m(years).sort_index()
    features = build_features(df)
    labels = build_labels(df, horizon=args.horizon, thr=args.thr)

    dataset = pd.concat([features, labels.rename("label")], axis=1)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()

    out_path = (
        DATA_DIR
        / "ml"
        / f"ndxm_directional_{args.start_year}-{args.end_year}_h{args.horizon}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_parquet(out_path)

    print(f"Saved ML dataset to {out_path} with {len(dataset):,} rows and {dataset.shape[1]} columns")


if __name__ == "__main__":
    main()
