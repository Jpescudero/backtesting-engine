"""
V1 — Basic Opening Sweep Study
------------------------------
Baseline version using forward returns and StrategyBarridaApertura.
Compatible with optimizer via preload_data() and run_with_params().
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from src.data.feeds import NPZOHLCVFeed, OHLCVArrays
from src.config.paths import REPORTS_DIR, ensure_directories_exist
from src.strategies.barrida_apertura import StrategyBarridaApertura


SessionWindow = Tuple[str, str]


@dataclass
class SweepStudyConfig:
    index: str
    start_year: int
    end_year: int
    horizon: int
    timezone: str


# ============================================================
# DATA LOADING
# ============================================================

def load_index_sessions(symbol: str, years: Iterable[int], windows, tz: str):
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_years(list(years))
    idx_utc = pd.to_datetime(data.ts, unit="ns", utc=True)

    df = pd.DataFrame(
        {
            "open": data.o,
            "high": data.h,
            "low": data.low,
            "close": data.c,
            "volume": data.v,
        },
        index=idx_utc,
    ).sort_index()

    idx_local = df.index.tz_convert(tz)
    minutes = idx_local.hour * 60 + idx_local.minute
    mask = np.zeros(len(df), dtype=bool)

    for s, e in windows:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        smin = sh * 60 + sm
        emin = eh * 60 + em
        mask |= (minutes >= smin) & (minutes <= emin)

    return df.loc[mask], feed.base_dir


def to_ohlcv_arrays(df):
    ts = df.index.view("int64")
    return OHLCVArrays(
        ts=ts,
        o=df["open"].to_numpy(),
        h=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        c=df["close"].to_numpy(),
        v=df["volume"].to_numpy(),
    )


def compute_forward_returns(close: pd.Series, horizon: int):
    return close.shift(-horizon) / close - 1.0


# ============================================================
# OPTIMIZER API
# ============================================================

def preload_data():
    df, _ = load_index_sessions(
        "NDXm",
        [2018, 2019, 2020, 2021, 2022],
        (("08:55", "10:00"), ("14:55", "16:00")),
        "Europe/Madrid",
    )
    return df, None


def run_with_params(df, context, params):
    strat = StrategyBarridaApertura(
        wick_factor=params.get("wick_factor", 1.5),
        atr_percentile=params.get("atr_percentile", 0.4),
        volume_percentile=params.get("volume_percentile", 0.6),
    )

    arrays = to_ohlcv_arrays(df)
    signals = strat.generate_signals(arrays).signals
    sig_series = pd.Series(signals, index=df.index)

    fwd = compute_forward_returns(df["close"], horizon=15)
    trades = fwd[sig_series == 1].dropna().rename("fwd_return")

    return trades.to_frame()


# ============================================================
# MAIN (for standalone analysis)
# ============================================================

def main():
    cfg = SweepStudyConfig(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        horizon=15,
        timezone="Europe/Madrid",
    )

    years = list(range(cfg.start_year, cfg.end_year + 1))
    windows = (("08:55", "10:00"), ("14:55", "16:00"))

    df, _ = load_index_sessions(cfg.index, years, windows, cfg.timezone)
    arrays = to_ohlcv_arrays(df)

    strat = StrategyBarridaApertura()
    signals = strat.generate_signals(arrays).signals
    sig_series = pd.Series(signals, index=df.index)

    fwd = compute_forward_returns(df["close"], cfg.horizon)
    trades = fwd[sig_series == 1].dropna()

    out = REPORTS_DIR / "research" / "microstructure" / "v1"
    out.mkdir(parents=True, exist_ok=True)
    trades.rename("fwd_return").to_csv(out / "opening_sweeps_v1_trades.csv")

    print("Signals:", (sig_series == 1).sum())
    print("Mean return:", trades.mean())
    print("Winrate:", (trades > 0).mean())


if __name__ == "__main__":
    main()
