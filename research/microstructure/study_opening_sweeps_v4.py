"""
V4 — Opening Sweep Strategy (Production Version)
------------------------------------------------
This is the consolidated, optimized, and parameter-fixed version
based on optimizer_v3_fast results.

Optimized for:
- speed
- clarity
- integration with a backtesting engine
- stability based on empirical results

Default parameters come from optimizer:
    wick_factor = 1.5
    atr_percentile = 0.5
    volume_percentile = 0.4
    sl_buffer_atr = 0.3
    sl_buffer_relative = 0.1
    tp_multiplier = 1.2
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.data.feeds import NPZOHLCVFeed


# ======================================================================
# CONFIG (defaults fully determined by optimization)
# ======================================================================

@dataclass
class SweepStudyConfigV4:
    index: str = "NDXm"
    timezone: str = "Europe/Madrid"

    # Optimized detection filters
    wick_factor: float = 1.5
    atr_percentile: float = 0.5
    volume_percentile: float = 0.4

    # Optimized SL/TP parameters
    sl_buffer_atr: float = 0.3
    sl_buffer_relative: float = 0.1
    tp_multiplier: float = 1.2

    # Execution
    max_horizon: int = 30


# ======================================================================
# DATA LOADING
# ======================================================================

def load_sessions(symbol, years, windows, tz):
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_years(list(years))

    idx = pd.to_datetime(data.ts, unit="ns", utc=True)

    df = pd.DataFrame({
        "open": data.o,
        "high": data.h,
        "low": data.low,
        "close": data.c,
        "volume": data.v,
    }, index=idx).sort_index()

    idx_local = df.index.tz_convert(tz)
    minutes = idx_local.hour*60 + idx_local.minute

    mask = np.zeros(len(df), dtype=bool)
    for s,e in windows:
        sh,sm = map(int, s.split(":"))
        eh,em = map(int, e.split(":"))
        smin = sh*60 + sm
        emin = eh*60 + em
        mask |= (minutes >= smin) & (minutes <= emin)

    return df.loc[mask]


# ======================================================================
# PRECOMPUTATIONS
# ======================================================================

def precalc_atr(df):
    c = df.close.values
    h = df.high.values
    l = df.low.values

    tr = np.maximum(h-l, np.maximum(abs(h-np.roll(c,1)), abs(l-np.roll(c,1))))
    atr = pd.Series(tr).rolling(20).mean().bfill().values
    atr_norm = atr / pd.Series(atr).rolling(1000).mean().bfill().values
    atr_norm = np.nan_to_num(atr_norm)
    return atr, atr_norm


def precalc_signals(df, cfg):
    o = df.open.values
    h = df.high.values
    l = df.low.values
    c = df.close.values
    v = df.volume.values

    atr, atr_norm = precalc_atr(df)

    atr_thr = np.quantile(atr_norm, cfg.atr_percentile)
    vol_thr = np.quantile(v, cfg.volume_percentile)

    N = len(df)
    signals = np.zeros(N, dtype=np.int8)

    for i in range(2, N):
        body = abs(c[i] - o[i])
        wick = (o[i] - l[i]) if c[i] >= o[i] else (c[i] - l[i])
        wick_factor = wick / (body + 1e-9)

        if wick_factor < cfg.wick_factor:
            continue
        if v[i] < vol_thr:
            continue
        if atr_norm[i] < atr_thr:
            continue

        if not (c[i-1] < o[i-1] and c[i-2] < o[i-2]):
            continue

        signals[i] = 1

    return signals, atr, atr_norm


# ======================================================================
# TRADE SIMULATION
# ======================================================================

def simulate_trades(df, signals, atr, atr_norm, cfg):
    c = df.close.values
    l = df.low.values
    h = df.high.values

    trades = []

    for i in range(len(df)):
        if signals[i] != 1:
            continue

        entry = c[i]
        low_sweep = l[i]

        sl = low_sweep - (cfg.sl_buffer_atr*atr[i] + cfg.sl_buffer_relative*atr_norm[i]*atr[i])
        dist = entry - sl
        tp = entry + cfg.tp_multiplier * dist

        # simulate horizon
        for j in range(i+1, min(i+1+cfg.max_horizon, len(df))):
            if l[j] <= sl:
                exit_price = sl
                result = "SL"
                break
            if h[j] >= tp:
                exit_price = tp
                result = "TP"
                break
        else:
            # horizon exit
            exit_price = c[min(i+cfg.max_horizon, len(df)-1)]
            result = "HORIZON"

        r_mult = (exit_price - entry) / (entry - sl)

        trades.append({
            "entry_idx": i,
            "exit_idx": j if result != "HORIZON" else min(i+cfg.max_horizon, len(df)-1),
            "entry_price": entry,
            "exit_price": exit_price,
            "sl": sl,
            "tp": tp,
            "result": result,
            "r_multiple": r_mult,
        })

    return pd.DataFrame(trades)


# ======================================================================
# MAIN API
# ======================================================================

def run_v4():
    cfg = SweepStudyConfigV4()

    years = [2018,2019,2020,2021,2022]
    windows = (("08:55","10:00"), ("14:55","16:00"))
    df = load_sessions(cfg.index, years, windows, cfg.timezone)

    signals, atr, atr_norm = precalc_signals(df, cfg)
    trades = simulate_trades(df, signals, atr, atr_norm, cfg)

    return trades


if __name__ == "__main__":
    trades = run_v4()
    print(trades.head())
    print("Total trades:", len(trades))
