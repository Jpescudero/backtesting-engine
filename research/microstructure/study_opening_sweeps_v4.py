"""
V4 — Opening Sweep Strategy (Production Version with Visual Outputs)
--------------------------------------------------------------------
This version adds:
- Optimized signals
- Optimized SL/TP (based on optimizer results)
- Trade simulation
- Visual outputs:
    * Equity Curve
    * Histogram of R-multiple
    * Summary .txt
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from src.data.feeds import NPZOHLCVFeed
from src.config.paths import REPORTS_DIR, ensure_directories_exist


# ======================================================================
# CONFIG (optimizer defaults)
# ======================================================================

@dataclass
class SweepStudyConfigV4:
    index: str = "NDXm"
    timezone: str = "Europe/Madrid"

    wick_factor: float = 1.5
    atr_percentile: float = 0.5
    volume_percentile: float = 0.4

    sl_buffer_atr: float = 0.3
    sl_buffer_relative: float = 0.1
    tp_multiplier: float = 1.2

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
# PRECOMPUTATION
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

        # SL / TP
        sl = low_sweep - (cfg.sl_buffer_atr*atr[i] + cfg.sl_buffer_relative*atr_norm[i]*atr[i])
        dist = entry - sl
        tp = entry + cfg.tp_multiplier * dist

        # simulate
        exit_price = None
        result = "HORIZON"
        exit_idx = min(i+cfg.max_horizon, len(df)-1)

        for j in range(i+1, exit_idx+1):
            if l[j] <= sl:
                exit_price = sl
                result = "SL"
                exit_idx = j
                break
            if h[j] >= tp:
                exit_price = tp
                result = "TP"
                exit_idx = j
                break

        if exit_price is None:
            exit_price = c[exit_idx]

        r_metric = (exit_price - entry) / (entry - sl)

        trades.append({
            "i": i,
            "exit_i": exit_idx,
            "entry": entry,
            "exit": exit_price,
            "sl": sl,
            "tp": tp,
            "result": result,
            "r_multiple": r_metric
        })

    return pd.DataFrame(trades)


# ======================================================================
# VISUAL OUTPUTS
# ======================================================================

def generate_visuals(trades, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    eq = (1 + trades["r_multiple"]).cumprod()
    plt.figure(figsize=(10,4))
    plt.plot(eq)
    plt.title("Equity Curve — V4")
    plt.grid(True)
    plt.savefig(outdir / "equity_v4.png", dpi=140)
    plt.close()

    # Histogram
    plt.figure(figsize=(6,4))
    plt.hist(trades["r_multiple"], bins=40, alpha=0.7)
    plt.title("Histogram of R-multiple")
    plt.grid(True)
    plt.savefig(outdir / "histogram_v4.png", dpi=140)
    plt.close()

    # Summary
    sharpe = trades["r_multiple"].mean() / (trades["r_multiple"].std() + 1e-9)
    winrate = (trades["r_multiple"] > 0).mean()

    summary = outdir / "summary_v4.txt"
    with open(summary, "w") as f:
        f.write("V4 Strategy Summary\n")
        f.write("====================\n\n")
        f.write(f"Total trades: {len(trades)}\n")
        f.write(f"Mean R: {trades['r_multiple'].mean():.6f}\n")
        f.write(f"Median R: {trades['r_multiple'].median():.6f}\n")
        f.write(f"Sharpe: {sharpe:.6f}\n")
        f.write(f"Winrate: {winrate:.4f}\n")

    print(f"Visuals saved to: {outdir}")


# ======================================================================
# ENTRY POINT
# ======================================================================

def run_v4():
    cfg = SweepStudyConfigV4()

    years = [2018,2019,2020,2021,2022]
    windows = (("08:55","10:00"),("14:55","16:00"))
    df = load_sessions(cfg.index, years, windows, cfg.timezone)

    signals, atr, atr_norm = precalc_signals(df, cfg)
    trades = simulate_trades(df, signals, atr, atr_norm, cfg)

    outdir = REPORTS_DIR / "research" / "microstructure" / "v4"
    outdir.mkdir(parents=True, exist_ok=True)

    trades.to_csv(outdir / "trades_v4.csv", index=False)
    generate_visuals(trades, outdir)

    return trades


if __name__ == "__main__":
    trades = run_v4()
    print("Trades generated:", len(trades))
