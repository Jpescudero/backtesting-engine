"""
V3 — Opening Sweep Study with SL/TP
-----------------------------------
Advanced research version:
- Sweep + absorption + microstructural filters
- Dynamic Stop Loss
- Dynamic Take Profit based on R-multiple
- Trade simulation
- Optimizer–compatible API
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.feeds import NPZOHLCVFeed
from src.config.paths import REPORTS_DIR, ensure_directories_exist


SessionWindow = Tuple[str, str]


# ============================================================
# CONFIG
# ============================================================

@dataclass
class SweepStudyConfigV3:
    index: str
    start_year: int
    end_year: int
    timezone: str = "Europe/Madrid"

    min_wick_factor: float = 1.8
    min_atr_percentile: float = 0.5
    require_volume_percentile: float = 0.6

    sl_buffer_atr: float = 0.3
    sl_buffer_relative: float = 0.1
    tp_multiplier: float = 1.2

    max_horizon: int = 30


# ============================================================
# DATA LOADING
# ============================================================

def load_sessions(symbol: str, years: Iterable[int], windows, tz: str):
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
    mins = idx_local.hour * 60 + idx_local.minute

    mask = np.zeros(len(df), dtype=bool)
    for s, e in windows:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        smin = sh*60 + sm
        emin = eh*60 + em
        mask |= (mins >= smin) & (mins <= emin)

    return df.loc[mask], feed.base_dir


# ============================================================
# SIGNAL DETECTION + SCORE
# ============================================================

def detect_sweep_signals_v3(df, cfg: SweepStudyConfigV3):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    v = df.volume.values
    idx = df.index

    hl = h - l
    hc = np.abs(h - np.roll(c,1))
    lc = np.abs(l - np.roll(c,1))
    tr = np.maximum(hl, np.maximum(hc, lc))

    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    atr_thr = atr_norm.quantile(cfg.min_atr_percentile)
    vol_thr = np.quantile(v, cfg.require_volume_percentile)

    signals = np.zeros(len(df), dtype=np.int8)
    scores = np.zeros(len(df), dtype=float)

    for i in range(2, len(df)):
        body = abs(c[i] - o[i])
        wick_down = (o[i] - l[i]) if c[i] >= o[i] else (c[i] - l[i])
        wick_factor = wick_down / (body + 1e-9)

        if wick_factor < cfg.min_wick_factor:
            continue
        if v[i] < vol_thr:
            continue
        if atr_norm.iloc[i] < atr_thr:
            continue

        if not ((c[i-1] < o[i-1]) and (c[i-2] < o[i-2])):
            continue

        mid = l[i] + 0.5 * wick_down
        absorption = 1.0 if c[i] > mid else 0.0

        signals[i] = 1

        wick_norm = min(wick_factor / 3.0, 1.0)
        score = (
            0.4 * wick_norm +
            0.3 * atr_norm.iloc[i] +
            0.2 * (v[i] / max(v)) +
            0.1 * absorption
        )
        scores[i] = min(score, 1.0)

    return pd.Series(signals, index=idx), pd.Series(scores, index=idx), atr, atr_norm


# ============================================================
# SL / TP CALCULATION
# ============================================================

def compute_sl_tp(entry_price, low_sweep, atr_val, atr_norm, cfg: SweepStudyConfigV3):
    sl = low_sweep - (cfg.sl_buffer_atr * atr_val + cfg.sl_buffer_relative * atr_norm * atr_val)
    dist = entry_price - sl
    tp = entry_price + cfg.tp_multiplier * dist
    return sl, tp


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, i_entry, sl, tp, max_horizon):
    entry_price = df.close.iloc[i_entry]
    for j in range(i_entry+1, min(i_entry+1+max_horizon, len(df))):
        if df.low.iloc[j] <= sl:
            return "SL", sl, df.index[j]
        if df.high.iloc[j] >= tp:
            return "TP", tp, df.index[j]

    final_price = df.close.iloc[min(i_entry+max_horizon, len(df)-1)]
    return "HORIZON", final_price, df.index[min(i_entry+max_horizon, len(df)-1)]


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_directories_exist()

    cfg = SweepStudyConfigV3("NDXm", 2018, 2022)

    years = list(range(cfg.start_year, cfg.end_year+1))
    windows = (("08:55","10:00"),("14:55","16:00"))

    df, _ = load_sessions(cfg.index, years, windows, cfg.timezone)

    signals, scores, atr, atr_norm = detect_sweep_signals_v3(df, cfg)

    trades = []

    for i in range(len(df)):
        if signals.iloc[i] != 1:
            continue

        entry = df.close.iloc[i]
        low_sweep = df.low.iloc[i]

        sl, tp = compute_sl_tp(entry, low_sweep, atr.iloc[i], atr_norm.iloc[i], cfg)
        result, exit_price, exit_time = simulate_trade(df, i, sl, tp, cfg.max_horizon)

        r = (exit_price - entry) / (entry - sl)

        trades.append({
            "timestamp": df.index[i],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "result": result,
            "r_multiple": r,
            "score": scores.iloc[i],
        })

    trades_df = pd.DataFrame(trades)

    out = REPORTS_DIR / "research" / "microstructure" / "v3"
    out.mkdir(parents=True, exist_ok=True)

    trades_df.to_csv(out / "opening_sweeps_v3_trades.csv", index=False)

    eq = (1 + trades_df["r_multiple"].fillna(0)).cumprod()
    eq.plot(figsize=(10,4))
    plt.grid(True)
    plt.savefig(out / "opening_sweeps_v3_equity.png", dpi=150)
    plt.close()


# ============================================================
# OPTIMIZER API
# ============================================================

def preload_data():
    df, _ = load_sessions("NDXm",
                          [2018,2019,2020,2021,2022],
                          (("08:55","10:00"),("14:55","16:00")),
                          "Europe/Madrid")

    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    tr = np.maximum(h-l, np.maximum(abs(h - np.roll(c,1)), abs(l - np.roll(c,1))))
    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()
    atr_norm = atr_norm.replace([np.inf,-np.inf],np.nan).bfill().ffill()

    return df, (atr, atr_norm)


def run_with_params(df, df_ohlcv, params):
    atr, atr_norm = df_ohlcv

    cfg = SweepStudyConfigV3(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        min_wick_factor=params.get("wick_factor", 1.8),
        min_atr_percentile=params.get("atr_percentile", 0.5),
        require_volume_percentile=params.get("volume_percentile", 0.6),
        sl_buffer_atr=params.get("sl_buffer_atr", 0.3),
        sl_buffer_relative=params.get("sl_buffer_relative", 0.1),
        tp_multiplier=params.get("tp_multiplier", 1.2),
    )

    signals, scores, _, _ = detect_sweep_signals_v3(df, cfg)

    trades = []

    for i in range(len(df)):
        if signals.iloc[i] != 1:
            continue

        entry = df.close.iloc[i]
        low_sweep = df.low.iloc[i]

        sl, tp = compute_sl_tp(entry, low_sweep, atr.iloc[i], atr_norm.iloc[i], cfg)
        result, exit_price, exit_time = simulate_trade(df, i, sl, tp, cfg.max_horizon)

        r = (exit_price - entry) / (entry - sl)

        trades.append({"r_multiple": r})

    return pd.DataFrame(trades)


if __name__ == "__main__":
    main()
