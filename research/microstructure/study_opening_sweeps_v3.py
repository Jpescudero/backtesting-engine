"""
V3 — Optimized Opening Sweep Study with SL/TP (FAST VERSION)
------------------------------------------------------------
Key improvements:

- Precomputes ATR, ATR_norm once
- Precomputes sweep signals once
- Precomputes scores once
- Trade simulation isolated, fast, lightweight
- Compatible with optimizer
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from src.data.feeds import NPZOHLCVFeed
from src.config.paths import REPORTS_DIR, ensure_directories_exist


# ======================================================================
# CONFIG
# ======================================================================

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

    idx_loc = df.index.tz_convert(tz)
    minutes = idx_loc.hour * 60 + idx_loc.minute
    mask = np.zeros(len(df), dtype=bool)

    for s,e in windows:
        sh,sm = map(int, s.split(":"))
        eh,em = map(int, e.split(":"))
        smin = sh*60 + sm
        emin = eh*60 + em
        mask |= (minutes >= smin) & (minutes <= emin)

    return df.loc[mask]


# ======================================================================
# PRECOMPUTATION (FAST)
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


def precalc_sweep_signals(df, cfg):
    o = df.open.values
    h = df.high.values
    l = df.low.values
    c = df.close.values
    v = df.volume.values

    atr, atr_norm = precalc_atr(df)

    atr_thr = np.quantile(atr_norm, cfg.min_atr_percentile)
    vol_thr = np.quantile(v, cfg.require_volume_percentile)

    N = len(df)
    signals = np.zeros(N, dtype=np.int8)
    scores  = np.zeros(N, dtype=float)

    maxv = v.max()

    for i in range(2, N):
        body = abs(c[i]-o[i])
        wick = (o[i]-l[i]) if c[i] >= o[i] else (c[i]-l[i])
        wick_factor = wick / (body + 1e-9)

        if wick_factor < cfg.min_wick_factor:
            continue
        if v[i] < vol_thr:
            continue
        if atr_norm[i] < atr_thr:
            continue
        if not (c[i-1]<o[i-1] and c[i-2]<o[i-2]):
            continue

        mid = l[i] + 0.5*wick
        absorption = 1.0 if c[i] > mid else 0.0

        signals[i] = 1

        wick_norm = min(wick_factor/3.0, 1.0)
        scores[i] = min(
            0.4*wick_norm + 0.3*atr_norm[i] + 0.2*(v[i]/maxv) + 0.1*absorption,
            1.0
        )

    return signals, scores, atr, atr_norm


# ======================================================================
# TRADE SIMULATION (FAST)
# ======================================================================

def simulate_trade_fast(df, idx_entry, sl, tp, max_horizon):
    c = df.close.values
    lo = df.low.values
    hi = df.high.values

    entry = c[idx_entry]

    for j in range(idx_entry+1, min(idx_entry+1+max_horizon, len(df))):
        if lo[j] <= sl:
            return "SL", sl
        if hi[j] >= tp:
            return "TP", tp

    final_price = c[min(idx_entry+max_horizon, len(df)-1)]
    return "HORIZON", final_price


# ======================================================================
# OPTIMIZER API
# ======================================================================

def preload_data():
    years = [2018,2019,2020,2021,2022]
    windows = (("08:55","10:00"),("14:55","16:00"))
    df = load_sessions("NDXm", years, windows, "Europe/Madrid")

    cfg_default = SweepStudyConfigV3("NDXm", 2018, 2022)
    signals, scores, atr, atr_norm = precalc_sweep_signals(df, cfg_default)

    context = {
        "signals": signals,
        "scores": scores,
        "atr": atr,
        "atr_norm": atr_norm,
    }

    return df, context


def run_with_params(df, context, params):
    sig = context["signals"]
    scores = context["scores"]
    atr = context["atr"]
    atr_norm = context["atr_norm"]

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

    dfc = df.close.values
    dfl = df.low.values

    trades = []

    for i in range(len(df)):
        if sig[i] != 1:
            continue

        entry = dfc[i]
        low_sweep = dfl[i]

        sl = low_sweep - (cfg.sl_buffer_atr*atr[i] + cfg.sl_buffer_relative*atr_norm[i]*atr[i])
        dist = entry - sl
        tp = entry + cfg.tp_multiplier * dist

        result, exit_price = simulate_trade_fast(df, i, sl, tp, cfg.max_horizon)
        r = (exit_price - entry) / (entry - sl)

        trades.append({"r_multiple": r})

    return pd.DataFrame(trades)


# ======================================================================
# MAIN (optional for standalone use)
# ======================================================================

def main():
    ensure_directories_exist()
    df, ctx = preload_data()

    cfg = SweepStudyConfigV3("NDXm", 2018, 2022)
    trades = run_with_params(df, ctx, {})

    out = REPORTS_DIR / "research" / "microstructure" / "v3_fast"
    out.mkdir(parents=True, exist_ok=True)

    trades.to_csv(out / "opening_sweeps_v3_fast_trades.csv", index=False)
    print("Done! Trades:", len(trades))


if __name__ == "__main__":
    main()
