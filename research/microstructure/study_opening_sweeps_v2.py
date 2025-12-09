"""
V2 — Microstructural Opening Sweep Study
----------------------------------------
Adds wick filter, ATR filter, volume filter.
Still uses forward returns. Optimizer-compatible.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from src.data.feeds import NPZOHLCVFeed
from src.config.paths import REPORTS_DIR, ensure_directories_exist


SessionWindow = Tuple[str, str]


@dataclass
class SweepStudyConfigV2:
    index: str
    start_year: int
    end_year: int
    timezone: str
    min_wick_factor: float = 1.5
    min_atr_percentile: float = 0.4
    require_volume_percentile: float = 0.5


# ============================================================
# DATA LOADING
# ============================================================

def load_sessions(symbol: str, years: Iterable[int], windows, tz: str):
    feed = NPZOHLCVFeed(symbol=symbol, timeframe="1m")
    data = feed.load_years(list(years))

    idx = pd.to_datetime(data.ts, unit="ns", utc=True)
    df = pd.DataFrame(
        {
            "open": data.o,
            "high": data.h,
            "low": data.low,
            "close": data.c,
            "volume": data.v,
        },
        index=idx,
    ).sort_index()

    idx_local = df.index.tz_convert(tz)
    minutes = idx_local.hour * 60 + idx_local.minute

    mask = np.zeros(len(df), dtype=bool)
    for s, e in windows:
        sh, sm = map(int, s.split(":"))
        eh, em = map(int, e.split(":"))
        smin = sh*60 + sm
        emin = eh*60 + em
        mask |= (minutes >= smin) & (minutes <= emin)

    return df.loc[mask], feed.base_dir


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_sweep_signals(df, cfg: SweepStudyConfigV2):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    v = df.volume.values
    idx = df.index

    hl = h - l
    hc = np.abs(h - np.roll(c,1))
    lc = np.abs(l - np.roll(c,1))
    tr = np.maximum(hl, np.maximum(hc, lc))

    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()
    atr_norm = atr_norm.replace([np.inf,-np.inf], np.nan).bfill().ffill()

    atr_thr = atr_norm.quantile(cfg.min_atr_percentile)
    vol_thr = np.quantile(v, cfg.require_volume_percentile)

    sig = np.zeros(len(df), dtype=np.int8)

    for i in range(2, len(df)):
        body = abs(c[i] - o[i])
        wick_down = o[i]-l[i] if c[i]>=o[i] else c[i]-l[i]

        if wick_down <= cfg.min_wick_factor * body:
            continue
        if v[i] < vol_thr:
            continue
        if atr_norm.iloc[i] < atr_thr:
            continue
        if not ((c[i-1]<o[i-1]) and (c[i-2]<o[i-2])):
            continue

        sig[i] = 1

    return pd.Series(sig, index=idx)


# ============================================================
# FORWARD RETURNS (DYNAMIC)
# ============================================================

def compute_dynamic_forward_returns(df, atr_norm):
    atr_clean = atr_norm.replace([np.inf,-np.inf], np.nan).bfill().ffill()
    horizon = (5 + (atr_clean * 20)).clip(5, 30).astype(int)

    fwd = []
    for i in range(len(df)):
        h = horizon[i]
        if i+h < len(df):
            fwd.append(df.close.iloc[i+h] / df.close.iloc[i] - 1)
        else:
            fwd.append(np.nan)

    return pd.Series(fwd, index=df.index)


# ============================================================
# OPTIMIZER API
# ============================================================

def preload_data():
    df, _ = load_sessions(
        "NDXm",
        [2018,2019,2020,2021,2022],
        (("08:55","10:00"),("14:55","16:00")),
        "Europe/Madrid",
    )
    return df, None


def run_with_params(df, context, params):
    cfg = SweepStudyConfigV2(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        timezone="Europe/Madrid",
        min_wick_factor=params.get("wick_factor",1.5),
        min_atr_percentile=params.get("atr_percentile",0.4),
        require_volume_percentile=params.get("volume_percentile",0.6),
    )

    signals = detect_sweep_signals(df, cfg)
    entries = signals==1

    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    tr = np.maximum(h-l, np.maximum(abs(h-np.roll(c,1)), abs(l-np.roll(c,1))))
    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()

    fwd = compute_dynamic_forward_returns(df, atr_norm)
    trades = fwd[entries].dropna().rename("fwd_return")
    return trades.to_frame()


# ============================================================
# MAIN (standalone)
# ============================================================

def main():
    cfg = SweepStudyConfigV2("NDXm", 2018, 2022, "Europe/Madrid")

    years = list(range(cfg.start_year, cfg.end_year+1))
    windows = (("08:55","10:00"),("14:55","16:00"))

    df, _ = load_sessions(cfg.index, years, windows, cfg.timezone)

    o,h,l,c = df.open.values, df.high.values, df.low.values, df.close.values
    tr = np.maximum(h-l, np.maximum(abs(h-np.roll(c,1)), abs(l-np.roll(c,1))))
    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()

    signals = detect_sweep_signals(df, cfg)
    entries = (signals==1)

    fwd = compute_dynamic_forward_returns(df, atr_norm)
    trades = fwd[entries].dropna()

    out = REPORTS_DIR / "research" / "microstructure" / "v2"
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "opening_sweeps_v2_trades.csv")


if __name__ == "__main__":
    main()
