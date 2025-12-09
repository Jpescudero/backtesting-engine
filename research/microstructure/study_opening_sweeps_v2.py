"""
Improved statistical study for opening sweep patterns — Version 2
Includes microstructural filters, volatility filters and dynamic horizon.
"""

from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# AUTO-SET PROJECT ROOT
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feeds import NPZOHLCVFeed, OHLCVArrays
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


# ----------------------------------------------------------------------
# LOADING & FILTERING
# ----------------------------------------------------------------------

def load_sessions(symbol: str, years: Iterable[int], windows, tz: str):
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
        s_h, s_m = map(int, s.split(":"))
        e_h, e_m = map(int, e.split(":"))
        s_min = s_h * 60 + s_m
        e_min = e_h * 60 + e_m
        mask |= (minutes >= s_min) & (minutes <= e_min)

    return df.loc[mask], feed.base_dir


# ----------------------------------------------------------------------
# MICROSTRUCTURAL FILTERS V2
# ----------------------------------------------------------------------

def detect_sweep_signals(df, cfg: SweepStudyConfigV2):
    """
    Sweep detection with:
    - Wick size filter
    - ATR percentile filter
    - Volume percentile filter
    - Trend confirmation (previous candles directional)
    """
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    v = df.volume.values
    idx = df.index

    # --- ATR -------------------------------------------------------------
    hl = h - l
    hc = np.abs(h - np.roll(c, 1))
    lc = np.abs(l - np.roll(c, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))

    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()

    # Clean NaNs for robust filtering
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    # Compute thresholds *inside* the function
    atr_thr = atr_norm.quantile(cfg.min_atr_percentile)
    vol_thr = np.quantile(v, cfg.require_volume_percentile)

    signals = np.zeros(len(df), dtype=np.int8)

    for i in range(2, len(df)):
        body = abs(c[i] - o[i])
        wick_down = o[i] - l[i] if c[i] >= o[i] else c[i] - l[i]

        # Sweep pattern: strong lower wick
        sweep_down = (wick_down > cfg.min_wick_factor * body)

        # Trend prior: two bearish candles
        trend_down = (c[i-1] < o[i-1]) and (c[i-2] < o[i-2])

        # Final filter: ATR + Volume
        if sweep_down and trend_down:
            if atr_norm.iloc[i] >= atr_thr and v[i] >= vol_thr:
                signals[i] = 1

    return pd.Series(signals, index=idx)



# ----------------------------------------------------------------------
# RETURNS & PLOTS
# ----------------------------------------------------------------------

def compute_dynamic_forward_returns(df, atr_norm):
    # Clean ATR norm for NaN/inf issues
    atr_clean = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    dynamic_horizon = (5 + (atr_clean * 20)).clip(lower=5, upper=30)
    dynamic_horizon = dynamic_horizon.astype(int)
    rets = []
    for i in range(len(df)):
        h = dynamic_horizon[i]
        if i + h < len(df):
            r = df.close.iloc[i + h] / df.close.iloc[i] - 1
            rets.append(r)
        else:
            rets.append(np.nan)
    return pd.Series(rets, index=df.index)


def save_plots_v2(returns, outdir):
    fig, ax = plt.subplots(figsize=(9, 4))
    returns.hist(bins=60, ax=ax)
    ax.set_title("Opening Sweep — Forward Returns (V2)")
    fig.savefig(outdir / "opening_sweeps_v2_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    (1 + returns.fillna(0)).cumprod().plot(ax=ax)
    ax.set_title("Equity Curve (V2, No Costs)")
    fig.savefig(outdir / "opening_sweeps_v2_equity.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    ensure_directories_exist()

    cfg = SweepStudyConfigV2(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        timezone="Europe/Madrid",
    )

    years = list(range(cfg.start_year, cfg.end_year + 1))
    windows = (("08:55", "10:00"), ("14:55", "16:00"))

    df, data_path = load_sessions(cfg.index, years, windows, cfg.timezone)

    # Build ATR for dynamic horizon
    # (reuse logic from detection)
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    hl = h - l
    hc = np.abs(h - np.roll(c, 1))
    lc = np.abs(l - np.roll(c, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = pd.Series(tr).rolling(20).mean().bfill().ffill()
    atr_norm = atr / atr.rolling(1000).mean()

    # Detect improved signals
    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    # Compute dynamic returns
    fwd_returns = compute_dynamic_forward_returns(df, atr_norm)
    trade_returns = fwd_returns[entries]

    # Save outputs
    out = REPORTS_DIR / "research" / "microstructure" / "v2"
    out.mkdir(parents=True, exist_ok=True)

    trade_returns.dropna().to_csv(out / "opening_sweeps_v2_trades.csv")
    save_plots_v2(trade_returns.dropna(), out)

    print("Signals:", entries.sum())
    print("Mean return:", trade_returns.mean())
    print("Win rate:", (trade_returns > 0).mean())

if __name__ == "__main__":
    main()
