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

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from research.microstructure.session_loader import (  # noqa: E402
    SessionLoadConfig,
    load_sessions,
)

from src.config.paths import REPORTS_DIR, ensure_directories_exist  # noqa: E402

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


def _load_df(cfg: SweepStudyConfigV3) -> pd.DataFrame:
    df, _ = load_sessions(
        SessionLoadConfig(
            symbol=cfg.index,
            years=range(cfg.start_year, cfg.end_year + 1),
            windows=(("08:55", "10:00"), ("14:55", "16:00")),
            timezone=cfg.timezone,
        )
    )
    return df


# ============================================================
# SIGNAL DETECTION + SCORE
# ============================================================


def detect_sweep_signals_v3(
    df: pd.DataFrame, cfg: SweepStudyConfigV3
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    open_, high, low, close = (
        df.open.values,
        df.high.values,
        df.low.values,
        df.close.values,
    )
    volume = df.volume.values
    idx = df.index

    hl = high - low
    hc = np.abs(high - np.roll(close, 1))
    lc = np.abs(low - np.roll(close, 1))
    tr = np.maximum(hl, np.maximum(hc, lc))

    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    atr_thr = atr_norm.quantile(cfg.min_atr_percentile)
    vol_thr = np.quantile(volume, cfg.require_volume_percentile)

    signals = np.zeros(len(df), dtype=np.int8)
    scores = np.zeros(len(df), dtype=float)

    for i in range(2, len(df)):
        body = abs(close[i] - open_[i])
        wick_down = (open_[i] - low[i]) if close[i] >= open_[i] else (close[i] - low[i])
        wick_factor = wick_down / (body + 1e-9)

        if wick_factor < cfg.min_wick_factor:
            continue
        if volume[i] < vol_thr:
            continue
        if atr_norm.iloc[i] < atr_thr:
            continue

        if not ((close[i - 1] < open_[i - 1]) and (close[i - 2] < open_[i - 2])):
            continue

        mid = low[i] + 0.5 * wick_down
        absorption = 1.0 if close[i] > mid else 0.0

        signals[i] = 1

        wick_norm = min(wick_factor / 3.0, 1.0)
        score = (
            0.4 * wick_norm
            + 0.3 * atr_norm.iloc[i]
            + 0.2 * (volume[i] / max(volume))
            + 0.1 * absorption
        )
        scores[i] = min(score, 1.0)

    return pd.Series(signals, index=idx), pd.Series(scores, index=idx), atr, atr_norm


# ============================================================
# SL / TP CALCULATION
# ============================================================


def compute_sl_tp(
    entry_price: float,
    low_sweep: float,
    atr_val: float,
    atr_norm: float,
    cfg: SweepStudyConfigV3,
) -> tuple[float, float]:
    sl = low_sweep - (cfg.sl_buffer_atr * atr_val + cfg.sl_buffer_relative * atr_norm * atr_val)
    dist = entry_price - sl
    tp = entry_price + cfg.tp_multiplier * dist
    return sl, tp


# ============================================================
# TRADE SIMULATION
# ============================================================


def simulate_trade(
    df: pd.DataFrame, i_entry: int, sl: float, tp: float, max_horizon: int
) -> tuple[str, float, pd.Timestamp]:
    for j in range(i_entry + 1, min(i_entry + 1 + max_horizon, len(df))):
        if df.low.iloc[j] <= sl:
            return "SL", sl, df.index[j]
        if df.high.iloc[j] >= tp:
            return "TP", tp, df.index[j]

    final_price = df.close.iloc[min(i_entry + max_horizon, len(df) - 1)]
    return "HORIZON", final_price, df.index[min(i_entry + max_horizon, len(df) - 1)]


# ============================================================
# MAIN
# ============================================================


def _report_folder() -> Path:
    ensure_directories_exist()
    root = REPORTS_DIR / "research" / "microstructure" / "reports" / "v3"
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_outputs(trades_df: pd.DataFrame, out: Path) -> None:
    trades_df.to_csv(out / "trades.csv", index=False)

    equity = (1 + trades_df["r_multiple"].fillna(0)).cumprod()
    equity.plot(figsize=(10, 4))
    plt.grid(True)
    plt.savefig(out / "equity_curve.png", dpi=150)
    plt.close()

    summary = pd.DataFrame(
        {
            "count": [len(trades_df)],
            "mean": [trades_df["r_multiple"].mean()],
            "std": [trades_df["r_multiple"].std()],
            "winrate": [(trades_df["r_multiple"] > 0).mean()],
        }
    )
    summary.to_csv(out / "summary.csv", index=False)


def main() -> None:
    cfg = SweepStudyConfigV3("NDXm", 2018, 2022)

    df = _load_df(cfg)
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

        trades.append(
            {
                "timestamp": df.index[i],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "exit_time": exit_time,
                "result": result,
                "r_multiple": r,
                "score": scores.iloc[i],
            }
        )

    trades_df = pd.DataFrame(trades)

    out = _report_folder()
    _save_outputs(trades_df, out)
    print(f"Saved V3 trades to {out}")


# ============================================================
# OPTIMIZER API
# ============================================================


def preload_data() -> tuple[pd.DataFrame, tuple[pd.Series, pd.Series]]:
    cfg = SweepStudyConfigV3("NDXm", 2018, 2022)
    df = _load_df(cfg)

    high = df.high.values
    low = df.low.values
    close = df.close.values

    tr = np.maximum(
        high - low,
        np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))),
    )
    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()
    atr_norm = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    return df, (atr, atr_norm)


def run_with_params(
    df: pd.DataFrame, df_ohlcv: tuple[pd.Series, pd.Series], params: dict[str, float]
) -> pd.DataFrame:
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
