"""
V2 — Microstructural Opening Sweep Study
----------------------------------------
Adds wick filter, ATR filter, volume filter.
Still uses forward returns. Optimizer-compatible.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from research.microstructure.session_loader import (  # noqa: E402
    SessionLoadConfig,
    load_sessions,
)

from src.config.paths import REPORTS_DIR, ensure_directories_exist  # noqa: E402


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


def _load_df(cfg: SweepStudyConfigV2) -> pd.DataFrame:
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
# SIGNAL DETECTION
# ============================================================


def detect_sweep_signals(df: pd.DataFrame, cfg: SweepStudyConfigV2) -> pd.Series:
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

    sig = np.zeros(len(df), dtype=np.int8)

    for i in range(2, len(df)):
        body = abs(close[i] - open_[i])
        wick_down = (open_[i] - low[i]) if close[i] >= open_[i] else (close[i] - low[i])

        if wick_down <= cfg.min_wick_factor * body:
            continue

        if volume[i] < vol_thr:
            continue
        if atr_norm.iloc[i] < atr_thr:
            continue

        if not ((close[i - 1] < open_[i - 1]) and (close[i - 2] < open_[i - 2])):
            continue

        sig[i] = 1

    return pd.Series(sig, index=idx)


# ============================================================
# FORWARD RETURNS
# ============================================================


def compute_dynamic_forward_returns(df: pd.DataFrame, atr_norm: pd.Series) -> pd.Series:
    atr_clean = atr_norm.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    horizon = (5 + (atr_clean * 20)).clip(5, 30).astype(int)

    fwd = []
    for i in range(len(df)):
        h = horizon[i]
        if i + h < len(df):
            fwd.append(df.close.iloc[i + h] / df.close.iloc[i] - 1)
        else:
            fwd.append(np.nan)
    return pd.Series(fwd, index=df.index)


# ============================================================
# MAIN
# ============================================================


def _report_folder() -> Path:
    ensure_directories_exist()
    root = REPORTS_DIR / "research" / "microstructure" / "reports" / "v2"
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = root / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_trades(trades: pd.Series, outdir: Path) -> None:
    trades.to_csv(outdir / "trades.csv", header=True)
    summary = pd.DataFrame(
        {
            "count": [len(trades)],
            "mean": [trades.mean()],
            "std": [trades.std()],
            "winrate": [(trades > 0).mean()],
        }
    )
    summary.to_csv(outdir / "summary.csv", index=False)


def main() -> None:
    cfg = SweepStudyConfigV2("NDXm", 2018, 2022, "Europe/Madrid")
    df = _load_df(cfg)

    high = df.high.values
    low = df.low.values
    close = df.close.values

    tr = np.maximum(
        high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1)))
    )
    atr = pd.Series(tr).rolling(20).mean().bfill().ffill()
    atr_norm = atr / atr.rolling(1000).mean()

    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    fwd = compute_dynamic_forward_returns(df, atr_norm)
    trades = fwd[entries].dropna()

    out = _report_folder()
    _save_trades(trades, out)
    print(f"Saved V2 trades to {out}")


# ============================================================
# OPTIMIZER API
# ============================================================


def preload_data() -> pd.DataFrame:
    cfg = SweepStudyConfigV2("NDXm", 2018, 2022, "Europe/Madrid")
    return _load_df(cfg)


def run_with_params(df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    cfg = SweepStudyConfigV2(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        timezone="Europe/Madrid",
        min_wick_factor=params.get("wick_factor", 1.5),
        min_atr_percentile=params.get("atr_percentile", 0.4),
        require_volume_percentile=params.get("volume_percentile", 0.6),
    )

    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    high = df.high.values
    low = df.low.values
    close = df.close.values

    tr = np.maximum(
        high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    )
    atr = pd.Series(tr).rolling(20).mean().bfill()
    atr_norm = atr / atr.rolling(1000).mean()

    fwd = compute_dynamic_forward_returns(df, atr_norm)
    trades = fwd[entries].dropna().rename("fwd_return")

    return trades.to_frame()


if __name__ == "__main__":
    main()
