"""V1 — Opening Sweep study using forward returns only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from research.microstructure.session_loader import SessionLoadConfig, load_sessions

from src.config.paths import REPORTS_DIR, ensure_directories_exist


@dataclass
class SweepStudyConfigV1:
    """Configuration for the baseline opening sweep study."""

    index: str
    start_year: int
    end_year: int
    timezone: str
    min_wick_factor: float = 1.2
    min_atr_percentile: float = 0.2
    require_volume_percentile: float = 0.4


WINDOWS: tuple[tuple[str, str], ...] = (("08:55", "10:00"), ("14:55", "16:00"))


# ============================================================
# DATA PREPARATION
# ============================================================


def _base_df(cfg: SweepStudyConfigV1) -> pd.DataFrame:
    years = list(range(cfg.start_year, cfg.end_year + 1))
    df, _ = load_sessions(
        SessionLoadConfig(
            symbol=cfg.index,
            years=years,
            windows=WINDOWS,
            timezone=cfg.timezone,
        )
    )
    return df


# ============================================================
# SIGNAL DETECTION
# ============================================================


def detect_sweep_signals(df: pd.DataFrame, cfg: SweepStudyConfigV1) -> pd.Series:
    """Identify opening sweep entries based on wick prominence and filters."""

    open_, high, low, close = (
        df.open.values,
        df.high.values,
        df.low.values,
        df.close.values,
    )
    volume = df.volume.values

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

        signals[i] = 1

    return pd.Series(signals, index=df.index)


# ============================================================
# FORWARD RETURNS
# ============================================================


def compute_forward_returns(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    """Compute fixed-horizon forward returns."""

    forward = np.empty(len(df))
    forward[:] = np.nan

    for i in range(len(df)):
        if i + horizon < len(df):
            forward[i] = df.close.iloc[i + horizon] / df.close.iloc[i] - 1

    return pd.Series(forward, index=df.index)


# ============================================================
# REPORTING
# ============================================================


def _report_folder(name: str) -> Path:
    ensure_directories_exist()
    root = REPORTS_DIR / "research" / "microstructure" / "reports" / name
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = root / timestamp
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


# ============================================================
# MAIN EXECUTION
# ============================================================


def main() -> None:
    cfg = SweepStudyConfigV1("NDXm", 2018, 2022, "Europe/Madrid")
    df = _base_df(cfg)

    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    fwd = compute_forward_returns(df)
    trades = fwd[entries].dropna().rename("fwd_return")

    outdir = _report_folder("v1")
    _save_trades(trades, outdir)
    print(f"Saved V1 trades to {outdir}")


# ============================================================
# OPTIMIZER API
# ============================================================


def preload_data() -> pd.DataFrame:
    cfg = SweepStudyConfigV1("NDXm", 2018, 2022, "Europe/Madrid")
    return _base_df(cfg)


def run_with_params(df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    cfg = SweepStudyConfigV1(
        index="NDXm",
        start_year=2018,
        end_year=2022,
        timezone="Europe/Madrid",
        min_wick_factor=params.get("wick_factor", 1.2),
        min_atr_percentile=params.get("atr_percentile", 0.2),
        require_volume_percentile=params.get("volume_percentile", 0.4),
    )

    signals = detect_sweep_signals(df, cfg)
    entries = signals == 1

    fwd = compute_forward_returns(df)
    trades = fwd[entries].dropna().rename("fwd_return")
    return trades.to_frame()


if __name__ == "__main__":
    main()
