"""Statistical study of opening sweep patterns for NDXm."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.data.feeds import OHLCVArrays

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("study_opening_sweeps_inputs.txt")


def _ensure_project_root_on_path() -> None:
    """Add repository root to ``sys.path`` when executed directly."""

    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


_ensure_project_root_on_path()

SessionWindow = Tuple[str, str]


@dataclass
class SweepStudyConfig:
    """Configuration values required to run the sweep study."""

    index: str
    start_year: int
    end_year: int
    horizon: int
    timezone: str


def load_config(file_path: Path = CONFIG_PATH) -> SweepStudyConfig:
    """Read configuration values from a local text file.

    The file must use ``key=value`` pairs, and lines starting with ``#``
    are treated as comments. ``start_year`` and ``end_year`` are required;
    ``horizon`` and ``timezone`` are optional. ``index`` defaults to
    ``NDXm`` if unspecified.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found at {file_path}. Please create it locally."
        )

    values = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(
                "Invalid line in configuration file. Use 'key=value' pairs for inputs."
            )
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()

    try:
        start_year = int(values["start_year"])
        end_year = int(values["end_year"])
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError("The configuration file must define 'start_year' and 'end_year'.") from exc

    if start_year > end_year:
        raise ValueError("'start_year' cannot be greater than 'end_year'.")

    horizon = int(values.get("horizon", "15"))
    timezone = values.get("timezone", "Europe/Madrid")

    return SweepStudyConfig(
        index=values.get("index", "NDXm"),
        start_year=start_year,
        end_year=end_year,
        horizon=horizon,
        timezone=timezone,
    )


def load_index_sessions(
    symbol: str, years: Iterable[int], windows: Tuple[SessionWindow, ...], tz: str
) -> Tuple[pd.DataFrame, Path]:
    """Load 1m bars and keep only the desired intraday windows.

    Returns the filtered dataframe along with the local NPZ directory used,
    so downstream outputs can clearly document the data origin.
    """

    from src.data.feeds import NPZOHLCVFeed

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

    for start_str, end_str in windows:
        start_h, start_m = start_str.split(":")
        end_h, end_m = end_str.split(":")
        start_min = int(start_h) * 60 + int(start_m)
        end_min = int(end_h) * 60 + int(end_m)
        mask |= (minutes >= start_min) & (minutes <= end_min)

    return df.loc[mask], feed.base_dir


def to_ohlcv_arrays(df: pd.DataFrame) -> "OHLCVArrays":
    from src.data.feeds import OHLCVArrays

    ts = df.index.view("int64")
    return OHLCVArrays(
        ts=ts,
        o=df["open"].to_numpy(),
        h=df["high"].to_numpy(),
        low=df["low"].to_numpy(),
        c=df["close"].to_numpy(),
        v=df["volume"].to_numpy(),
    )


def compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward returns for a given holding horizon."""

    return close.shift(-horizon) / close - 1.0


def save_plots(returns: pd.Series, output_dir: Path) -> None:
    """Persist histogram and equity curve plots for the trade returns."""

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    returns.hist(bins=50, ax=ax)
    ax.set_title("Opening sweep forward returns")
    ax.set_xlabel("Return after signal")
    ax.set_ylabel("Frequency")
    hist_path = output_dir / "opening_sweeps_return_hist.png"
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    equity = (1 + returns).cumprod()
    equity.plot(ax=ax)
    ax.set_title("Cumulative gross equity (no costs)")
    ax.set_ylabel("Equity multiple")
    ax.grid(True)
    eq_path = output_dir / "opening_sweeps_equity.png"
    fig.tight_layout()
    fig.savefig(eq_path, dpi=150)
    plt.close(fig)

    print(f"Saved histogram to {hist_path}")
    print(f"Saved equity curve to {eq_path}")


def main() -> None:
    from src.config.paths import REPORTS_DIR, ensure_directories_exist
    from src.strategies.barrida_apertura import StrategyBarridaApertura

    config = load_config()

    ensure_directories_exist()

    years = list(range(config.start_year, config.end_year + 1))
    session_windows: Tuple[SessionWindow, ...] = (("08:55", "10:00"), ("14:55", "16:00"))

    df, data_path = load_index_sessions(config.index, years, session_windows, tz=config.timezone)
    if df.empty:
        raise ValueError("No data found for the requested years/windows")

    strategy = StrategyBarridaApertura()
    data_arrays = to_ohlcv_arrays(df)
    signals = strategy.generate_signals(data_arrays).signals

    signal_series = pd.Series(signals, index=df.index, dtype=np.int8)
    fwd_returns = compute_forward_returns(df["close"], horizon=config.horizon)

    entries = signal_series == 1
    trade_returns = fwd_returns.loc[entries].dropna()

    params = pd.DataFrame(
        {
            "parameter": [
                "index",
                "start_year",
                "end_year",
                "horizon",
                "timezone",
                "session_windows",
                "data_path",
                "timeframe",
            ],
            "value": [
                config.index,
                config.start_year,
                config.end_year,
                config.horizon,
                config.timezone,
                ";".join([f"{start}-{end}" for start, end in session_windows]),
                str(data_path),
                "1m",
            ],
        }
    )

    metrics = pd.DataFrame(
        {
            "metric": [
                "signals_total",
                "signals_per_day",
                "win_rate",
                "mean_return",
                "median_return",
                "avg_positive_return",
                "avg_negative_return",
            ],
            "value": [
                entries.sum(),
                entries.sum() / df.index.normalize().nunique(),
                (trade_returns > 0).mean() if not trade_returns.empty else 0.0,
                trade_returns.mean() if not trade_returns.empty else 0.0,
                trade_returns.median() if not trade_returns.empty else 0.0,
                trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0.0,
                trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0.0,
            ],
        }
    )

    output_dir = REPORTS_DIR / "research" / "microstructure"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "opening_sweeps_summary.csv"
    trades_path = output_dir / "opening_sweeps_trades.csv"

    params.to_csv(output_dir / "opening_sweeps_parameters.csv", index=False)
    metrics.to_csv(summary_path, index=False)
    trade_returns.rename("fwd_return").to_frame().to_csv(trades_path)

    save_plots(trade_returns, output_dir)

    print(params)
    print(metrics)
    print(f"Saved parameters to {output_dir / 'opening_sweeps_parameters.csv'}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved trade returns to {trades_path}")


if __name__ == "__main__":
    main()
