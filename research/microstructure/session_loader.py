"""Shared helpers for loading intraday session data for opening sweep studies."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.feeds import NPZOHLCVFeed  # noqa: E402

SessionWindow = Tuple[str, str]


@dataclass
class SessionLoadConfig:
    """Configuration to load sessions for a specific symbol and timezone."""

    symbol: str
    years: Sequence[int]
    windows: Sequence[SessionWindow]
    timezone: str


def load_sessions(config: SessionLoadConfig) -> tuple[pd.DataFrame, str]:
    """Load OHLCV data filtered by intraday windows.

    Args:
        config: SessionLoadConfig with symbol, years, windows, and timezone.

    Returns:
        Tuple with filtered OHLCV dataframe and the NPZ feed base directory.
    """

    feed = NPZOHLCVFeed(symbol=config.symbol, timeframe="1m")
    data = feed.load_years(list(config.years))

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

    idx_local = df.index.tz_convert(config.timezone)
    mins = idx_local.hour * 60 + idx_local.minute

    mask = np.zeros(len(df), dtype=bool)
    for start, end in config.windows:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        start_min = sh * 60 + sm
        end_min = eh * 60 + em
        mask |= (mins >= start_min) & (mins <= end_min)

    return df.loc[mask], feed.base_dir
