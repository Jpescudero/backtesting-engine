from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.backtesting.core.models import MarketDataBatch
from src.backtesting.data.interfaces import DataLoader
from src.data.feeds import NPZOHLCVFeed


class NPZDataLoader(DataLoader):
    """Ingestor basado en ficheros NPZ de OHLCV.

    El loader aplica una normalización mínima (orden temporal y conversión a
    arrays contiguos) y documenta que los timestamps están expresados en
    nanosegundos desde época Unix.
    """

    def __init__(self, symbol: str, timeframe: str = "1m", base_dir: Path | None = None) -> None:
        self.feed = NPZOHLCVFeed(symbol=symbol, timeframe=timeframe, base_dir=base_dir)
        self._timeframe_ns = self._parse_timeframe_to_ns(timeframe)

    def load(self) -> MarketDataBatch:
        ohlcv = self.feed.load_all()
        timestamps = np.asarray(ohlcv.ts)
        open_ = np.asarray(ohlcv.o)
        high = np.asarray(ohlcv.h)
        low = np.asarray(ohlcv.low)
        close = np.asarray(ohlcv.c)
        volume = np.asarray(ohlcv.v)

        self._ensure_no_nans(
            {
                "timestamps": timestamps,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

        if np.any(np.diff(timestamps) < 0):
            raise ValueError("Los timestamps del NPZ no están ordenados; orden temporal corrupto")

        sort_idx = np.argsort(timestamps)
        sorted_ts = timestamps[sort_idx]
        self._ensure_no_critical_gaps(sorted_ts)

        normalized = MarketDataBatch(
            timestamps=sorted_ts,
            open=open_[sort_idx],
            high=high[sort_idx],
            low=low[sort_idx],
            close=close[sort_idx],
            volume=volume[sort_idx],
        )
        return normalized

    @staticmethod
    def _ensure_no_nans(arrays: dict[str, np.ndarray]) -> None:
        columns_with_nan: list[str] = []
        for name, arr in arrays.items():
            if np.isnan(arr).any():
                columns_with_nan.append(name)

        if columns_with_nan:
            joined = ", ".join(columns_with_nan)
            raise ValueError(f"Datos con valores NaN detectados en: {joined}")

    def _ensure_no_critical_gaps(self, timestamps: Iterable[int | float | np.ndarray]) -> None:
        if self._timeframe_ns is None:
            return

        ts_array = np.asarray(timestamps, dtype=np.int64)
        if ts_array.size < 2:
            return

        diffs = np.diff(ts_array)
        critical_gap = self._timeframe_ns * 5
        mask = diffs > critical_gap
        if np.any(mask):
            idx = int(np.argmax(mask))
            gap_seconds = diffs[idx] / 1e9
            raise ValueError(
                f"Gap crítico detectado entre posiciones {idx} y {idx + 1}: {gap_seconds:.2f} segundos"
            )

    @staticmethod
    def _parse_timeframe_to_ns(timeframe: str) -> int | None:
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        try:
            value = int(timeframe[:-1])
            unit = timeframe[-1]
        except (ValueError, IndexError):
            return None

        if unit not in units:
            return None
        return value * units[unit] * 1_000_000_000
