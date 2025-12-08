from __future__ import annotations

from pathlib import Path

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

    def load(self) -> MarketDataBatch:
        ohlcv = self.feed.load_all()
        timestamps = np.asarray(ohlcv.ts)
        sort_idx = np.argsort(timestamps)
        normalized = MarketDataBatch(
            timestamps=timestamps[sort_idx],
            open=np.asarray(ohlcv.o)[sort_idx],
            high=np.asarray(ohlcv.h)[sort_idx],
            low=np.asarray(ohlcv.low)[sort_idx],
            close=np.asarray(ohlcv.c)[sort_idx],
            volume=np.asarray(ohlcv.v)[sort_idx],
        )
        return normalized
