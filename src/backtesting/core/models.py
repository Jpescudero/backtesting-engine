from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

import numpy as np

from src.data.feeds import OHLCVArrays


@dataclass
class MarketDataBatch:
    """OHLCV data container used by the new engine pipeline.

    All arrays must share the same length and represent the same timeline.  The
    container enforces monotonic timestamps to keep the broker/strategy
    contracts deterministic.
    """

    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(self.timestamps), len(self.open), len(self.high), len(self.low), len(self.close), len(self.volume)}
        if len(lengths) != 1:
            raise ValueError("Todos los arrays OHLCV deben tener la misma longitud")

        if self.timestamps.ndim != 1:
            raise ValueError("Los timestamps deben ser un array 1D")

        if not np.all(self.timestamps[1:] >= self.timestamps[:-1]):
            raise ValueError("Los timestamps deben estar ordenados de forma creciente")

    @property
    def size(self) -> int:
        return len(self.timestamps)

    def slice_from(self, start: int) -> "MarketDataBatch":
        return MarketDataBatch(
            timestamps=self.timestamps[start:],
            open=self.open[start:],
            high=self.high[start:],
            low=self.low[start:],
            close=self.close[start:],
            volume=self.volume[start:],
        )

    def to_ohlcv_arrays(self) -> OHLCVArrays:
        return OHLCVArrays(ts=self.timestamps, o=self.open, h=self.high, low=self.low, c=self.close, v=self.volume)

    @classmethod
    def from_ohlcv(cls, data: OHLCVArrays) -> "MarketDataBatch":
        return cls(
            timestamps=np.asarray(data.ts),
            open=np.asarray(data.o),
            high=np.asarray(data.h),
            low=np.asarray(data.low),
            close=np.asarray(data.c),
            volume=np.asarray(data.v),
        )


@dataclass
class OrderRequest:
    """Order instruction emitted by a strategy.

    Attributes
    ----------
    symbol: str
        Instrument identifier.
    side: str
        "buy" or "sell".
    quantity: float
        Units requested. Must be positive; the broker handles min/max sizes.
    timestamp: datetime
        When the order was created by the strategy timeline.
    order_type: str
        Only "market" is supported by the default simulated broker; limit/stop
        requests are passed through for future extensions.
    bar_index: int
        Index of the bar used for decision making. Broker latency is applied on
        top of this timestamp when producing the fill.
    price: float | None
        Desired limit/stop price; ignored for market orders.
    """

    symbol: str
    side: str
    quantity: float
    timestamp: datetime
    order_type: str = "market"
    bar_index: int = 0
    price: float | None = None

    def validate(self, max_index: int) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("side debe ser 'buy' o 'sell'")
        if self.quantity <= 0:
            raise ValueError("quantity debe ser positivo")
        if self.bar_index < 0 or self.bar_index >= max_index:
            raise ValueError("bar_index fuera de rango para el dataset")


@dataclass
class OrderFill:
    """Execution result produced by a broker simulation."""

    order: OrderRequest
    filled_quantity: float
    avg_price: float
    status: str
    submitted_at: datetime
    executed_at: datetime

    @property
    def notional(self) -> float:
        return self.filled_quantity * self.avg_price


@dataclass
class LatencyProfile:
    """Latency contract applied by brokers.

    A positive latency means that orders emitted at ``timestamp`` are only
    eligible for execution after ``timestamp + latency``.  The broker is
    responsible for applying the delay while preserving the original ordering
    of requests within the batch.
    """

    latency: timedelta = field(default_factory=lambda: timedelta())

    def apply(self, base_ts: datetime) -> datetime:
        return base_ts + self.latency


@dataclass
class TradeTimeline:
    """Minimal log of the trading timeline used by the engine."""

    fills: list[OrderFill]

    def add_fills(self, batch: Iterable[OrderFill]) -> None:
        self.fills.extend(batch)

    def realized_pnl(self) -> float:
        return sum(fill.notional if fill.order.side == "sell" else -fill.notional for fill in self.fills)
