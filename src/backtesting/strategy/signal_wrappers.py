from __future__ import annotations

from datetime import datetime
from typing import Callable, Sequence

import numpy as np
from src.backtesting.core.models import MarketDataBatch, OrderRequest
from src.backtesting.strategy.interfaces import Strategy

SignalGenerator = Callable[[MarketDataBatch], np.ndarray]


class SignalStrategyAdapter(Strategy):
    """Adaptador de estrategias basadas en arrays de señales.

    El generador debe devolver un array int8 del mismo tamaño que ``data.close``
    con señales cronológicas. El wrapper convierte las señales en órdenes de
    mercado respetando el timeline original.
    """

    def __init__(self, symbol: str, signal_generator: SignalGenerator, qty: float = 1.0) -> None:
        self.symbol = symbol
        self.signal_generator = signal_generator
        self.qty = qty
        self._signals: np.ndarray | None = None

    def prepare(self, data: MarketDataBatch) -> None:
        signals = self.signal_generator(data)
        if signals.shape[0] != data.size:
            raise ValueError("Las señales deben tener la misma longitud que el dataset")
        self._signals = signals.astype(np.int8)

    def on_bar(self, bar_index: int, data: MarketDataBatch) -> Sequence[OrderRequest]:
        if self._signals is None:
            raise RuntimeError("La estrategia debe llamarse a prepare() antes de iterar")

        signal = int(self._signals[bar_index])
        if signal == 0:
            return []

        side = "buy" if signal > 0 else "sell"
        ts_value = data.timestamps[bar_index]
        ts_dt = datetime.utcfromtimestamp(float(ts_value) / 1e9)
        return [
            OrderRequest(
                symbol=self.symbol,
                side=side,
                quantity=self.qty,
                timestamp=ts_dt,
                bar_index=bar_index,
            )
        ]


def simple_momentum_signal(threshold: float = 0.001) -> SignalGenerator:
    def generator(data: MarketDataBatch) -> np.ndarray:
        returns = np.zeros_like(data.close, dtype=np.float64)
        returns[1:] = (data.close[1:] - data.close[:-1]) / data.close[:-1]
        signals = np.zeros_like(returns, dtype=np.int8)
        signals[returns > threshold] = 1
        signals[returns < -threshold] = -1
        return signals

    return generator
