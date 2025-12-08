from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.backtesting.core.models import MarketDataBatch, OrderFill, TradeTimeline
from src.backtesting.data.interfaces import DataLoader
from src.backtesting.execution.interfaces import Broker
from src.backtesting.strategy.interfaces import Strategy


@dataclass
class BacktestEngine:
    """Coordinador de backtests con contratos explícitos de datos y latencia."""

    loader: DataLoader
    strategy: Strategy
    broker: Broker
    timeline: TradeTimeline = field(default_factory=lambda: TradeTimeline(fills=[]))

    def run(self) -> TradeTimeline:
        data = self.loader.load()
        self.strategy.prepare(data)

        fills: List[OrderFill] = []
        for bar_index in range(data.size):
            orders = self.strategy.on_bar(bar_index, data)
            fills.extend(self.broker.process_orders(orders, data))

        self.timeline.add_fills(fills)
        return self.timeline
