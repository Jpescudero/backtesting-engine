from __future__ import annotations

from datetime import datetime
from typing import List, Sequence

from src.backtesting.core.models import LatencyProfile, MarketDataBatch, OrderFill, OrderRequest
from src.backtesting.execution.interfaces import Broker


class SimulatedBroker(Broker):
    """Broker simple con latencia configurable y órdenes de mercado.

    La latencia se aplica a cada orden de forma independiente manteniendo el
    orden de llegada. Los fills usan el precio de cierre de la barra asociada
    al ``bar_index`` de la orden.
    """

    def __init__(
        self,
        latency: LatencyProfile | None = None,
        initial_cash: float = float("inf"),
        max_notional_per_order: float | None = None,
    ) -> None:
        self.latency = latency or LatencyProfile()
        self.cash = initial_cash
        self.max_notional_per_order = max_notional_per_order

    def process_orders(self, orders: Sequence[OrderRequest], data: MarketDataBatch) -> Sequence[OrderFill]:
        fills: List[OrderFill] = []
        for order in orders:
            order.validate(max_index=data.size)
            submitted_at: datetime = order.timestamp
            executed_at: datetime = self.latency.apply(submitted_at)
            price = float(data.close[order.bar_index])
            notional = order.quantity * price

            if self.max_notional_per_order is not None and notional > self.max_notional_per_order:
                raise ValueError(
                    f"Límite de riesgo excedido: notional {notional:.2f} > {self.max_notional_per_order:.2f}"
                )

            if order.side == "buy" and notional > self.cash:
                raise ValueError(
                    f"Fondos insuficientes para comprar {order.quantity} @ {price:.2f}; disponible {self.cash:.2f}"
                )

            fills.append(
                OrderFill(
                    order=order,
                    filled_quantity=order.quantity,
                    avg_price=price,
                    status="filled",
                    submitted_at=submitted_at,
                    executed_at=executed_at,
                )
            )

            if order.side == "buy":
                self.cash -= notional
            else:
                self.cash += notional
        return fills
