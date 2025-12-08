from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.backtesting.core.models import MarketDataBatch, OrderFill, OrderRequest


class Broker(ABC):
    """Contrato mínimo para brokers simulados.

    Deben aplicar la política de latencia documentada en la implementación y
    devolver fills en el mismo orden en que fueron recibidos.
    """

    @abstractmethod
    def process_orders(self, orders: Sequence[OrderRequest], data: MarketDataBatch) -> Sequence[OrderFill]:  # pragma: no cover
        """Procesa un batch de órdenes y devuelve los fills resultantes."""
        raise NotImplementedError
