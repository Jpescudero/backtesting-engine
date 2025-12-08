from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.backtesting.core.models import MarketDataBatch, OrderRequest


class Strategy(ABC):
    """Contrato de estrategia que opera barra a barra.

    Las implementaciones deben respetar el orden temporal del dataset y
    documentar cualquier dependencia de latencia. El broker garantiza que los
    fills se producirán en el mismo orden en que se devuelven los órdenes.
    """

    def prepare(self, data: MarketDataBatch) -> None:  # pragma: no cover - hook opcional
        """Hook opcional de pre-cálculo (indicadores, máscaras, etc.)."""

    @abstractmethod
    def on_bar(self, bar_index: int, data: MarketDataBatch) -> Sequence[OrderRequest]:  # pragma: no cover - contrato
        """Genera órdenes para la barra actual."""
        raise NotImplementedError
