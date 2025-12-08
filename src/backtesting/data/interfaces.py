from __future__ import annotations

from abc import ABC, abstractmethod

from src.backtesting.core.models import MarketDataBatch


class DataLoader(ABC):
    """Contrato para ingestores de datos.

    Deben devolver un :class:`MarketDataBatch` ordenado y consistente. Los
    loaders son responsables de cualquier normalización previa (tipos, NaNs,
    gaps) y deben documentar la resolución temporal que ofrecen.
    """

    @abstractmethod
    def load(self) -> MarketDataBatch:  # pragma: no cover - contrato
        """Carga los datos históricos en memoria."""
        raise NotImplementedError
