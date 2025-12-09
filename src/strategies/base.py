"""Base classes and containers for strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from src.data.feeds import OHLCVArrays


@dataclass
class StrategyResult:
    """
    Resultado de una estrategia discrecional / cuantitativa:

    - signals: array int8 del mismo tamaño que las barras:
        +1 -> abrir largo
        -1 -> cierre (o corto, en el futuro)
         0 -> nada

    - meta: diccionario para guardar información extra (p.ej. máscaras,
      parámetros usados, etc.).
    """

    signals: np.ndarray
    meta: Dict[str, Any]


class BaseStrategy:
    """
    Clase base muy sencilla: cualquier estrategia debe implementar
    generate_signals(data) y devolver un StrategyResult.
    """

    def generate_signals(self, data: OHLCVArrays) -> StrategyResult:
        raise NotImplementedError("Las estrategias deben implementar generate_signals().")


@dataclass
class SignalEntry:
    """Orden de entrada generada por estrategias barra a barra."""

    direction: str
    size: float = 1.0


class Strategy:
    """Interfaz mínima para estrategias barra a barra del motor nuevo."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.position = type("PositionState", (), {"is_open": False})()
        self.data: Any = None

    def preload(self, df: Any) -> None:  # pragma: no cover - interfaz
        """Hook opcional para precalcular indicadores."""

    def generate_signals(self, idx: int, row: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def on_fill(self, trade: Any) -> None:  # pragma: no cover
        """Callback al ejecutarse una orden."""

    def set_stop_loss(self, price: float) -> None:  # pragma: no cover
        """Registrar stop-loss en el motor."""
        raise NotImplementedError

    def set_take_profit(self, price: float) -> None:  # pragma: no cover
        """Registrar take-profit en el motor."""
        raise NotImplementedError
