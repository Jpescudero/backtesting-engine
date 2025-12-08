# src/strategies/base.py

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
