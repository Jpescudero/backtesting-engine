"""Utilidades de gestión de riesgo para cálculo de tamaño de posición."""

from __future__ import annotations

import numpy as np


def compute_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    risk_pct: float = 0.0075,
    point_value: float = 1.0,
    rr_ref: float = 1.5,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> int:
    """
    Calcula el tamaño dinámico de la posición (número de contratos) basado en
    el riesgo máximo permitido, la distancia al SL y la distancia al TP.

    Parámetros
    ----------
    equity:
        Equity actual de la cuenta.
    entry_price:
        Precio de entrada de la operación.
    stop_loss:
        Nivel de stop loss de la operación.
    take_profit:
        Nivel de take profit de la operación.
    risk_pct:
        Porcentaje de equity a arriesgar en el stop loss. Por defecto 0.75%.
    point_value:
        Valor monetario de 1 punto del subyacente para 1 contrato. Por defecto 1.0.
    rr_ref:
        Ratio TP/SL de referencia para escalar el tamaño. Por defecto 1.5.
    min_scale:
        Límite inferior del factor de escala. Por defecto 0.5.
    max_scale:
        Límite superior del factor de escala. Por defecto 1.5.

    Retorna
    -------
    int
        Tamaño entero de la posición (número de contratos), no negativo.
    """

    sl_dist = abs(entry_price - stop_loss)
    tp_dist = abs(take_profit - entry_price)

    # Si el SL no define una distancia válida, no se abre posición.
    if sl_dist <= 0:
        return 0

    # Riesgo máximo en dinero permitido por trade.
    risk_cash = equity * risk_pct

    # Tamaño base limitado por la distancia al SL.
    units_base = risk_cash / (sl_dist * point_value)
    if units_base <= 0:
        return 0

    # Ratio teórico beneficio/riesgo y factor de escala asociado.
    rr = tp_dist / sl_dist
    scale = rr / rr_ref
    scale = max(min_scale, min(max_scale, scale))

    # Tamaño final entero, sin permitir valores negativos.
    units = int(np.floor(units_base * scale))
    return max(units, 0)
