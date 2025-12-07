# src/engine/core.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from numba import njit

from src.data.feeds import OHLCVArrays


# =====================
# Configuración & Resultados
# =====================

@dataclass
class BacktestConfig:
    """
    Configuración básica del backtest.
    """
    initial_cash: float = 100_000.0
    commission_per_trade: float = 1.0   # comisión fija por operación (entrada o salida)
    trade_size: float = 1.0             # contratos/unidades por operación
    min_trade_size: float = 0.01        # tamaño mínimo permitido por contrato/lote
    max_trade_size: float = 1000.0      # límite superior para el tamaño por operación
    slippage: float = 0.0               # slippage en puntos

    # Gestión de riesgo
    sl_pct: float = 0.01                # stop loss a -1%
    tp_pct: float = 0.02                # take profit a +2%
    risk_per_trade_pct: float = 0.0     # riesgo fijo por trade (0.0025 => 0.25%)
    atr_stop_mult: float = 0.0          # múltiplo de ATR para calcular el SL
    atr_tp_mult: float = 0.0            # múltiplo de ATR para calcular el TP
    point_value: float = 1.0            # valor monetario de 1 punto para 1.0 contrato
    max_bars_in_trade: int = 60         # duración máxima del trade en barras

    # Parámetros de la estrategia de ejemplo
    entry_threshold: float = 0.001      # 0.1% de subida respecto al cierre anterior para entrar


@dataclass
class BacktestResult:
    """
    Resultado del backtest.
    """
    equity: np.ndarray                  # serie de equity
    cash: float                         # efectivo final
    position: float                     # posición final
    trade_log: Dict[str, np.ndarray]    # arrays con info de los trades
    extra: Dict[str, Any]               # parámetros y metadatos


# =====================
# Estrategia de ejemplo (Numba)
# =====================

@njit
def _example_strategy_long_on_up_move(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    entry_threshold: float,
) -> np.ndarray:
    """
    Estrategia de ejemplo: genera señales según el movimiento de la barra actual
    respecto al cierre anterior.

    Devuelve:
        +1 -> señal de compra
        -1 -> señal de venta/cierre
         0 -> nada
    """
    n = c.shape[0]
    signals = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        if not np.isfinite(c[i]) or not np.isfinite(c[i - 1]):
            signals[i] = 0
            continue

        ret = (c[i] - c[i - 1]) / c[i - 1]

        if ret > entry_threshold:
            signals[i] = 1
        elif ret < -entry_threshold:
            signals[i] = -1

    return signals


@njit
def _calculate_affordable_qty(
    available_cash: float,
    price_per_unit: float,
    commission_per_trade: float,
    desired_qty: float,
    min_trade_size: float,
) -> float:
    """
    Devuelve la cantidad máxima que se puede abrir sin pasar a efectivo negativo.

    La cantidad resultante se ajusta a múltiplos de min_trade_size para permitir
    operar con fracciones de contrato (p.ej. lotes de 0.01).
    """
    effective_cash = available_cash - commission_per_trade
    if effective_cash <= 0.0 or price_per_unit <= 0.0 or desired_qty <= 0.0:
        return 0.0

    max_units = np.floor((effective_cash / price_per_unit) / min_trade_size) * min_trade_size
    if max_units < min_trade_size:
        return 0.0

    return max_units if max_units < desired_qty else desired_qty


@njit
def _round_to_step(value: float, step: float) -> float:
    if step <= 0.0:
        return value
    return np.round(value / step) * step


@njit
def _clip_size(value: float, min_trade_size: float, max_trade_size: float) -> float:
    if value < min_trade_size:
        return 0.0
    if max_trade_size > 0.0 and value > max_trade_size:
        return max_trade_size
    return value


@njit
def _compute_risk_based_qty(
    equity: float,
    atr_value: float,
    atr_stop_mult: float,
    point_value: float,
    risk_per_trade_pct: float,
    min_trade_size: float,
    max_trade_size: float,
) -> float:
    """
    Tamaño de posición ajustado a la volatilidad:

    lotes = (equity * risk_per_trade_pct) / (atr_stop_mult * ATR * point_value)

    Se redondea al múltiplo de min_trade_size más cercano y se recorta con un
    máximo opcional.
    """
    if (
        equity <= 0.0
        or atr_value <= 0.0
        or atr_stop_mult <= 0.0
        or point_value <= 0.0
        or risk_per_trade_pct <= 0.0
    ):
        return 0.0

    stop_distance = atr_stop_mult * atr_value
    loss_per_unit = stop_distance * point_value

    if loss_per_unit <= 0.0:
        return 0.0

    theoretical_qty = (equity * risk_per_trade_pct) / loss_per_unit
    rounded_qty = _round_to_step(theoretical_qty, min_trade_size)

    return _clip_size(rounded_qty, min_trade_size, max_trade_size)


# =====================
# Motor de backtest con SL/TP & duración (Numba)
# =====================

@njit
def _backtest_with_risk(
    ts: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    initial_cash: float,
    commission_per_trade: float,
    trade_size: float,
    min_trade_size: float,
    slippage: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    max_bars_in_trade: int,
) -> Tuple[
    np.ndarray,  # equity
    float,       # cash final
    float,       # posición final
    int,         # número de trades
    np.ndarray,  # trade_entry_idx
    np.ndarray,  # trade_exit_idx
    np.ndarray,  # trade_entry_price
    np.ndarray,  # trade_exit_price
    np.ndarray,  # trade_qty
    np.ndarray,  # trade_pnl
    np.ndarray,  # trade_holding_bars
    np.ndarray,  # trade_exit_reason
]:
    """
    Motor de backtest con:
      - Estrategia simple (_example_strategy_long_on_up_move).
      - SL / TP en % desde precio de entrada.
      - Máxima duración del trade.
      - Registro de trades.

    exit_reason:
      1 -> Stop Loss
      2 -> Take Profit
      3 -> Time Stop (duración máxima)
      4 -> Señal contraria
    """
    n = c.shape[0]
    equity = np.empty(n, dtype=np.float64)

    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    entry_bar_idx = -1

    # Señales de la estrategia
    signals = _example_strategy_long_on_up_move(o, h, l, c, v, entry_threshold)

    # Prealocación para el log de trades
    max_trades = n // 2 + 1

    trade_entry_idx = np.empty(max_trades, dtype=np.int64)
    trade_exit_idx = np.empty(max_trades, dtype=np.int64)
    trade_entry_price = np.empty(max_trades, dtype=np.float64)
    trade_exit_price = np.empty(max_trades, dtype=np.float64)
    trade_qty = np.empty(max_trades, dtype=np.float64)
    trade_pnl = np.empty(max_trades, dtype=np.float64)
    trade_holding_bars = np.empty(max_trades, dtype=np.int64)
    trade_exit_reason = np.empty(max_trades, dtype=np.int8)

    trade_count = 0

    for i in range(n):
        price = c[i]

        # --- Defensa: si el precio no es finito, saltamos la barra ---
        if not np.isfinite(price):
            if i == 0:
                equity[i] = initial_cash
            else:
                equity[i] = equity[i - 1]
            continue
        # -------------------------------------------------------------

        # 1) Comprobar SL / TP / max_bars antes de nuevas señales
        if position > 0.0:
            ret_from_entry = (price - entry_price) / entry_price
            bars_in_trade = i - entry_bar_idx

            reason = 0

            if ret_from_entry <= -sl_pct:
                reason = 1  # SL
            elif ret_from_entry >= tp_pct:
                reason = 2  # TP
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop

            if reason != 0:
                # Cerrar posición
                trade_price = price - slippage
                cash += trade_price * position
                cash -= commission_per_trade

                realized_pnl = (trade_price - entry_price) * position - commission_per_trade * 1.0
                hold_bars = bars_in_trade

                if trade_count < max_trades:
                    trade_entry_idx[trade_count] = entry_bar_idx
                    trade_exit_idx[trade_count] = i
                    trade_entry_price[trade_count] = entry_price
                    trade_exit_price[trade_count] = trade_price
                    trade_qty[trade_count] = position
                    trade_pnl[trade_count] = realized_pnl
                    trade_holding_bars[trade_count] = hold_bars
                    trade_exit_reason[trade_count] = reason
                    trade_count += 1

                position = 0.0
                entry_price = 0.0
                entry_bar_idx = -1

        # 2) Procesar señal de la estrategia (compra/venta)
        sig = signals[i]

        # Señal de venta: cerrar por señal contraria
        if sig == -1 and position > 0.0:
            trade_price = price - slippage
            cash += trade_price * position
            cash -= commission_per_trade

            realized_pnl = (trade_price - entry_price) * position - commission_per_trade * 1.0
            hold_bars = i - entry_bar_idx

            if trade_count < max_trades:
                trade_entry_idx[trade_count] = entry_bar_idx
                trade_exit_idx[trade_count] = i
                trade_entry_price[trade_count] = entry_price
                trade_exit_price[trade_count] = trade_price
                trade_qty[trade_count] = position
                trade_pnl[trade_count] = realized_pnl
                trade_holding_bars[trade_count] = hold_bars
                trade_exit_reason[trade_count] = 4  # señal contraria
                trade_count += 1

            position = 0.0
            entry_price = 0.0
            entry_bar_idx = -1

        # Señal de compra: abrir si no hay posición
        if sig == 1 and position == 0.0:
            trade_price = price + slippage
            qty = _calculate_affordable_qty(
                available_cash=cash,
                price_per_unit=trade_price,
                commission_per_trade=commission_per_trade,
                desired_qty=trade_size,
                min_trade_size=min_trade_size,
            )
            if qty > 0.0:
                cash -= trade_price * qty
                cash -= commission_per_trade
                position = qty
                entry_price = trade_price
                entry_bar_idx = i

        # 3) Mark-to-market de la equity
        equity[i] = cash + position * price

    return (
        equity,
        cash,
        position,
        trade_count,
        trade_entry_idx,
        trade_exit_idx,
        trade_entry_price,
        trade_exit_price,
        trade_qty,
        trade_pnl,
        trade_holding_bars,
        trade_exit_reason,
    )


# =====================
# Interfaz de alto nivel (Python)
# =====================

def run_backtest_basic(
    data: OHLCVArrays,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Ejecuta un backtest con:
      - Estrategia de ejemplo basada en momentum.
      - SL / TP en %.
      - Máxima duración del trade.
      - Log de trades completo.
    """
    if config is None:
        config = BacktestConfig()

    (
        equity,
        cash,
        position,
        trade_count,
        trade_entry_idx,
        trade_exit_idx,
        trade_entry_price,
        trade_exit_price,
        trade_qty,
        trade_pnl,
        trade_holding_bars,
        trade_exit_reason,
    ) = _backtest_with_risk(
        ts=data.ts,
        o=data.o,
        h=data.h,
        l=data.l,
        c=data.c,
        v=data.v,
        initial_cash=config.initial_cash,
        commission_per_trade=config.commission_per_trade,
        trade_size=config.trade_size,
        min_trade_size=config.min_trade_size,
        slippage=config.slippage,
        entry_threshold=config.entry_threshold,
        sl_pct=config.sl_pct,
        tp_pct=config.tp_pct,
        max_bars_in_trade=config.max_bars_in_trade,
    )

    trade_log: Dict[str, np.ndarray] = {}
    if trade_count > 0:
        trade_log = {
            "entry_idx": trade_entry_idx[:trade_count],
            "exit_idx": trade_exit_idx[:trade_count],
            "entry_price": trade_entry_price[:trade_count],
            "exit_price": trade_exit_price[:trade_count],
            "qty": trade_qty[:trade_count],
            "pnl": trade_pnl[:trade_count],
            "holding_bars": trade_holding_bars[:trade_count],
            "exit_reason": trade_exit_reason[:trade_count],
        }

    extra: Dict[str, Any] = {
        "initial_cash": config.initial_cash,
        "commission_per_trade": config.commission_per_trade,
        "trade_size": config.trade_size,
        "min_trade_size": config.min_trade_size,
        "max_trade_size": config.max_trade_size,
        "slippage": config.slippage,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "atr_stop_mult": config.atr_stop_mult,
        "atr_tp_mult": config.atr_tp_mult,
        "point_value": config.point_value,
        "max_bars_in_trade": config.max_bars_in_trade,
        "entry_threshold": config.entry_threshold,
        "n_trades": trade_count,
    }

    return BacktestResult(
        equity=equity,
        cash=cash,
        position=position,
        trade_log=trade_log,
        extra=extra,
    )
# =====================
# Motor de backtest usando SEÑALES externas (Numba)
# =====================

@njit
def _backtest_with_risk_from_signals(
    ts: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    signals: np.ndarray,
    atr: np.ndarray,
    initial_cash: float,
    commission_per_trade: float,
    trade_size: float,
    min_trade_size: float,
    max_trade_size: float,
    slippage: float,
    sl_pct: float,
    tp_pct: float,
    risk_per_trade_pct: float,
    atr_stop_mult: float,
    atr_tp_mult: float,
    point_value: float,
    max_bars_in_trade: int,
) -> Tuple[
    np.ndarray,  # equity
    float,       # cash final
    float,       # posición final
    int,         # número de trades
    np.ndarray,  # trade_entry_idx
    np.ndarray,  # trade_exit_idx
    np.ndarray,  # trade_entry_price
    np.ndarray,  # trade_exit_price
    np.ndarray,  # trade_qty
    np.ndarray,  # trade_pnl
    np.ndarray,  # trade_holding_bars
    np.ndarray,  # trade_exit_reason
]:
    """
    Igual que _backtest_with_risk, pero usando un array de señales externo.

    Parámetros clave:
      - signals: array int8 del mismo tamaño que c:
          +1 -> señal de compra/entrada larga
          -1 -> cierre por señal contraria
           0 -> nada

    exit_reason:
      1 -> Stop Loss
      2 -> Take Profit
      3 -> Time Stop (duración máxima)
      4 -> Señal contraria
    """
    n = c.shape[0]
    equity = np.empty(n, dtype=np.float64)

    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    entry_bar_idx = -1
    stop_price = 0.0
    tp_price = 0.0
    use_atr_stops = False

    # Prealocación para el log de trades
    max_trades = n // 2 + 1

    trade_entry_idx = np.empty(max_trades, dtype=np.int64)
    trade_exit_idx = np.empty(max_trades, dtype=np.int64)
    trade_entry_price = np.empty(max_trades, dtype=np.float64)
    trade_exit_price = np.empty(max_trades, dtype=np.float64)
    trade_qty = np.empty(max_trades, dtype=np.float64)
    trade_pnl = np.empty(max_trades, dtype=np.float64)
    trade_holding_bars = np.empty(max_trades, dtype=np.int64)
    trade_exit_reason = np.empty(max_trades, dtype=np.int8)

    trade_count = 0

    for i in range(n):
        price = c[i]

        # --- Defensa: si el precio no es finito, saltamos la barra ---
        if not np.isfinite(price):
            if i == 0:
                equity[i] = initial_cash
            else:
                equity[i] = equity[i - 1]
            continue
        # -------------------------------------------------------------

        # 1) Comprobar SL / TP / max_bars antes de nuevas señales
        if position > 0.0:
            bars_in_trade = i - entry_bar_idx

            reason = 0

            if use_atr_stops:
                if stop_price > 0.0 and price <= stop_price:
                    reason = 1  # SL
                elif tp_price > 0.0 and price >= tp_price:
                    reason = 2  # TP
            else:
                ret_from_entry = (price - entry_price) / entry_price
                if ret_from_entry <= -sl_pct:
                    reason = 1  # SL
                elif ret_from_entry >= tp_pct:
                    reason = 2  # TP

            if reason == 0 and bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop

            if reason != 0:
                # Cerrar posición
                trade_price = price - slippage
                cash += trade_price * position
                cash -= commission_per_trade

                realized_pnl = (trade_price - entry_price) * position - commission_per_trade * 1.0
                hold_bars = bars_in_trade

                if trade_count < max_trades:
                    trade_entry_idx[trade_count] = entry_bar_idx
                    trade_exit_idx[trade_count] = i
                    trade_entry_price[trade_count] = entry_price
                    trade_exit_price[trade_count] = trade_price
                    trade_qty[trade_count] = position
                    trade_pnl[trade_count] = realized_pnl
                    trade_holding_bars[trade_count] = hold_bars
                    trade_exit_reason[trade_count] = reason
                    trade_count += 1

                position = 0.0
                entry_price = 0.0
                entry_bar_idx = -1
                stop_price = 0.0
                tp_price = 0.0
                use_atr_stops = False

        # 2) Procesar señal de la estrategia (compra/venta)
        sig = signals[i]

        # Señal de venta: cerrar por señal contraria
        if sig == -1 and position > 0.0:
            trade_price = price - slippage
            cash += trade_price * position
            cash -= commission_per_trade

            realized_pnl = (trade_price - entry_price) * position - commission_per_trade * 1.0
            hold_bars = i - entry_bar_idx

            if trade_count < max_trades:
                trade_entry_idx[trade_count] = entry_bar_idx
                trade_exit_idx[trade_count] = i
                trade_entry_price[trade_count] = entry_price
                trade_exit_price[trade_count] = trade_price
                trade_qty[trade_count] = position
                trade_pnl[trade_count] = realized_pnl
                trade_holding_bars[trade_count] = hold_bars
                trade_exit_reason[trade_count] = 4  # señal contraria
                trade_count += 1

            position = 0.0
            entry_price = 0.0
            entry_bar_idx = -1

        # Señal de compra: abrir posición si estamos flat
        if sig == 1 and position == 0.0:
            trade_price = price + slippage
            atr_val = atr[i] if i < atr.shape[0] else np.nan
            has_atr = np.isfinite(atr_val)

            desired_qty = trade_size
            if risk_per_trade_pct > 0.0 and has_atr and atr_stop_mult > 0.0:
                current_equity = cash + position * price
                risk_qty = _compute_risk_based_qty(
                    equity=current_equity,
                    atr_value=atr_val,
                    atr_stop_mult=atr_stop_mult,
                    point_value=point_value,
                    risk_per_trade_pct=risk_per_trade_pct,
                    min_trade_size=min_trade_size,
                    max_trade_size=max_trade_size,
                )
                if risk_qty > 0.0:
                    desired_qty = risk_qty

            desired_qty = _clip_size(desired_qty, min_trade_size, max_trade_size)

            qty = _calculate_affordable_qty(
                available_cash=cash,
                price_per_unit=trade_price,
                commission_per_trade=commission_per_trade,
                desired_qty=desired_qty,
                min_trade_size=min_trade_size,
            )
            if qty > 0.0:
                position = qty
                cash -= trade_price * position
                cash -= commission_per_trade
                entry_price = trade_price
                entry_bar_idx = i

                if has_atr and atr_stop_mult > 0.0:
                    stop_distance = atr_stop_mult * atr_val
                    stop_price = trade_price - stop_distance
                    tp_price = 0.0
                    if atr_tp_mult > 0.0:
                        tp_price = trade_price + atr_tp_mult * atr_val
                    use_atr_stops = stop_distance > 0.0
                else:
                    stop_price = 0.0
                    tp_price = 0.0
                    use_atr_stops = False

        # 3) Mark-to-market de la equity
        equity[i] = cash + position * price

    return (
        equity,
        cash,
        position,
        trade_count,
        trade_entry_idx,
        trade_exit_idx,
        trade_entry_price,
        trade_exit_price,
        trade_qty,
        trade_pnl,
        trade_holding_bars,
        trade_exit_reason,
    )


# =====================
# Interfaz de alto nivel usando señales externas
# =====================

def run_backtest_with_signals(
    data: OHLCVArrays,
    signals: np.ndarray,
    atr: np.ndarray | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Ejecuta un backtest usando un array de señales externo (int8: -1, 0, +1).

    Es igual que run_backtest_basic, pero:
      - No calcula la estrategia dentro del motor.
      - Usa las señales proporcionadas.
    """
    if config is None:
        config = BacktestConfig()

    if signals.shape[0] != data.c.shape[0]:
        raise ValueError(
            f"El tamaño de signals ({signals.shape[0]}) no coincide con el número de barras ({data.c.shape[0]})."
        )

    if atr is None:
        atr = np.full_like(data.c, np.nan, dtype=float)
    elif atr.shape[0] != data.c.shape[0]:
        raise ValueError(
            f"El tamaño de atr ({atr.shape[0]}) no coincide con el número de barras ({data.c.shape[0]})."
        )

    (
        equity,
        cash,
        position,
        trade_count,
        trade_entry_idx,
        trade_exit_idx,
        trade_entry_price,
        trade_exit_price,
        trade_qty,
        trade_pnl,
        trade_holding_bars,
        trade_exit_reason,
    ) = _backtest_with_risk_from_signals(
        ts=data.ts,
        o=data.o,
        h=data.h,
        l=data.l,
        c=data.c,
        v=data.v,
        signals=signals.astype(np.int8),
        atr=atr,
        initial_cash=config.initial_cash,
        commission_per_trade=config.commission_per_trade,
        trade_size=config.trade_size,
        min_trade_size=config.min_trade_size,
        max_trade_size=config.max_trade_size,
        slippage=config.slippage,
        sl_pct=config.sl_pct,
        tp_pct=config.tp_pct,
        risk_per_trade_pct=config.risk_per_trade_pct,
        atr_stop_mult=config.atr_stop_mult,
        atr_tp_mult=config.atr_tp_mult,
        point_value=config.point_value,
        max_bars_in_trade=config.max_bars_in_trade,
    )

    trade_log: Dict[str, np.ndarray] = {}
    if trade_count > 0:
        trade_log = {
            "entry_idx": trade_entry_idx[:trade_count],
            "exit_idx": trade_exit_idx[:trade_count],
            "entry_price": trade_entry_price[:trade_count],
            "exit_price": trade_exit_price[:trade_count],
            "qty": trade_qty[:trade_count],
            "pnl": trade_pnl[:trade_count],
            "holding_bars": trade_holding_bars[:trade_count],
            "exit_reason": trade_exit_reason[:trade_count],
        }

    extra: Dict[str, Any] = {
        "initial_cash": config.initial_cash,
        "commission_per_trade": config.commission_per_trade,
        "trade_size": config.trade_size,
        "min_trade_size": config.min_trade_size,
        "max_trade_size": config.max_trade_size,
        "slippage": config.slippage,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "atr_stop_mult": config.atr_stop_mult,
        "atr_tp_mult": config.atr_tp_mult,
        "point_value": config.point_value,
        "max_bars_in_trade": config.max_bars_in_trade,
        "entry_threshold": config.entry_threshold,
        "n_trades": trade_count,
    }

    return BacktestResult(
        equity=equity,
        cash=cash,
        position=position,
        trade_log=trade_log,
        extra=extra,
    )
