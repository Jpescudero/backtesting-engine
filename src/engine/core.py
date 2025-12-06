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

    Parámetros clásicos:
      - initial_cash: equity inicial.
      - commission_per_trade: comisión fija por operación (entrada o salida).
      - trade_size: tamaño "por defecto" de la posición (en contratos/lotes).
      - slippage: deslizamiento en puntos (se suma/reste al precio de entrada/salida).
      - sl_pct / tp_pct: stop loss y take profit en % relativo al precio de entrada.
      - max_bars_in_trade: duración máxima de la operación en barras.

    Gestión de riesgo:
      - risk_per_trade_pct: % de la equity que queremos arriesgar en cada trade.
        Si es 0.0, se usa siempre trade_size fijo.
      - point_value: valor monetario de 1 punto de movimiento del subyacente.
        (Por ejemplo, 1.0 EUR/pt para muchos CFDs sobre índices).
      - margin_rate: % de nominal que se exige como margen.
        Si es 0.0, no se aplica chequeo de margen (comportamiento antiguo).
    """
    initial_cash: float = 100_000.0
    commission_per_trade: float = 1.0
    trade_size: float = 1.0
    slippage: float = 0.0

    # Gestión de riesgo
    sl_pct: float = 0.01
    tp_pct: float = 0.02
    max_bars_in_trade: int = 60

    # Parámetro de la estrategia de ejemplo
    entry_threshold: float = 0.001

    # Risk management avanzado
    risk_per_trade_pct: float = 0.0
    point_value: float = 1.0
    margin_rate: float = 0.0


@dataclass
class BacktestResult:
    """
    Resultado del backtest.
    """
    equity: np.ndarray            # serie de equity
    cash: float                   # efectivo final (cash interno del motor)
    position: float               # posición final
    trade_log: Dict[str, np.ndarray]  # arrays con info de los trades
    extra: Dict[str, Any]         # parámetros y metadatos


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


# =====================
# Helpers numba-friendly
# =====================

@njit
def _compute_position_size_long(
    equity_before: float,
    entry_price: float,
    sl_pct: float,
    trade_size: float,
    risk_per_trade_pct: float,
    point_value: float,
) -> float:
    """
    Calcula el tamaño de la posición (solo largos) en función del riesgo
    por trade. Si risk_per_trade_pct == 0.0, devuelve trade_size.
    """
    if risk_per_trade_pct <= 0.0 or sl_pct <= 0.0:
        return trade_size

    sl_price = entry_price * (1.0 - sl_pct)
    price_diff = entry_price - sl_price  # > 0 si sl_pct > 0

    if price_diff <= 0.0:
        return trade_size

    risk_capital = equity_before * risk_per_trade_pct
    risk_per_unit = price_diff * point_value

    if risk_per_unit <= 0.0:
        return trade_size

    qty = risk_capital / risk_per_unit
    return qty


@njit
def _compute_position_size_short(
    equity_before: float,
    entry_price: float,
    sl_pct: float,
    trade_size: float,
    risk_per_trade_pct: float,
    point_value: float,
) -> float:
    """
    Versión simétrica para cortos.
    Si risk_per_trade_pct == 0.0, devuelve trade_size.
    """
    if risk_per_trade_pct <= 0.0 or sl_pct <= 0.0:
        return trade_size

    # Para cortos, SL por encima del precio de entrada
    sl_price = entry_price * (1.0 + sl_pct)
    price_diff = sl_price - entry_price

    if price_diff <= 0.0:
        return trade_size

    risk_capital = equity_before * risk_per_trade_pct
    risk_per_unit = price_diff * point_value

    if risk_per_unit <= 0.0:
        return trade_size

    qty = risk_capital / risk_per_unit
    return qty


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
    slippage: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    max_bars_in_trade: int,
    risk_per_trade_pct: float,
    point_value: float,
    margin_rate: float,
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

            if sl_pct > 0.0 and ret_from_entry <= -sl_pct:
                reason = 1  # SL
            elif tp_pct > 0.0 and ret_from_entry >= tp_pct:
                reason = 2  # TP
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop

            if reason != 0:
                # Cerrar posición (largo)
                trade_price = price - slippage
                notional = trade_price * position * point_value
                cash += notional
                cash -= commission_per_trade

                realized_pnl = (trade_price - entry_price) * position * point_value - commission_per_trade
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
            notional = trade_price * position * point_value
            cash += notional
            cash -= commission_per_trade

            realized_pnl = (trade_price - entry_price) * position * point_value - commission_per_trade
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
            entry_px = price + slippage

            # Equity antes de abrir (no hay posición)
            equity_before = cash  # como no hay posición, equity = cash

            # Tamaño de la posición según el riesgo
            qty = _compute_position_size_long(
                equity_before=equity_before,
                entry_price=entry_px,
                sl_pct=sl_pct,
                trade_size=trade_size,
                risk_per_trade_pct=risk_per_trade_pct,
                point_value=point_value,
            )

            if qty > 0.0:
                notional = entry_px * qty * point_value

                allow = True
                if margin_rate > 0.0:
                    margin_required = margin_rate * notional
                    # Chequeo de margen: exigimos equity suficiente para cubrir margen + comisión
                    if equity_before < margin_required + commission_per_trade:
                        allow = False

                if allow:
                    # Abrir largo (contabilidad simple: restamos nominal y comisión)
                    cash -= notional
                    cash -= commission_per_trade

                    position = qty
                    entry_price = entry_px
                    entry_bar_idx = i

        # 3) Mark-to-market de la equity
        equity[i] = cash + position * price * point_value

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
# Interfaz de alto nivel (Python) para la estrategia de ejemplo
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
        slippage=config.slippage,
        entry_threshold=config.entry_threshold,
        sl_pct=config.sl_pct,
        tp_pct=config.tp_pct,
        max_bars_in_trade=config.max_bars_in_trade,
        risk_per_trade_pct=config.risk_per_trade_pct,
        point_value=config.point_value,
        margin_rate=config.margin_rate,
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
        "slippage": config.slippage,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
        "max_bars_in_trade": config.max_bars_in_trade,
        "entry_threshold": config.entry_threshold,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "point_value": config.point_value,
        "margin_rate": config.margin_rate,
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
    initial_cash: float,
    commission_per_trade: float,
    trade_size: float,
    slippage: float,
    sl_pct: float,
    tp_pct: float,
    max_bars_in_trade: int,
    risk_per_trade_pct: float,
    point_value: float,
    margin_rate: float,
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
          -1 -> señal de cierre (o entrada corta, no usada aquí)
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
        if position != 0.0:
            if position > 0.0:
                # Largo
                ret_from_entry = (price - entry_price) / entry_price
            else:
                # Corto (por si en el futuro lo usamos)
                ret_from_entry = (entry_price - price) / entry_price

            bars_in_trade = i - entry_bar_idx
            reason = 0

            if sl_pct > 0.0 and ret_from_entry <= -sl_pct:
                reason = 1  # SL
            elif tp_pct > 0.0 and ret_from_entry >= tp_pct:
                reason = 2  # TP
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop

            if reason != 0:
                # Cerrar posición
                if position > 0.0:
                    trade_price = price - slippage
                else:
                    trade_price = price + slippage

                notional = trade_price * position * point_value
                cash += notional
                cash -= commission_per_trade

                realized_pnl = (trade_price - entry_price) * position * point_value - commission_per_trade
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

        # Señal de cierre por señal contraria
        if sig == -1 and position != 0.0:
            if position > 0.0:
                trade_price = price - slippage
            else:
                trade_price = price + slippage

            notional = trade_price * position * point_value
            cash += notional
            cash -= commission_per_trade

            realized_pnl = (trade_price - entry_price) * position * point_value - commission_per_trade
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
