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
    slippage: float = 0.0               # slippage en puntos

    # Gestión de riesgo
    sl_pct: float = 0.01                # stop loss a -1%
    tp_pct: float = 0.02                # take profit a +2%
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
            cost = trade_price * trade_size + commission_per_trade
            if cash >= cost:
                cash -= cost
                position = trade_size
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
        "slippage": config.slippage,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
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
