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

    Notas importantes:
      - qty (posición) se interpreta como LOTES.
      - point_value = € por punto para 1.0 lote.
        Ejemplo típico en CFDs de índices en MT:
          1 lote = 1 €/punto  -> point_value = 1.0
          1 lote = 2 €/punto  -> point_value = 2.0
      - risk_per_trade_pct controla el % de equity arriesgado por operación.
    """

    initial_cash: float = 100_000.0
    commission_per_trade: float = 1.0  # comisión fija por operación (entrada o salida)

    # Tamaño fijo por operación (sólo se usa si risk_per_trade_pct <= 0)
    trade_size: float = 1.0  # LOTES

    # Slippage en puntos de precio
    slippage: float = 0.0

    # Gestión de riesgo
    sl_pct: float = 0.01  # stop loss a -1%
    tp_pct: float = 0.02  # take profit a +2%
    max_bars_in_trade: int = 60  # duración máxima del trade en barras

    # NUEVO: gestión del tamaño por % de equity
    #   0.0 -> desactivado (usa trade_size fijo)
    #   0.01 -> arriesga ~1% de la equity por operación
    risk_per_trade_pct: float = 0.0

    # Valor monetario de un punto de precio para 1.0 lote (€/punto).
    # Ejemplo MetaTrader común: 1 lote NAS100 = 1 €/punto -> point_value = 1.0
    point_value: float = 1.0

    # Parámetros de la estrategia de ejemplo (sólo para run_backtest_basic)
    entry_threshold: float = 0.001  # 0.1% de subida respecto al cierre anterior


@dataclass
class BacktestResult:
    """
    Resultado del backtest.
    """

    equity: np.ndarray  # serie de equity
    cash: float  # efectivo final
    position: float  # posición final (en LOTES, signo incluye largo/corto)
    trade_log: Dict[str, np.ndarray]  # arrays con info de los trades
    extra: Dict[str, Any]  # parámetros y metadatos


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
    risk_per_trade_pct: float,
    point_value: float,
) -> Tuple[
    np.ndarray,  # equity
    float,  # cash final
    float,  # posición final
    int,  # número de trades
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
    - SL / TP en % desde precio de entrada (intrabar).
    - Máxima duración del trade.
    - Registro de trades.
    - Position sizing por porcentaje de equity (fracciones de lote permitidas).

    exit_reason:
        1 -> Stop Loss
        2 -> Take Profit
        3 -> Time Stop (duración máxima)
        4 -> Señal contraria
    """
    n = c.shape[0]
    equity = np.empty(n, dtype=np.float64)

    cash = initial_cash
    position = 0.0  # >0 largo
    entry_price = 0.0
    entry_bar_idx = -1

    signals = _example_strategy_long_on_up_move(o, h, l, c, v, entry_threshold)

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

        if not np.isfinite(price):
            if i == 0:
                equity[i] = initial_cash
            else:
                equity[i] = equity[i - 1]
            continue

        # ---- 1) Gestión de SL/TP/time_stop (solo largos aquí) ----
        if position > 0.0:
            bars_in_trade = i - entry_bar_idx
            reason = 0
            trade_price = price

            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)

            hit_sl = l[i] <= sl_price
            hit_tp = h[i] >= tp_price

            if hit_sl or hit_tp:
                if hit_sl:
                    reason = 1
                    raw_exit = sl_price
                else:
                    reason = 2
                    raw_exit = tp_price

                trade_price = raw_exit - slippage
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3
                trade_price = price - slippage

            if reason != 0:
                cash += trade_price * position * point_value
                cash -= commission_per_trade

                realized_pnl = (
                    (trade_price - entry_price) * position * point_value
                    - 2.0 * commission_per_trade
                )
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

        # ---- 2) Señal de estrategia ----
        sig = signals[i]

        # Cierre por señal contraria
        if sig == -1 and position > 0.0:
            trade_price = price - slippage
            cash += trade_price * position * point_value
            cash -= commission_per_trade

            realized_pnl = (
                (trade_price - entry_price) * position * point_value
                - 2.0 * commission_per_trade
            )
            hold_bars = i - entry_bar_idx

            if trade_count < max_trades:
                trade_entry_idx[trade_count] = entry_bar_idx
                trade_exit_idx[trade_count] = i
                trade_entry_price[trade_count] = entry_price
                trade_exit_price[trade_count] = trade_price
                trade_qty[trade_count] = position
                trade_pnl[trade_count] = realized_pnl
                trade_holding_bars[trade_count] = hold_bars
                trade_exit_reason[trade_count] = 4
                trade_count += 1

            position = 0.0
            entry_price = 0.0
            entry_bar_idx = -1

        # Apertura de largos
        if sig == 1 and position == 0.0:
            entry_px = price + slippage

            # Distancia al SL para calcular riesgo por lote
            sl_price = entry_px * (1.0 - sl_pct)
            price_diff = abs(entry_px - sl_price) * point_value  # € de riesgo por 1.0 lote

            if risk_per_trade_pct > 0.0 and price_diff > 0.0:
                equity_before = cash  # position == 0
                risk_capital = risk_per_trade_pct * equity_before
                qty = risk_capital / price_diff  # LOTES (fracción permitida)
            else:
                qty = trade_size  # LOTES

            if qty > 0.0:
                cost = entry_px * qty * point_value + commission_per_trade
                if cash >= cost:
                    cash -= entry_px * qty * point_value
                    cash -= commission_per_trade
                    position = qty
                    entry_price = entry_px
                    entry_bar_idx = i

        # ---- 3) Mark-to-market ----
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
# Interfaz de alto nivel (Python) – estrategia interna
# =====================


def run_backtest_basic(
    data: OHLCVArrays,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Ejecuta un backtest con:
    - Estrategia de ejemplo basada en momentum.
    - SL / TP en % (intrabar).
    - Máxima duración del trade.
    - Position sizing fijo o por % de equity (con fracciones de lote).
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
# Motor usando SEÑALES externas (Numba)
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
) -> Tuple[
    np.ndarray,  # equity
    float,  # cash final
    float,  # posición final
    int,  # número de trades
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

    signals:
        +1 -> señal de entrada LARGA
        -1 -> señal de entrada CORTA
         0 -> nada

    Permite fracciones de lote en el position sizing.
    """
    n = c.shape[0]
    equity = np.empty(n, dtype=np.float64)

    cash = initial_cash
    position = 0.0  # >0 largo, <0 corto (en LOTES)
    entry_price = 0.0
    entry_bar_idx = -1

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

        if not np.isfinite(price):
            if i == 0:
                equity[i] = initial_cash
            else:
                equity[i] = equity[i - 1]
            continue

        # ---- 1) Gestión de SL/TP/time_stop ----
        if position != 0.0:
            bars_in_trade = i - entry_bar_idx
            reason = 0
            trade_price = price

            sign = 1.0 if position > 0.0 else -1.0

            if sign > 0.0:
                sl_price = entry_price * (1.0 - sl_pct)
                tp_price = entry_price * (1.0 + tp_pct)

                hit_sl = l[i] <= sl_price
                hit_tp = h[i] >= tp_price
            else:
                sl_price = entry_price * (1.0 + sl_pct)
                tp_price = entry_price * (1.0 - tp_pct)

                hit_sl = h[i] >= sl_price
                hit_tp = l[i] <= tp_price

            if hit_sl or hit_tp:
                if hit_sl:
                    reason = 1
                    raw_exit = sl_price
                else:
                    reason = 2
                    raw_exit = tp_price

                if sign > 0.0:
                    trade_price = raw_exit - slippage
                else:
                    trade_price = raw_exit + slippage
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3
                if sign > 0.0:
                    trade_price = price - slippage
                else:
                    trade_price = price + slippage

            if reason != 0:
                cash += trade_price * position * point_value
                cash -= commission_per_trade

                realized_pnl = (
                    (trade_price - entry_price) * position * point_value
                    - 2.0 * commission_per_trade
                )
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
                equity[i] = cash
                continue

        # ---- 2) Señales externas ----
        sig = signals[i]

        # Cierre por señal contraria si estamos en posición
        if position != 0.0:
            sign = 1.0 if position > 0.0 else -1.0
            close_by_signal = (
                (sign > 0.0 and sig == -1) or (sign < 0.0 and sig == 1)
            )
            if close_by_signal:
                if sign > 0.0:
                    trade_price = price - slippage
                else:
                    trade_price = price + slippage

                cash += trade_price * position * point_value
                cash -= commission_per_trade

                realized_pnl = (
                    (trade_price - entry_price) * position * point_value
                    - 2.0 * commission_per_trade
                )
                hold_bars = i - entry_bar_idx

                if trade_count < max_trades:
                    trade_entry_idx[trade_count] = entry_bar_idx
                    trade_exit_idx[trade_count] = i
                    trade_entry_price[trade_count] = entry_price
                    trade_exit_price[trade_count] = trade_price
                    trade_qty[trade_count] = position
                    trade_pnl[trade_count] = realized_pnl
                    trade_holding_bars[trade_count] = hold_bars
                    trade_exit_reason[trade_count] = 4
                    trade_count += 1

                position = 0.0
                entry_price = 0.0
                entry_bar_idx = -1

        # Aperturas (solo si estamos flat)
        if position == 0.0:
            if sig == 1 or sig == -1:
                sign = 1.0 if sig == 1 else -1.0

                if sign > 0.0:
                    entry_px = price + slippage
                    sl_price = entry_px * (1.0 - sl_pct)
                else:
                    entry_px = price - slippage
                    sl_price = entry_px * (1.0 + sl_pct)

                price_diff = abs(entry_px - sl_price) * point_value  # € riesgo por 1.0 lote

                if risk_per_trade_pct > 0.0 and price_diff > 0.0:
                    equity_before = cash  # position == 0
                    risk_capital = risk_per_trade_pct * equity_before
                    qty = risk_capital / price_diff  # LOTES (puede ser fracción)
                else:
                    qty = trade_size

                if qty > 0.0:
                    gross = entry_px * qty * point_value

                    if sign > 0.0:
                        cost_ok = cash >= (gross + commission_per_trade)
                        if cost_ok:
                            cash -= gross
                            cash -= commission_per_trade
                            position = sign * qty
                            entry_price = entry_px
                            entry_bar_idx = i
                    else:
                        # cortos: ingresamos efectivo al vender
                        cash += gross
                        cash -= commission_per_trade
                        position = sign * qty
                        entry_price = entry_px
                        entry_bar_idx = i

        # ---- 3) Mark-to-market ----
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
# Interfaz de alto nivel usando señales externas
# =====================


def run_backtest_with_signals(
    data: OHLCVArrays,
    signals: np.ndarray,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Ejecuta un backtest usando un array de señales externo (int8: -1, 0, +1).

    - SL/TP intrabar.
    - Máxima duración del trade.
    - Position sizing fijo o por % de equity (con fracciones de lote).
    """
    if config is None:
        config = BacktestConfig()

    if signals.shape[0] != data.c.shape[0]:
        raise ValueError(
            f"El tamaño de signals ({signals.shape[0]}) no coincide con "
            f"el número de barras ({data.c.shape[0]})."
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
        initial_cash=config.initial_cash,
        commission_per_trade=config.commission_per_trade,
        trade_size=config.trade_size,
        slippage=config.slippage,
        sl_pct=config.sl_pct,
        tp_pct=config.tp_pct,
        max_bars_in_trade=config.max_bars_in_trade,
        risk_per_trade_pct=config.risk_per_trade_pct,
        point_value=config.point_value,
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
        "n_trades": trade_count,
    }

    return BacktestResult(
        equity=equity,
        cash=cash,
        position=position,
        trade_log=trade_log,
        extra=extra,
    )
