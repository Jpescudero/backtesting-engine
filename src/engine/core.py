# src/engine/core.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from numba import njit

from src.costs import CostModel
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
    trade_size: float = 1.0  # contratos/unidades por operación
    min_trade_size: float = 0.01  # tamaño mínimo permitido por contrato/lote
    max_trade_size: float = 1000.0  # límite superior para el tamaño por operación
    cost_config_path: str = "config/costs/costs.yaml"
    cost_instrument: Optional[str] = None

    # Gestión de riesgo
    sl_pct: float = 0.01  # stop loss a -1%
    tp_pct: float = 0.02  # take profit a +2%
    risk_per_trade_pct: float = 0.0  # riesgo fijo por trade (0.0025 => 0.25%)
    atr_stop_mult: float = 0.0  # múltiplo de ATR para calcular el SL
    atr_tp_mult: float = 0.0  # múltiplo de ATR para calcular el TP
    point_value: float = 1.0  # valor monetario de 1 punto para 1.0 contrato
    max_bars_in_trade: int = 60  # duración máxima del trade en barras

    # Parámetros de la estrategia de ejemplo
    entry_threshold: float = 0.001  # 0.1% de subida respecto al cierre anterior para entrar


@dataclass
class BacktestResult:
    """
    Resultado del backtest.
    """

    equity: np.ndarray  # serie de equity
    cash: float  # efectivo final
    position: float  # posición final
    trade_log: Dict[str, np.ndarray]  # arrays con info de los trades
    extra: Dict[str, Any]  # parámetros y metadatos
    equity_net: np.ndarray | None = None
    cash_net: float | None = None
    snapshots: List["BacktestSnapshot"] | None = None
    state_log: "BacktestStateLog" | None = None


@dataclass
class BacktestStateLog:
    cash: np.ndarray
    position: np.ndarray
    entry_price: np.ndarray
    stop_price: np.ndarray
    take_profit: np.ndarray
    entry_bar_idx: np.ndarray
    use_atr_stop: np.ndarray


@dataclass
class BacktestSnapshot:
    index: int
    ts: int
    cash: float
    position: float
    entry_price: float
    entry_bar_idx: int
    stop_price: float
    take_profit: float
    use_atr_stop: bool

    def to_dict(self) -> Dict[str, Any]:
        open_orders: list[Dict[str, Any]] = []
        if self.position > 0.0:
            open_orders.append(
                {
                    "type": "long_position",
                    "qty": self.position,
                    "entry_price": self.entry_price,
                    "entry_bar_idx": self.entry_bar_idx,
                    "stop_price": self.stop_price,
                    "take_profit": self.take_profit,
                    "use_atr_stop": self.use_atr_stop,
                }
            )

        return {
            "index": self.index,
            "ts": int(self.ts),
            "cash": float(self.cash),
            "position": float(self.position),
            "entry_price": float(self.entry_price),
            "entry_bar_idx": int(self.entry_bar_idx),
            "stop_price": float(self.stop_price),
            "take_profit": float(self.take_profit),
            "use_atr_stop": bool(self.use_atr_stop),
            "open_orders": open_orders,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacktestSnapshot":
        return cls(
            index=int(data.get("index", 0)),
            ts=int(data.get("ts", 0)),
            cash=float(data.get("cash", 0.0)),
            position=float(data.get("position", 0.0)),
            entry_price=float(data.get("entry_price", 0.0)),
            entry_bar_idx=int(data.get("entry_bar_idx", -1)),
            stop_price=float(data.get("stop_price", 0.0)),
            take_profit=float(data.get("take_profit", 0.0)),
            use_atr_stop=bool(data.get("use_atr_stop", False)),
        )

    def initial_state(self) -> Tuple[float, float, float, int, float, float, bool]:
        return (
            self.cash,
            self.position,
            self.entry_price,
            self.entry_bar_idx,
            self.stop_price,
            self.take_profit,
            self.use_atr_stop,
        )


def build_snapshots(
    ts: np.ndarray, state_log: BacktestStateLog, snapshot_interval: int | None
) -> List[BacktestSnapshot]:
    if snapshot_interval is None or snapshot_interval <= 0:
        return []

    snapshots: List[BacktestSnapshot] = []
    n = state_log.cash.shape[0]
    indices = list(range(0, n, snapshot_interval))
    if (n - 1) not in indices:
        indices.append(n - 1)

    for idx in indices:
        if idx >= n:
            continue
        cash_val = state_log.cash[idx]
        if not np.isfinite(cash_val):
            continue
        entry_price = state_log.entry_price[idx]
        stop_price = state_log.stop_price[idx]
        take_profit = state_log.take_profit[idx]
        if not np.isfinite(entry_price):
            entry_price = 0.0
        if not np.isfinite(stop_price):
            stop_price = 0.0
        if not np.isfinite(take_profit):
            take_profit = 0.0

        snapshots.append(
            BacktestSnapshot(
                index=idx,
                ts=int(ts[idx]) if idx < ts.shape[0] else 0,
                cash=float(cash_val),
                position=float(state_log.position[idx]),
                entry_price=float(entry_price),
                entry_bar_idx=int(state_log.entry_bar_idx[idx]),
                stop_price=float(stop_price),
                take_profit=float(take_profit),
                use_atr_stop=bool(state_log.use_atr_stop[idx]),
            )
        )

    return snapshots


# =====================
# Estrategia de ejemplo (Numba)
# =====================


@njit
def _example_strategy_long_on_up_move(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
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
    desired_qty: float,
    min_trade_size: float,
) -> float:
    """
    Devuelve la cantidad máxima que se puede abrir sin pasar a efectivo negativo.

    La cantidad resultante se ajusta a múltiplos de min_trade_size para permitir
    operar con fracciones de contrato (p.ej. lotes de 0.01).
    """
    effective_cash = available_cash
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


@njit
def compute_position_size(
    equity: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    risk_pct: float = 0.01,
    point_value: float = 1.0,
    rr_ref: float = 1.5,
    min_scale: float = 0.5,
    max_scale: float = 1.5,
) -> int:
    """
    Calcula el tamaño dinámico de la posición (número de contratos)
    respetando un riesgo máximo al SL y ajustando por la relación TP/SL.

    Devuelve un entero >= 0.
    """

    if (
        not np.isfinite(equity)
        or not np.isfinite(entry_price)
        or not np.isfinite(stop_loss)
        or not np.isfinite(take_profit)
        or risk_pct <= 0.0
        or point_value <= 0.0
    ):
        return 0

    # Distancias en puntos
    sl_dist = np.abs(entry_price - stop_loss)
    tp_dist = np.abs(take_profit - entry_price)

    # SL inválido => no operamos
    if sl_dist <= 0.0:
        return 0

    # Riesgo máximo monetario permitido
    risk_cash = equity * risk_pct

    # Tamaño base limitado por el SL
    units_base = risk_cash / (sl_dist * point_value)
    if units_base <= 0.0:
        return 0

    # Ratio beneficio/riesgo teórico
    rr = tp_dist / sl_dist

    # Factor de escala según la distancia al TP
    scale = rr / rr_ref
    scale = max(min_scale, min(max_scale, scale))

    # Tamaño final entero y no negativo
    units = int(np.floor(units_base * scale))
    return max(units, 0)


# =====================
# Motor de backtest con SL/TP & duración (Numba)
# =====================


@njit
def _backtest_with_risk(
    ts: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    position_sizes: np.ndarray,
    initial_cash: float,
    trade_size: float,
    min_trade_size: float,
    entry_threshold: float,
    sl_pct: float,
    tp_pct: float,
    max_bars_in_trade: int,
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
      - SL / TP en % desde precio de entrada.
      - Máxima duración del trade.
      - Tamaño dinámico opcional por barra (position_sizes).
      - Registro de trades.

    exit_reason:
      1 -> Stop Loss
      2 -> Take Profit
      3 -> Time Stop (duración máxima)
      4 -> Señal contraria
    """
    n = c.shape[0]
    equity = np.full(n, np.nan, dtype=np.float64)

    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    entry_bar_idx = -1
    stop_price = 0.0
    tp_price = 0.0

    # Señales de la estrategia
    signals = _example_strategy_long_on_up_move(o, h, low, c, v, entry_threshold)

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
            price_low = low[i]
            price_high = h[i]
            if not np.isfinite(price_low):
                price_low = price
            if not np.isfinite(price_high):
                price_high = price
            bars_in_trade = i - entry_bar_idx

            reason = 0
            exit_level = price

            sl_hit = np.isfinite(price_low) and price_low <= stop_price
            tp_hit = np.isfinite(price_high) and price_high >= tp_price

            if sl_hit:
                reason = 1  # SL
                exit_level = stop_price
            elif tp_hit:
                reason = 2  # TP
                exit_level = tp_price
            elif bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop
                exit_level = price

            if reason != 0:
                # Cerrar posición
                trade_price = exit_level
                cash += trade_price * position

                realized_pnl = (trade_price - entry_price) * position
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

        # 2) Procesar señal de la estrategia (compra/venta)
        sig = signals[i]

        # Señal de venta: cerrar por señal contraria
        if sig == -1 and position > 0.0:
            trade_price = price
            cash += trade_price * position

            realized_pnl = (trade_price - entry_price) * position
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
            trade_price = price
            qty = _calculate_affordable_qty(
                available_cash=cash,
                price_per_unit=trade_price,
                desired_qty=trade_size,
                min_trade_size=min_trade_size,
            )
            if qty > 0.0:
                cash -= trade_price * qty
                position = qty
                entry_price = trade_price
                entry_bar_idx = i
                stop_price = trade_price * (1.0 - sl_pct)
                tp_price = trade_price * (1.0 + tp_pct)

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
    cost_model: CostModel | None = None,
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

    # Estrategia interna: no se calculan tamaños dinámicos, por lo que se
    # pasa un array vacío (NaN) para desactivar ese camino.
    position_sizes = np.full_like(data.c, np.nan, dtype=float)

    # No se usan stops personalizados en este modo básico, pero guardamos arrays
    # de NaN para mantener la interfaz uniforme con los pipelines que sí
    # proporcionan SL/TP por barra.
    stop_losses = np.full_like(data.c, np.nan, dtype=float)
    take_profits = np.full_like(data.c, np.nan, dtype=float)

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
        low=data.low,
        c=data.c,
        v=data.v,
        position_sizes=position_sizes,
        initial_cash=config.initial_cash,
        trade_size=config.trade_size,
        min_trade_size=config.min_trade_size,
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

    equity_net, cash_net, trade_log, cost_summary = _apply_costs(cost_model, trade_log, equity, cash)

    extra: Dict[str, Any] = {
        "initial_cash": config.initial_cash,
        "trade_size": config.trade_size,
        "min_trade_size": config.min_trade_size,
        "max_trade_size": config.max_trade_size,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "atr_stop_mult": config.atr_stop_mult,
        "atr_tp_mult": config.atr_tp_mult,
        "point_value": config.point_value,
        "max_bars_in_trade": config.max_bars_in_trade,
        "entry_threshold": config.entry_threshold,
        "n_trades": trade_count,
        # Guardamos SL/TP iniciales para poder plotearlos después
        "stop_losses": stop_losses,
        "take_profits": take_profits,
    }
    if cost_model is not None:
        extra["cost_model"] = asdict(cost_model.config)
    if cost_summary:
        extra["costs"] = cost_summary

    return BacktestResult(
        equity=equity,
        cash=cash,
        position=position,
        trade_log=trade_log,
        extra=extra,
        equity_net=equity_net,
        cash_net=cash_net,
    )


def _apply_costs(
    cost_model: CostModel | None,
    trade_log: Dict[str, np.ndarray],
    equity: np.ndarray,
    cash: float,
) -> tuple[np.ndarray | None, float | None, Dict[str, np.ndarray], Dict[str, float]]:
    """Aplica el modelo de costes al log de trades y devuelve métricas netas."""

    if cost_model is None or not trade_log:
        return None, None, trade_log, {}

    pnl_gross = np.asarray(trade_log.get("pnl", []), dtype=float)
    entries = np.asarray(trade_log.get("entry_price", []), dtype=float)
    exits = np.asarray(trade_log.get("exit_price", []), dtype=float)
    qty = np.asarray(trade_log.get("qty", []), dtype=float)
    exit_idx = np.asarray(trade_log.get("exit_idx", []), dtype=np.int64)

    n_trades = pnl_gross.shape[0]
    if n_trades == 0:
        return None, None, trade_log, {}

    pnl_net = np.empty_like(pnl_gross)
    pnl_gross_out = pnl_gross.copy()
    return_gross = np.zeros_like(pnl_gross)
    return_net = np.zeros_like(pnl_gross)
    cost_values = np.zeros_like(pnl_gross)
    commission = np.zeros_like(pnl_gross)
    spread_cost = np.zeros_like(pnl_gross)
    slippage_cost = np.zeros_like(pnl_gross)
    cost_returns = np.zeros_like(pnl_gross)

    for i in range(n_trades):
        side = "long" if qty[i] >= 0 else "short"
        trade_qty = abs(qty[i])
        breakdown = cost_model.breakdown(float(entries[i]), float(exits[i]), side, qty=trade_qty)
        commission[i] = breakdown["commission"]
        spread_cost[i] = breakdown["spread"]
        slippage_cost[i] = breakdown["slippage"]
        cost_values[i] = breakdown["total_cost"]
        cost_returns[i] = breakdown.get("cost_return", 0.0)

        pnl_net[i] = pnl_gross[i] - cost_values[i]
        notional = abs(entries[i]) * cost_model.config.contract_multiplier * trade_qty
        if notional > 0:
            return_gross[i] = pnl_gross[i] / notional
            return_net[i] = pnl_net[i] / notional

    trade_log_enhanced = {
        **trade_log,
        "pnl_gross": pnl_gross_out,
        "pnl_net": pnl_net,
        "pnl": pnl_net,
        "return_gross": return_gross,
        "return_net": return_net,
        "cost": cost_values,
        "commission": commission,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
        "cost_return": cost_returns,
    }

    equity_net = equity.copy()
    for idx, cost_val in zip(exit_idx, cost_values):
        if 0 <= idx < equity_net.shape[0]:
            equity_net[idx:] -= cost_val

    cost_summary = {
        "total_cost": float(cost_values.sum()),
        "commission": float(commission.sum()),
        "spread": float(spread_cost.sum()),
        "slippage": float(slippage_cost.sum()),
    }

    return equity_net, float(cash - cost_values.sum()), trade_log_enhanced, cost_summary


# =====================
# Motor de backtest usando SEÑALES externas (Numba)
# =====================


@njit
def _backtest_with_risk_from_signals(
    ts: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    v: np.ndarray,
    signals: np.ndarray,
    stop_losses: np.ndarray,
    take_profits: np.ndarray,
    position_sizes: np.ndarray,
    atr: np.ndarray,
    initial_cash: float,
    trade_size: float,
    min_trade_size: float,
    max_trade_size: float,
    sl_pct: float,
    tp_pct: float,
    risk_per_trade_pct: float,
    atr_stop_mult: float,
    atr_tp_mult: float,
    point_value: float,
    max_bars_in_trade: int,
    start_index: int = 0,
    initial_cash_state: float = 0.0,
    initial_position_state: float = 0.0,
    initial_entry_price: float = 0.0,
    initial_entry_bar_idx: int = -1,
    initial_stop_price: float = 0.0,
    initial_tp_price: float = 0.0,
    initial_use_atr_stop: bool = False,
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
    np.ndarray,  # cash_path
    np.ndarray,  # position_path
    np.ndarray,  # stop_price_path
    np.ndarray,  # take_profit_path
    np.ndarray,  # entry_bar_idx_path
    np.ndarray,  # use_atr_stop_path
    np.ndarray,  # entry_price_path
]:
    """
    Igual que _backtest_with_risk, pero usando un array de señales externo.

    Parámetros clave:
      - signals: array int8 del mismo tamaño que c:
          +1 -> señal de compra/entrada larga
          -1 -> cierre por señal contraria
           0 -> nada
      - position_sizes: tamaño deseado por barra (NaN/<=0 activa los parámetros clásicos)

    exit_reason:
      1 -> Stop Loss
      2 -> Take Profit
      3 -> Time Stop (duración máxima)
      4 -> Señal contraria
    """
    n = c.shape[0]
    equity = np.empty(n, dtype=np.float64)

    # Si reanudamos desde un snapshot, dejamos los valores previos como NaN para
    # evitar que queden restos de memoria (cero u otros valores) en la curva de
    # equity. Así la serie resultante refleja únicamente la parte efectiva del
    # backtest.
    if start_index > 0:
        equity[:start_index] = np.nan

    cash = initial_cash if start_index == 0 else initial_cash_state
    position = 0.0 if start_index == 0 else initial_position_state
    entry_price = 0.0 if start_index == 0 else initial_entry_price
    entry_bar_idx = -1 if start_index == 0 else initial_entry_bar_idx
    stop_price = 0.0 if start_index == 0 else initial_stop_price
    tp_price = 0.0 if start_index == 0 else initial_tp_price
    use_atr_stops = False if start_index == 0 else initial_use_atr_stop

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

    state_cash = np.full(n, np.nan, dtype=np.float64)
    state_position = np.full(n, np.nan, dtype=np.float64)
    state_stop = np.full(n, np.nan, dtype=np.float64)
    state_tp = np.full(n, np.nan, dtype=np.float64)
    state_entry_idx = np.full(n, -1, dtype=np.int64)
    state_entry_price = np.full(n, np.nan, dtype=np.float64)
    state_use_atr = np.zeros(n, dtype=np.int8)

    for i in range(start_index, n):
        price = c[i]

        # --- Defensa: si el precio no es finito, saltamos la barra ---
        if not np.isfinite(price):
            if i == 0:
                equity[i] = initial_cash
            else:
                equity[i] = equity[i - 1]
            state_cash[i] = cash
            state_position[i] = position
            state_stop[i] = stop_price
            state_tp[i] = tp_price
            state_entry_idx[i] = entry_bar_idx
            state_entry_price[i] = entry_price
            state_use_atr[i] = 1 if use_atr_stops else 0
            continue
        # -------------------------------------------------------------

        # 1) Comprobar SL / TP / max_bars antes de nuevas señales
        if position > 0.0:
            bars_in_trade = i - entry_bar_idx
            price_low = low[i]
            price_high = h[i]
            if not np.isfinite(price_low):
                price_low = price
            if not np.isfinite(price_high):
                price_high = price

            reason = 0
            exit_level = price

            if use_atr_stops:
                if stop_price > 0.0 and price_low <= stop_price:
                    reason = 1  # SL
                    exit_level = stop_price
                elif tp_price > 0.0 and price_high >= tp_price:
                    reason = 2  # TP
                    exit_level = tp_price
            else:
                if stop_price > 0.0 and price_low <= stop_price:
                    reason = 1  # SL
                    exit_level = stop_price
                elif tp_price > 0.0 and price_high >= tp_price:
                    reason = 2  # TP
                    exit_level = tp_price

            if reason == 0 and bars_in_trade >= max_bars_in_trade:
                reason = 3  # Time stop
                exit_level = price

            if reason != 0:
                # Cerrar posición
                trade_price = exit_level
                cash += trade_price * position

                realized_pnl = (trade_price - entry_price) * position
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
            trade_price = price
            cash += trade_price * position

            realized_pnl = (trade_price - entry_price) * position
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
            stop_price = 0.0
            tp_price = 0.0

        # Señal de compra: abrir posición si estamos flat
        if sig == 1 and position == 0.0:
            trade_price = price
            atr_val = atr[i] if i < atr.shape[0] else np.nan
            has_atr = np.isfinite(atr_val)

            sl_price = stop_losses[i] if i < stop_losses.shape[0] else np.nan
            tp_price_candidate = take_profits[i] if i < take_profits.shape[0] else np.nan
            has_custom_stops = np.isfinite(sl_price) and np.isfinite(tp_price_candidate)

            desired_qty = position_sizes[i] if np.isfinite(position_sizes[i]) else trade_size

            if has_custom_stops:
                current_equity = cash + position * price
                risk_param = risk_per_trade_pct if risk_per_trade_pct > 0.0 else 0.01
                risk_qty = compute_position_size(
                    equity=current_equity,
                    entry_price=trade_price,
                    stop_loss=sl_price,
                    take_profit=tp_price_candidate,
                    risk_pct=risk_param,
                    point_value=point_value,
                )
                if risk_qty > 0:
                    desired_qty = float(risk_qty)
            elif (
                risk_per_trade_pct > 0.0 and has_atr and atr_stop_mult > 0.0 and desired_qty <= 0.0
            ):
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
                if risk_qty > 0.0 and desired_qty <= 0.0:
                    desired_qty = risk_qty

            desired_qty = _clip_size(desired_qty, min_trade_size, max_trade_size)

            qty = _calculate_affordable_qty(
                available_cash=cash,
                price_per_unit=trade_price,
                desired_qty=desired_qty,
                min_trade_size=min_trade_size,
            )
            if qty > 0.0:
                position = qty
                cash -= trade_price * position
                entry_price = trade_price
                entry_bar_idx = i

                if has_custom_stops:
                    stop_price = sl_price
                    tp_price = tp_price_candidate
                    use_atr_stops = True
                elif has_atr and atr_stop_mult > 0.0:
                    stop_distance = atr_stop_mult * atr_val
                    stop_price = trade_price - stop_distance
                    tp_price = 0.0
                    if atr_tp_mult > 0.0:
                        tp_price = trade_price + atr_tp_mult * atr_val
                    use_atr_stops = stop_distance > 0.0
                else:
                    stop_price = trade_price * (1.0 - sl_pct) if sl_pct > 0.0 else 0.0
                    tp_price = trade_price * (1.0 + tp_pct) if tp_pct > 0.0 else 0.0
                    use_atr_stops = False

        # 3) Mark-to-market de la equity
        equity[i] = cash + position * price

        state_cash[i] = cash
        state_position[i] = position
        state_stop[i] = stop_price
        state_tp[i] = tp_price
        state_entry_idx[i] = entry_bar_idx
        state_entry_price[i] = entry_price
        state_use_atr[i] = 1 if use_atr_stops else 0

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
        state_cash,
        state_position,
        state_stop,
        state_tp,
        state_entry_idx,
        state_use_atr,
        state_entry_price,
    )


# =====================
# Interfaz de alto nivel usando señales externas
# =====================


def run_backtest_with_signals(
    data: OHLCVArrays,
    signals: np.ndarray,
    position_sizes: np.ndarray | None = None,
    atr: np.ndarray | None = None,
    stop_losses: np.ndarray | None = None,
    take_profits: np.ndarray | None = None,
    config: BacktestConfig | None = None,
    snapshot_interval: int | None = None,
    resume_from: BacktestSnapshot | None = None,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """
    Ejecuta un backtest usando un array de señales externo (int8: -1, 0, +1).

    Es igual que run_backtest_basic, pero:
      - No calcula la estrategia dentro del motor.
      - Usa las señales proporcionadas.
      - Permite tamaños dinámicos por barra (position_sizes) calculados por la estrategia.
    """
    if config is None:
        config = BacktestConfig()

    bars = data.c.shape[0]

    if signals.shape[0] != bars:
        raise ValueError(
            "El tamaño de signals "
            f"({signals.shape[0]}) no coincide con el número de barras ({bars})."
        )

    if position_sizes is None:
        position_sizes = np.full_like(data.c, np.nan, dtype=float)
    elif position_sizes.shape[0] != bars:
        raise ValueError(
            "El tamaño de position_sizes "
            f"({position_sizes.shape[0]}) no coincide con el número de barras ({bars})."
        )
    position_sizes = np.asarray(position_sizes, dtype=np.float64)

    if atr is None:
        atr = np.full_like(data.c, np.nan, dtype=float)
    elif atr.shape[0] != bars:
        raise ValueError(
            f"El tamaño de atr ({atr.shape[0]}) no coincide con el número de barras ({bars})."
        )

    if stop_losses is None:
        stop_losses = np.full_like(data.c, np.nan, dtype=float)
    elif stop_losses.shape[0] != bars:
        raise ValueError(
            "El tamaño de stop_losses "
            f"({stop_losses.shape[0]}) no coincide con el número de barras ({bars})."
        )
    stop_losses = np.asarray(stop_losses, dtype=np.float64)

    if take_profits is None:
        take_profits = np.full_like(data.c, np.nan, dtype=float)
    elif take_profits.shape[0] != bars:
        raise ValueError(
            "El tamaño de take_profits "
            f"({take_profits.shape[0]}) no coincide con el número de barras ({bars})."
        )
    take_profits = np.asarray(take_profits, dtype=np.float64)

    start_index = 0
    if resume_from is not None:
        start_index = int(resume_from.index) + 1
        if start_index >= data.c.shape[0]:
            raise ValueError("El snapshot apunta al final del dataset; nada que reanudar")

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
        state_cash,
        state_position,
        state_stop,
        state_tp,
        state_entry_idx,
        state_use_atr,
        state_entry_price,
    ) = _backtest_with_risk_from_signals(
        ts=data.ts,
        o=data.o,
        h=data.h,
        low=data.low,
        c=data.c,
        v=data.v,
        signals=signals.astype(np.int8),
        stop_losses=stop_losses,
        take_profits=take_profits,
        position_sizes=position_sizes,
        atr=atr,
        initial_cash=config.initial_cash,
        trade_size=config.trade_size,
        min_trade_size=config.min_trade_size,
        max_trade_size=config.max_trade_size,
        sl_pct=config.sl_pct,
        tp_pct=config.tp_pct,
        risk_per_trade_pct=config.risk_per_trade_pct,
        atr_stop_mult=config.atr_stop_mult,
        atr_tp_mult=config.atr_tp_mult,
        point_value=config.point_value,
        max_bars_in_trade=config.max_bars_in_trade,
        start_index=start_index,
        initial_cash_state=resume_from.cash if resume_from else 0.0,
        initial_position_state=resume_from.position if resume_from else 0.0,
        initial_entry_price=resume_from.entry_price if resume_from else 0.0,
        initial_entry_bar_idx=resume_from.entry_bar_idx if resume_from else -1,
        initial_stop_price=resume_from.stop_price if resume_from else 0.0,
        initial_tp_price=resume_from.take_profit if resume_from else 0.0,
        initial_use_atr_stop=resume_from.use_atr_stop if resume_from else False,
    )

    state_log = BacktestStateLog(
        cash=state_cash,
        position=state_position,
        entry_price=state_entry_price,
        stop_price=state_stop,
        take_profit=state_tp,
        entry_bar_idx=state_entry_idx,
        use_atr_stop=state_use_atr,
    )
    snapshots = build_snapshots(data.ts, state_log, snapshot_interval)

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

    equity_net, cash_net, trade_log, cost_summary = _apply_costs(cost_model, trade_log, equity, cash)

    extra: Dict[str, Any] = {
        "initial_cash": config.initial_cash,
        "trade_size": config.trade_size,
        "min_trade_size": config.min_trade_size,
        "max_trade_size": config.max_trade_size,
        "sl_pct": config.sl_pct,
        "tp_pct": config.tp_pct,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "atr_stop_mult": config.atr_stop_mult,
        "atr_tp_mult": config.atr_tp_mult,
        "point_value": config.point_value,
        "max_bars_in_trade": config.max_bars_in_trade,
        "entry_threshold": config.entry_threshold,
        "n_trades": trade_count,
        # Guardamos SL/TP iniciales para poder plotearlos después
        "stop_losses": stop_losses,
        "take_profits": take_profits,
    }

    if snapshot_interval:
        extra["snapshot_interval"] = snapshot_interval
    if resume_from is not None:
        extra["resumed_from_snapshot"] = resume_from.index
    if cost_model is not None:
        extra["cost_model"] = asdict(cost_model.config)
    if cost_summary:
        extra["costs"] = cost_summary

    return BacktestResult(
        equity=equity,
        cash=cash,
        position=position,
        trade_log=trade_log,
        extra=extra,
        snapshots=snapshots,
        state_log=state_log,
        equity_net=equity_net,
        cash_net=cash_net,
    )
