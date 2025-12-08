"""Colección de métricas incrementales para backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol

import pandas as pd


@dataclass
class EventPayload:
    """Evento genérico producido por el motor de backtest."""

    timestamp: pd.Timestamp
    type: str
    data: Dict[str, Any] | None = None


@dataclass
class BarSnapshot:
    """Instantánea de barra utilizada por los collectors."""

    timestamp: pd.Timestamp
    close: float
    equity: float
    position: float
    cash: float | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeSnapshot:
    """Información de un trade cerrado."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    exit_reason: str | None = None
    holding_bars: int | None = None
    meta: Dict[str, Any] | None = None


class MetricCollector(Protocol):
    """Interfaz mínima para recolectores de métricas."""

    name: str

    def on_event(self, event: EventPayload) -> None: ...

    def on_bar(self, bar: BarSnapshot) -> None: ...

    def on_trade(self, trade: TradeSnapshot) -> None: ...

    def to_frame(self) -> pd.DataFrame:
        """Devuelve el contenido del collector en formato tabular."""
        ...


class BaseCollector(MetricCollector):
    """Implementación base que acumula diccionarios de registros."""

    name: str = "base"

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def on_event(self, event: EventPayload) -> None:  # pragma: no cover - hook opcional
        return None

    def on_bar(self, bar: BarSnapshot) -> None:  # pragma: no cover - hook opcional
        return None

    def on_trade(self, trade: TradeSnapshot) -> None:  # pragma: no cover - hook opcional
        return None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


class PNLCollector(BaseCollector):
    """Acumula la serie de equity/pnl barra a barra."""

    name = "pnl"

    def on_bar(self, bar: BarSnapshot) -> None:
        self._records.append(
            {
                "timestamp": bar.timestamp,
                "close": float(bar.close),
                "equity": float(bar.equity),
                "position": float(bar.position),
                "cash": None if bar.cash is None else float(bar.cash),
                **(bar.extra or {}),
            }
        )


class DrawdownCollector(BaseCollector):
    """Calcula drawdown incremental sobre la equity."""

    name = "drawdown"

    def __init__(self) -> None:
        super().__init__()
        self._peak: float | None = None

    def on_bar(self, bar: BarSnapshot) -> None:
        equity = float(bar.equity)
        if self._peak is None:
            self._peak = equity
        else:
            self._peak = max(self._peak, equity)

        drawdown = 0.0 if not self._peak or self._peak == 0 else (equity / self._peak) - 1.0

        self._records.append(
            {
                "timestamp": bar.timestamp,
                "equity": equity,
                "peak_equity": float(self._peak),
                "drawdown": drawdown,
            }
        )


class ExposureCollector(BaseCollector):
    """Registra exposición nocional y tamaño de posición por barra."""

    name = "exposure"

    def on_bar(self, bar: BarSnapshot) -> None:
        notional = float(bar.position) * float(bar.close)
        entry = {
            "timestamp": bar.timestamp,
            "position": float(bar.position),
            "close": float(bar.close),
            "notional": notional,
        }
        if bar.cash is not None and bar.cash != 0:
            entry["leverage"] = notional / float(bar.cash)
        self._records.append(entry)


class TradesCollector(BaseCollector):
    """Almacena el log de trades cerrados."""

    name = "trades"

    def on_trade(self, trade: TradeSnapshot) -> None:
        record: Dict[str, Any] = {
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "entry_price": float(trade.entry_price),
            "exit_price": float(trade.exit_price),
            "qty": float(trade.qty),
            "pnl": float(trade.pnl),
            "exit_reason": trade.exit_reason,
            "holding_bars": trade.holding_bars,
        }
        if trade.meta:
            record.update(trade.meta)
        self._records.append(record)


def fill_collectors_from_bars(
    collectors: Iterable[MetricCollector], bars: Iterable[BarSnapshot]
) -> None:
    """Propaga un iterable de barras por todos los collectors registrados."""

    for bar in bars:
        for collector in collectors:
            collector.on_bar(bar)


def fill_collectors_from_trades(
    collectors: Iterable[MetricCollector], trades: Iterable[TradeSnapshot]
) -> None:
    """Propaga trades cerrados a todos los collectors registrados."""

    for trade in trades:
        for collector in collectors:
            collector.on_trade(trade)


CollectorMap = Mapping[str, MetricCollector]
