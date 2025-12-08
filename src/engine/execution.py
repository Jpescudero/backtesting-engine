"""Execution simulator with configurable latency, liquidity, and cost assumptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class ExecutionParameters:
    """Configuración detallada para la simulación de ejecución."""

    latency_ms_mean: float = 20.0
    latency_ms_std: float = 5.0
    book_depth_levels: int = 5
    base_slippage: float = 0.0
    liquidity_slippage: float = 0.0
    fixed_cost: float = 0.0
    variable_cost_pct: float = 0.0
    cancellation_prob: float = 0.0
    seed: int | None = None


@dataclass
class ExecutionResult:
    filled_qty: float
    avg_price: float
    executed_value: float
    total_cost: float
    latency_ms: float
    unfilled_qty: float
    partial_fill: bool
    canceled_liquidity: float


def _build_default_book(mid_price: float, depth: int) -> List[Tuple[float, float]]:
    """Crea un libro de órdenes sintético (lado ask) con profundidad decreciente."""
    levels: List[Tuple[float, float]] = []
    for i in range(1, depth + 1):
        price = mid_price * (1.0 + 0.0005 * i)
        liquidity = max(1.0, float(depth - i + 1))
        levels.append((price, liquidity))
    return levels


class ExecutionSimulator:
    """Simulador sencillo de ejecución con latencia y libros de órdenes."""

    def __init__(self, params: ExecutionParameters):
        self.params = params
        self._rng = np.random.default_rng(params.seed)

    def _sample_latency(self) -> float:
        latency = self._rng.normal(self.params.latency_ms_mean, self.params.latency_ms_std)
        return float(max(0.0, latency))

    def _apply_cancellations(
        self, book: Iterable[Tuple[float, float]]
    ) -> Tuple[List[Tuple[float, float]], float]:
        remaining_levels: List[Tuple[float, float]] = []
        canceled = 0.0
        for price, qty in book:
            if self._rng.random() < self.params.cancellation_prob:
                canceled += qty
                continue
            remaining_levels.append((price, qty))
        return remaining_levels, canceled

    def _compute_slippage_price(self, side: str, base_price: float, filled: float) -> float:
        impact = self.params.base_slippage + self.params.liquidity_slippage * filled
        return base_price + impact if side.lower() == "buy" else base_price - impact

    def simulate_order(
        self,
        side: str,
        quantity: float,
        mid_price: float,
        book: Iterable[Tuple[float, float]] | None = None,
    ) -> ExecutionResult:
        """Simula la ejecución de una orden agresiva sobre el lado opuesto del libro."""
        if quantity <= 0.0:
            raise ValueError("quantity must be positive")

        book_levels = (
            list(book)
            if book is not None
            else _build_default_book(mid_price, self.params.book_depth_levels)
        )
        if not book_levels:
            raise ValueError("book must contain at least one level")

        latency_ms = self._sample_latency()
        book_after_cancel, canceled_liquidity = self._apply_cancellations(book_levels)

        remaining_qty = float(quantity)
        filled_qty = 0.0
        cost_value = 0.0

        # Para compras usamos el libro de asks (precios ascendentes)
        levels = sorted(book_after_cancel, key=lambda x: x[0], reverse=side.lower() == "sell")
        for price, lvl_qty in levels:
            if remaining_qty <= 0:
                break
            tradable = min(remaining_qty, lvl_qty)
            filled_qty += tradable
            cost_value += tradable * price
            remaining_qty -= tradable

        avg_price = cost_value / filled_qty if filled_qty > 0 else 0.0
        slipped_price = (
            self._compute_slippage_price(side, avg_price, filled_qty) if filled_qty > 0 else 0.0
        )
        executed_value = slipped_price * filled_qty

        total_cost = self.params.fixed_cost + executed_value * self.params.variable_cost_pct

        return ExecutionResult(
            filled_qty=filled_qty,
            avg_price=slipped_price,
            executed_value=executed_value,
            total_cost=total_cost,
            latency_ms=latency_ms,
            unfilled_qty=remaining_qty,
            partial_fill=remaining_qty > 0.0,
            canceled_liquidity=canceled_liquidity,
        )


PREDEFINED_SCENARIOS = {
    "mercado_volatil": ExecutionParameters(
        latency_ms_mean=30.0,
        latency_ms_std=12.0,
        book_depth_levels=3,
        base_slippage=0.0008,
        liquidity_slippage=0.0003,
        fixed_cost=1.5,
        variable_cost_pct=0.0004,
        cancellation_prob=0.35,
    ),
    "bajo_volumen": ExecutionParameters(
        latency_ms_mean=18.0,
        latency_ms_std=5.0,
        book_depth_levels=2,
        base_slippage=0.0004,
        liquidity_slippage=0.0006,
        fixed_cost=0.5,
        variable_cost_pct=0.0002,
        cancellation_prob=0.6,
    ),
}


def get_scenario(name: str, *, seed: int | None = None) -> ExecutionParameters:
    """Devuelve una configuración predefinida con una seed opcional."""
    if name not in PREDEFINED_SCENARIOS:
        raise KeyError(f"Escenario desconocido: {name}")
    params = PREDEFINED_SCENARIOS[name]
    return ExecutionParameters(**{**params.__dict__, "seed": seed})
