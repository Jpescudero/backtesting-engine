import numpy as np
import pytest

from src.engine.execution import (
    ExecutionParameters,
    ExecutionSimulator,
    ExecutionResult,
    get_scenario,
)


def test_partial_fill_with_limited_depth():
    params = ExecutionParameters(base_slippage=0.0, liquidity_slippage=0.0, fixed_cost=0.0, variable_cost_pct=0.0)
    simulator = ExecutionSimulator(params)
    book = [(101.0, 5.0)]  # profundidad mínima

    result = simulator.simulate_order("buy", quantity=10.0, mid_price=100.0, book=book)

    assert isinstance(result, ExecutionResult)
    assert result.filled_qty == 5.0
    assert result.unfilled_qty == 5.0
    assert result.partial_fill is True
    assert pytest.approx(result.avg_price) == 101.0


def test_cancellations_remove_liquidity():
    params = ExecutionParameters(cancellation_prob=1.0, seed=123)
    simulator = ExecutionSimulator(params)
    book = [(100.5, 2.0), (100.6, 2.0)]

    result = simulator.simulate_order("buy", quantity=2.0, mid_price=100.0, book=book)

    assert result.filled_qty == 0.0
    assert result.unfilled_qty == 2.0
    assert result.canceled_liquidity == pytest.approx(4.0)


def test_predefined_scenarios_are_reproducible_with_seed():
    params_seed_1 = get_scenario("mercado_volatil", seed=42)
    params_seed_2 = get_scenario("mercado_volatil", seed=42)

    sim1 = ExecutionSimulator(params_seed_1)
    sim2 = ExecutionSimulator(params_seed_2)

    book = [(100.2, 3.0), (100.3, 3.0), (100.4, 3.0)]

    res1 = sim1.simulate_order("sell", quantity=4.0, mid_price=100.0, book=book)
    res2 = sim2.simulate_order("sell", quantity=4.0, mid_price=100.0, book=book)

    assert res1.filled_qty == pytest.approx(res2.filled_qty)
    assert res1.avg_price == pytest.approx(res2.avg_price)
    assert res1.latency_ms == pytest.approx(res2.latency_ms)
