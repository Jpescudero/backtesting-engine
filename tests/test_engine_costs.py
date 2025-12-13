from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("numba")

from src.costs import CostModel
from src.data.feeds import OHLCVArrays
from src.engine.core import BacktestConfig, run_backtest_with_signals


@pytest.fixture()
def simple_data() -> OHLCVArrays:
    ts = np.arange(3, dtype=np.int64)
    prices = np.array([100.0, 101.0, 102.0])
    return OHLCVArrays(ts=ts, o=prices, h=prices, low=prices, c=prices, v=prices)


def test_cost_model_applied_to_trade_log(simple_data: OHLCVArrays) -> None:
    model = CostModel.from_yaml("config/costs/costs.yaml", "NDX")
    config = BacktestConfig(
        initial_cash=100_000.0,
        trade_size=1.0,
        sl_pct=0.0,
        tp_pct=0.0,
        max_bars_in_trade=10,
        entry_threshold=0.0,
        cost_config_path=str(Path("config/costs/costs.yaml")),
        cost_instrument="NDX",
    )

    signals = np.array([1, 0, -1], dtype=np.int8)
    result = run_backtest_with_signals(simple_data, signals, config=config, cost_model=model)

    assert "pnl_net" in result.trade_log
    np.testing.assert_allclose(result.trade_log["pnl_gross"], np.array([2.0]))
    np.testing.assert_allclose(result.trade_log["cost"], np.array([4.0]))
    np.testing.assert_allclose(result.trade_log["pnl_net"], np.array([-2.0]))
    assert result.cash_net is not None
    assert result.equity_net is not None
    assert result.cash_net == pytest.approx(result.cash - 4.0)
    assert result.equity_net[-1] == pytest.approx(result.equity[-1] - 4.0)
