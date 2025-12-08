import numpy as np
import pytest

from src.data.feeds import OHLCVArrays
from src.engine.core import BacktestConfig, run_backtest_with_signals


def _build_minimal_feed(close_price: float = 100.0) -> OHLCVArrays:
    ts = np.array([0, 1], dtype=np.int64)
    prices = np.array([close_price, close_price], dtype=np.float64)
    return OHLCVArrays(ts=ts, o=prices, h=prices, low=prices, c=prices, v=prices)


def test_order_rejected_when_cash_insufficient_for_min_size():
    data = _build_minimal_feed(close_price=100.0)
    signals = np.array([0, 1], dtype=np.int8)
    config = BacktestConfig(initial_cash=10.0, commission_per_trade=1.0, trade_size=1.0, min_trade_size=1.0)

    with pytest.raises(ValueError, match="fondos insuficientes|tamaño calculado"):
        run_backtest_with_signals(data, signals, config=config)


def test_order_rejected_when_position_size_non_positive():
    data = _build_minimal_feed(close_price=50.0)
    signals = np.array([0, 1], dtype=np.int8)
    position_sizes = np.array([np.nan, -1.0], dtype=float)
    config = BacktestConfig(initial_cash=10_000.0, commission_per_trade=0.5, trade_size=1.0, min_trade_size=0.5)

    with pytest.raises(ValueError, match="tamaño calculado"):
        run_backtest_with_signals(data, signals, position_sizes=position_sizes, config=config)
