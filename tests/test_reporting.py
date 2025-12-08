import numpy as np
import pandas as pd

from src.analytics.reporting import equity_to_series
from src.data.feeds import OHLCVArrays
from src.engine.core import BacktestResult


def test_equity_to_series_deduplicates_and_sorts():
    ts = np.array([2, 1, 1], dtype=np.int64)
    prices = np.zeros_like(ts, dtype=float)
    volumes = np.zeros_like(ts, dtype=float)

    data = OHLCVArrays(ts=ts, o=prices, h=prices, low=prices, c=prices, v=volumes)
    result = BacktestResult(
        equity=np.array([1000.0, 1010.0, 1020.0]),
        cash=0.0,
        position=0.0,
        trade_log={},
        extra={},
    )

    series = equity_to_series(result, data)

    # El índice debe ser creciente y sin duplicados
    assert series.index.is_monotonic_increasing
    assert series.index.is_unique

    # Conserva el último valor para cada timestamp duplicado
    expected_index = pd.to_datetime([1, 2], unit="ns", utc=True)
    expected_values = np.array([1020.0, 1000.0])

    pd.testing.assert_index_equal(series.index, expected_index)
    np.testing.assert_allclose(series.values, expected_values)


def test_equity_to_series_drops_zeros_and_non_finite():
    ts = np.array([1, 2, 3, 4], dtype=np.int64)
    prices = np.zeros_like(ts, dtype=float)
    volumes = np.zeros_like(ts, dtype=float)

    data = OHLCVArrays(ts=ts, o=prices, h=prices, low=prices, c=prices, v=volumes)
    result = BacktestResult(
        equity=np.array([0.0, np.nan, np.inf, 10_000.0]),
        cash=0.0,
        position=0.0,
        trade_log={},
        extra={},
    )

    series = equity_to_series(result, data)

    expected_index = pd.to_datetime([4], unit="ns", utc=True)
    expected_values = np.array([10_000.0])

    pd.testing.assert_index_equal(series.index, expected_index)
    np.testing.assert_allclose(series.values, expected_values)
