import numpy as np
from src.data.feeds import OHLCVArrays
from src.engine.core import BacktestConfig, BacktestSnapshot, run_backtest_with_signals


def test_equity_pre_resume_indices_are_nan():
    """
    Al reanudar desde un snapshot, la parte previa del array de equity debe
    rellenarse con NaN para no contaminar la curva exportada.
    """

    # Datos mínimos: 4 barras con precios crecientes
    ts = np.arange(4, dtype=np.int64)
    prices = np.array([10.0, 11.0, 12.0, 13.0])
    data = OHLCVArrays(ts=ts, o=prices, h=prices, low=prices, c=prices, v=prices)

    # Snapshot en la barra 1 (reanudaría en start_index=2)
    snapshot = BacktestSnapshot(
        index=1,
        ts=int(ts[1]),
        cash=100_000.0,
        position=0.0,
        entry_price=0.0,
        entry_bar_idx=-1,
        stop_price=0.0,
        take_profit=0.0,
        use_atr_stop=False,
    )

    # Sin señales: equity debería mantenerse plana tras el punto de reanudación
    signals = np.zeros_like(prices, dtype=np.int8)
    result = run_backtest_with_signals(
        data=data, signals=signals, config=BacktestConfig(), resume_from=snapshot
    )

    # Los valores previos al índice de reanudación se rellenan con NaN
    assert np.isnan(result.equity[0])
    assert np.isnan(result.equity[1])

    # A partir del start_index la equity se registra con normalidad
    np.testing.assert_allclose(result.equity[2:], [100_000.0, 100_000.0])
