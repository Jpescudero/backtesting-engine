from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.backtesting.core.models import MarketDataBatch, OrderRequest
from src.backtesting.data.npz_loader import NPZDataLoader
from src.backtesting.execution.simulated_broker import SimulatedBroker


def _build_batch(close: list[float]) -> MarketDataBatch:
    base_ts = 1_700_000_000_000_000_000
    timestamps = np.array([base_ts + i * 60_000_000_000 for i in range(len(close))], dtype=np.int64)
    close_arr = np.array(close, dtype=float)
    return MarketDataBatch(
        timestamps=timestamps,
        open=close_arr,
        high=close_arr + 0.1,
        low=close_arr - 0.1,
        close=close_arr,
        volume=np.ones_like(close_arr),
    )


def test_npz_loader_rejects_nans(tmp_path: Path) -> None:
    base_dir = tmp_path / "TEST"
    base_dir.mkdir(parents=True)
    ts = np.array([1_700_000_000_000_000_000, 1_700_000_060_000_000_000], dtype=np.int64)
    np.savez(base_dir / "TEST_1m.npz", ts=ts, o=np.arange(2), h=np.arange(2), l=np.arange(2), c=np.array([1.0, np.nan]), v=np.arange(2))

    loader = NPZDataLoader(symbol="TEST", timeframe="1m", base_dir=base_dir)

    with pytest.raises(ValueError, match="NaN"):
        loader.load()


def test_npz_loader_rejects_unsorted_timestamps(tmp_path: Path) -> None:
    base_dir = tmp_path / "TEST"
    base_dir.mkdir(parents=True)
    ts = np.array([2, 1], dtype=np.int64)
    np.savez(base_dir / "TEST_1m.npz", ts=ts, o=np.arange(2), h=np.arange(2), l=np.arange(2), c=np.arange(2), v=np.arange(2))

    loader = NPZDataLoader(symbol="TEST", timeframe="1m", base_dir=base_dir)

    with pytest.raises(ValueError, match="ordenados"):
        loader.load()


def test_npz_loader_detects_critical_gaps(tmp_path: Path) -> None:
    base_dir = tmp_path / "TEST"
    base_dir.mkdir(parents=True)
    ts = np.array([0, 60_000_000_000, 1_200_000_000_000], dtype=np.int64)
    np.savez(base_dir / "TEST_1m.npz", ts=ts, o=np.arange(3), h=np.arange(3), l=np.arange(3), c=np.arange(3), v=np.arange(3))

    loader = NPZDataLoader(symbol="TEST", timeframe="1m", base_dir=base_dir)

    with pytest.raises(ValueError, match="Gap crítico"):
        loader.load()


def test_broker_rejects_orders_without_funds() -> None:
    data = _build_batch([10.0])
    broker = SimulatedBroker(initial_cash=5.0)
    order = OrderRequest(
        symbol="TEST",
        side="buy",
        quantity=1.0,
        timestamp=datetime.utcfromtimestamp(data.timestamps[0] / 1e9),
        bar_index=0,
    )

    with pytest.raises(ValueError, match="Fondos insuficientes"):
        broker.process_orders([order], data)

    assert broker.cash == 5.0


def test_broker_enforces_risk_limit() -> None:
    data = _build_batch([10.0])
    broker = SimulatedBroker(initial_cash=10_000.0, max_notional_per_order=20.0)
    order = OrderRequest(
        symbol="TEST",
        side="sell",
        quantity=3.0,
        timestamp=datetime.utcfromtimestamp(data.timestamps[0] / 1e9),
        bar_index=0,
    )

    with pytest.raises(ValueError, match="Límite de riesgo"):
        broker.process_orders([order], data)

    assert broker.cash == 10_000.0
