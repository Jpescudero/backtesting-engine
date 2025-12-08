from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
from src.backtesting.core.models import LatencyProfile, MarketDataBatch, OrderRequest
from src.backtesting.core.timeline import ensure_sequential_orders
from src.backtesting.data.npz_loader import NPZDataLoader
from src.backtesting.execution.simulated_broker import SimulatedBroker
from src.backtesting.strategy.signal_wrappers import SignalStrategyAdapter, simple_momentum_signal


@pytest.fixture()
def sample_market_data() -> MarketDataBatch:
    ts = np.array(
        [1_700_000_000_000_000_000, 1_700_000_060_000_000_000, 1_700_000_120_000_000_000],
        dtype=np.int64,
    )
    o = np.array([100.0, 101.0, 101.5])
    h = o + 0.5
    low = o - 0.5
    c = o + 0.2
    v = np.array([10, 15, 12])
    return MarketDataBatch(timestamps=ts, open=o, high=h, low=low, close=c, volume=v)


@pytest.fixture()
def sample_npz(tmp_path: Path) -> Path:
    base_dir = tmp_path / "TEST"
    base_dir.mkdir(parents=True)
    ts = np.array([1_700_000_060_000_000_000, 1_700_000_000_000_000_000], dtype=np.int64)
    np.savez(
        base_dir / "TEST_1m.npz",
        ts=ts,
        o=np.arange(2),
        h=np.arange(2),
        l=np.arange(2),
        c=np.arange(2),
        v=np.arange(2),
    )
    return base_dir


def test_npz_loader_contract(sample_npz: Path) -> None:
    loader = NPZDataLoader(symbol="TEST", timeframe="1m", base_dir=sample_npz)
    data = loader.load()

    assert isinstance(data, MarketDataBatch)
    assert data.size == 2
    assert np.all(np.diff(data.timestamps) >= 0)
    assert data.timestamps[0] < data.timestamps[1]


def test_simulated_broker_applies_latency(sample_market_data: MarketDataBatch) -> None:
    latency = LatencyProfile(latency=timedelta(seconds=1))
    broker = SimulatedBroker(latency=latency)

    order_time = datetime.utcfromtimestamp(sample_market_data.timestamps[0] / 1e9)
    order = OrderRequest(symbol="TEST", side="buy", quantity=1.0, timestamp=order_time, bar_index=0)
    fill = broker.process_orders([order], sample_market_data)[0]

    assert fill.submitted_at == order_time
    assert fill.executed_at - fill.submitted_at == timedelta(seconds=1)
    assert fill.avg_price == pytest.approx(sample_market_data.close[0])


def test_signal_adapter_respects_contract(sample_market_data: MarketDataBatch) -> None:
    strategy = SignalStrategyAdapter(
        symbol="TEST", signal_generator=simple_momentum_signal(threshold=0.001), qty=2.0
    )
    strategy.prepare(sample_market_data)

    batches = []
    for idx in range(sample_market_data.size):
        batches.append(strategy.on_bar(idx, sample_market_data))

    assert ensure_sequential_orders(batches)
    flattened: Iterable[OrderRequest] = (order for batch in batches for order in batch)
    for order in flattened:
        assert order.quantity == pytest.approx(2.0)
        assert order.bar_index < sample_market_data.size
        assert order.timestamp.tzinfo is None  # naive UTC timeline
