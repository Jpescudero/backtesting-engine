from __future__ import annotations

import math

from src.utils.risk import compute_position_size


def test_compute_position_size_basic():
    size = compute_position_size(
        equity=100_000,
        entry_price=105.0,
        stop_loss=100.0,
        take_profit=115.0,
        risk_pct=0.01,
        point_value=1.0,
    )
    assert size == 266


def test_compute_position_size_limits():
    zero_sl = compute_position_size(
        equity=50_000,
        entry_price=100.0,
        stop_loss=100.0,
        take_profit=110.0,
    )
    assert zero_sl == 0

    zero_units = compute_position_size(
        equity=10_000,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        risk_pct=0.0,
    )
    assert zero_units == 0


def test_compute_position_size_scaling_bounds():
    oversized = compute_position_size(
        equity=80_000,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=200.0,
        rr_ref=1.0,
        max_scale=1.0,
    )
    assert oversized == int(math.floor((80_000 * 0.0075 / 5.0) * 1.0))

    undersized = compute_position_size(
        equity=80_000,
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=96.0,
        rr_ref=2.0,
        min_scale=0.5,
        max_scale=1.0,
    )
    assert undersized == int(math.floor((80_000 * 0.0075 / 5.0) * 0.5))
