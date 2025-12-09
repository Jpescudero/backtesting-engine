"""Unit tests for OpeningSweepV4 parameter handling."""

from __future__ import annotations

from dataclasses import asdict

from src.strategies.opening_sweep_v4 import OpeningSweepV4, OpeningSweepV4Params


def test_opening_sweep_accepts_dataclass_config() -> None:
    """Passing a dataclass config should initialize without errors and keep values."""

    params = OpeningSweepV4Params(wick_factor=2.1)
    strategy = OpeningSweepV4(config=params)

    assert strategy.params.wick_factor == 2.1
    assert strategy.config == asdict(params)


def test_opening_sweep_merges_partial_dict_config() -> None:
    """Dict configs should merge with defaults preserving unspecified fields."""

    strategy = OpeningSweepV4(config={"wick_factor": 1.9})

    assert strategy.params.wick_factor == 1.9
    assert strategy.params.max_horizon == OpeningSweepV4Params().max_horizon
    assert strategy.config["wick_factor"] == 1.9
