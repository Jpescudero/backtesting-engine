"""Tests for configuration precedence and CLI defaults in main.py."""

from __future__ import annotations

import sys
import types


def _fake_njit(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


sys.modules.setdefault("numba", types.SimpleNamespace(njit=_fake_njit))

from main import _get_setting, parse_args


def test_parse_args_defaults_allow_config_override() -> None:
    """Arguments that are configurable via run_settings should default to None."""

    args = parse_args([])

    assert args.symbol is None
    assert args.timeframe is None
    assert args.strategy is None
    assert args.ema_short is None
    assert args.max_horizon is None


def test_get_setting_prefers_config_when_cli_absent() -> None:
    """Config file values should be used when CLI values are not provided."""

    config = {"strategy": "microstructure_sweep", "wick_factor": "2.0"}

    strategy = _get_setting(None, config, "strategy", "opening_sweep_v4")
    wick_factor = _get_setting(None, config, "wick_factor", 1.5, float)

    assert strategy == "microstructure_sweep"
    assert wick_factor == 2.0


def test_get_setting_cli_overrides_config() -> None:
    """Explicit CLI values must take precedence over config file values."""

    config = {"strategy": "microstructure_sweep"}

    strategy = _get_setting("opening_sweep_v4", config, "strategy", "microstructure_reversal")

    assert strategy == "opening_sweep_v4"
