"""Tests for lightweight serialization helpers used in backtest outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.analytics.backtest_output import _to_serializable


def _serialize(value: Any) -> Any:
    """Helper to keep assertions concise."""

    return _to_serializable(value)


def test_to_serializable_handles_python_complex_real():
    result = _serialize(3 + 0j)

    assert result == 3.0


def test_to_serializable_handles_python_complex_imaginary():
    result = _serialize(1 + 2j)

    assert result == {"real": 1.0, "imag": 2.0}


def test_to_serializable_handles_numpy_complex():
    value = np.complex128(5 + 0j)

    assert _serialize(value) == 5.0


def test_to_serializable_handles_numpy_complex_with_imag():
    value = np.complex128(0 - 3j)

    assert _serialize(value) == {"real": 0.0, "imag": -3.0}
