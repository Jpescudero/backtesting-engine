"""Cost utilities delegating to the centralized :mod:`src.costs` model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.costs import CostModel

_DEFAULT_COSTS_PATH = Path(__file__).resolve().parents[3] / "config/costs/costs.yaml"


def load_cost_model(params: dict[str, Any]) -> CostModel:
    """Load a :class:`CostModel` based on research parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Parameter dictionary containing at least ``SYMBOL`` and optionally
        ``COSTS_CONFIG_PATH`` to override the default location.

    Returns
    -------
    CostModel
        Cost model configured for the requested instrument.
    """

    symbol = str(params.get("SYMBOL"))
    if not symbol:
        raise ValueError("SYMBOL parameter is required to load costs")

    config_path = Path(params.get("COSTS_CONFIG_PATH", _DEFAULT_COSTS_PATH))
    return CostModel.from_yaml(str(config_path), symbol)


def compute_trade_cost_breakdown(
    cost_model: CostModel, entry_price: float, exit_price: float, side: str, qty: float = 1.0
) -> dict[str, float]:
    """Compatibility shim returning a CostModel breakdown."""

    return cost_model.breakdown(entry_price=entry_price, exit_price=exit_price, side=side, qty=qty)


def compute_trade_costs(cost_model: CostModel, entry_price: float, exit_price: float, side: str, qty: float = 1.0) -> float:
    """Compatibility wrapper returning only total return cost."""

    breakdown = cost_model.breakdown(entry_price=entry_price, exit_price=exit_price, side=side, qty=qty)
    return breakdown["cost_return"]
