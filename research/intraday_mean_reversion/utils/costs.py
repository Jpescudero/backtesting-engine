"""Transaction cost modeling utilities."""

from __future__ import annotations

from typing import Any


def compute_trade_costs(params: dict[str, Any], entry_price: float, exit_price: float) -> float:
    """Estimate relative trade costs for a single round-trip trade.

    Parameters
    ----------
    params : dict[str, Any]
        Parameter dictionary containing cost fields.
    entry_price : float
        Price at which the trade is entered.
    exit_price : float
        Price at which the trade is exited.

    Returns
    -------
    float
        Total cost expressed as a return fraction.
    """

    spread_points = float(params.get("SPREAD_POINTS", 0.0))
    slippage_points = float(params.get("SLIPPAGE_POINTS", 0.0))
    commission = float(params.get("COMMISSION_PER_CONTRACT", 0.0))
    multiplier = float(params.get("CONTRACT_MULTIPLIER", 1.0))

    cost_spread = (spread_points + slippage_points) / max(entry_price, 1e-12)
    notional = entry_price * multiplier
    cost_commission = commission / max(notional, 1e-12)

    return cost_spread + cost_commission
