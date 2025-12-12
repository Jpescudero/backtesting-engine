"""Transaction cost modeling utilities."""

from __future__ import annotations

from typing import Any


_DEF_EPS = 1e-12


def _resolve_param(params: dict[str, Any], primary: str, fallback: str, default: float) -> float:
    """Resolve a parameter by primary name with a backward-compatible fallback."""

    if primary in params:
        return float(params[primary])
    if fallback in params:
        return float(params[fallback])
    return default


def compute_trade_cost_breakdown(params: dict[str, Any], entry_price: float, exit_price: float) -> dict[str, float]:
    """Estimate return-normalized round-trip transaction costs.

    Spread and slippage are assumed to be quoted in price points and applied
    round-trip (entry plus exit). Commission is specified per contract per
    round-trip in cash terms. Costs are normalized by entry notional to return
    space so they are directly comparable to percentage PnL.

    Parameters
    ----------
    params : dict[str, Any]
        Parameter dictionary containing cost fields. Supports aliases to remain
        backward compatible.
    entry_price : float
        Price at which the trade is entered.
    exit_price : float
        Price at which the trade is exited.

    Returns
    -------
    dict[str, float]
        Dictionary with spread, slippage, commission, total return costs and
        the corresponding total cost in price points.
    """

    spread_points = _resolve_param(params, "SPREAD_POINTS", "spread_points", 0.0)
    slippage_points = _resolve_param(params, "SLIPPAGE_POINTS", "slippage_points", 0.0)
    commission_cash = _resolve_param(
        params, "COMMISSION_CASH_PER_CONTRACT", "COMMISSION_PER_CONTRACT", 0.0
    )
    multiplier = _resolve_param(params, "CONTRACT_MULTIPLIER", "contract_multiplier", 1.0)

    notional = max(entry_price * multiplier, _DEF_EPS)

    cost_spread_return = spread_points / max(entry_price, _DEF_EPS)
    cost_slippage_return = slippage_points / max(entry_price, _DEF_EPS)
    cost_commission_return = commission_cash / notional

    return {
        "cost_spread_return": cost_spread_return,
        "cost_slippage_return": cost_slippage_return,
        "cost_commission_return": cost_commission_return,
        "cost_total_return": cost_spread_return + cost_slippage_return + cost_commission_return,
        "cost_total_points": spread_points + slippage_points,
    }


def compute_trade_costs(params: dict[str, Any], entry_price: float, exit_price: float) -> float:
    """Compatibility wrapper returning only the total cost as a return."""

    breakdown = compute_trade_cost_breakdown(params, entry_price, exit_price)
    return breakdown["cost_total_return"]
