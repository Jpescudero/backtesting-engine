from __future__ import annotations

import numpy as np

from src.backtesting.core.models import OrderFill


def cumulative_pnl(fills: list[OrderFill]) -> float:
    return float(
        sum(fill.notional if fill.order.side == "sell" else -fill.notional for fill in fills)
    )


def trade_count(fills: list[OrderFill]) -> int:
    return len(fills)


def average_fill_latency(fills: list[OrderFill]) -> float:
    if not fills:
        return 0.0
    deltas = [(fill.executed_at - fill.submitted_at).total_seconds() for fill in fills]
    return float(np.mean(deltas))
