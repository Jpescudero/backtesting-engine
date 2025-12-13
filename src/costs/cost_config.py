"""Configuration dataclasses for cost modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CostType = Literal["bps", "points"]


@dataclass
class CostConfig:
    """Cost configuration for a specific instrument.

    Attributes
    ----------
    instrument : str
        Instrument symbol.
    cost_type : Literal["bps", "points"]
        Whether costs are expressed in basis points or price points.
    commission_per_side : float
        Commission charged per side (entry or exit) in currency terms if
        ``cost_type`` is ``"points"`` or in basis points of notional when
        ``cost_type`` is ``"bps"``.
    spread : float
        Spread cost expressed in price points or basis points depending on
        ``cost_type``. Applied once per round trip.
    slippage : float
        Slippage per side expressed in points or basis points.
    contract_multiplier : float
        Multiplier to convert price differences into currency PnL.
    currency : str
        Currency of the cost inputs.
    venue : Optional[str]
        Trading venue identifier.
    """

    instrument: str
    cost_type: CostType
    commission_per_side: float
    spread: float
    slippage: float
    contract_multiplier: float
    currency: str
    venue: Optional[str] = None
