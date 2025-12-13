"""Centralized cost model implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:  # pragma: no cover - import guard
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from src.costs.cost_config import CostConfig, CostType

TradeSide = Literal["long", "short"]


@dataclass
class CostModel:
    """Cost model computing transaction costs for a given instrument."""

    config: CostConfig

    @classmethod
    def from_yaml(cls, path: str, instrument: str) -> "CostModel":
        """Build a :class:`CostModel` from a YAML configuration file.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        instrument : str
            Instrument symbol to load from the ``instruments`` section.

        Returns
        -------
        CostModel
            Instantiated cost model for the requested instrument.

        Raises
        ------
        FileNotFoundError
            If the YAML configuration file does not exist.
        KeyError
            If the instrument is not present and no defaults are provided.
        ValueError
            If the configuration is malformed.
        """

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Cost configuration file not found: {config_path}")

        payload = _load_yaml_file(config_path)

        default_cfg = payload.get("default", {})
        instrument_cfg = payload.get("instruments", {}).get(instrument)
        if instrument_cfg is None:
            raise KeyError(f"Instrument '{instrument}' not found in cost configuration")

        merged = {**default_cfg, **instrument_cfg}
        required_fields = {
            "cost_type",
            "commission_per_side",
            "spread",
            "slippage",
            "contract_multiplier",
            "currency",
        }
        missing = required_fields - merged.keys()
        if missing:
            raise ValueError(f"Missing cost fields for {instrument}: {', '.join(sorted(missing))}")

        cost_type: CostType = merged["cost_type"]
        if cost_type not in {"bps", "points"}:
            raise ValueError(f"Invalid cost_type '{cost_type}' for {instrument}")

        venue = instrument_cfg.get("venue", merged.get("venue"))
        cfg = CostConfig(
            instrument=instrument,
            cost_type=cost_type,
            commission_per_side=float(merged["commission_per_side"]),
            spread=float(merged["spread"]),
            slippage=float(merged["slippage"]),
            contract_multiplier=float(merged["contract_multiplier"]),
            currency=str(merged["currency"]),
            venue=str(venue) if venue is not None else None,
        )
        return cls(cfg)

    def estimate_trade_cost(
        self,
        entry_price: float,
        exit_price: float,
        side: TradeSide,
        qty: float = 1.0,
    ) -> float:
        """Estimate total monetary cost for a round-trip trade.

        Parameters
        ----------
        entry_price : float
            Executed entry price.
        exit_price : float
            Executed exit price.
        side : {"long", "short"}
            Direction of the trade.
        qty : float, optional
            Quantity traded, by default 1.0.

        Returns
        -------
        float
            Total cost in currency units.
        """

        _validate_side(side)
        notional = max(entry_price, 0.0) * self.config.contract_multiplier * qty
        avg_price = (entry_price + exit_price) / 2.0 if entry_price and exit_price else entry_price or exit_price

        commission = self._commission_cost(notional, qty)
        spread_cost = self._spread_cost(avg_price, qty)
        slippage_cost = self._slippage_cost(avg_price, qty)

        return commission + spread_cost + slippage_cost

    def apply_to_gross_return(
        self,
        gross_return: float,
        entry_price: float,
        exit_price: float,
        side: TradeSide,
        qty: float = 1.0,
    ) -> float:
        """Apply costs to a gross return to obtain net return."""

        cost = self.estimate_trade_cost(entry_price, exit_price, side, qty=qty)
        notional = max(entry_price, 0.0) * self.config.contract_multiplier * qty
        if notional == 0:
            return gross_return
        return gross_return - cost / notional

    def breakdown(
        self,
        entry_price: float,
        exit_price: float,
        side: TradeSide,
        qty: float = 1.0,
    ) -> dict[str, float]:
        """Return a detailed breakdown of costs for a trade."""

        _validate_side(side)
        notional = max(entry_price, 0.0) * self.config.contract_multiplier * qty
        avg_price = (entry_price + exit_price) / 2.0 if entry_price and exit_price else entry_price or exit_price

        commission = self._commission_cost(notional, qty)
        spread_cost = self._spread_cost(avg_price, qty)
        slippage_cost = self._slippage_cost(avg_price, qty)
        total_cost = commission + spread_cost + slippage_cost

        return {
            "commission": commission,
            "spread": spread_cost,
            "slippage": slippage_cost,
            "total_cost": total_cost,
            "cost_return": total_cost / notional if notional else 0.0,
        }

    def _commission_cost(self, notional: float, qty: float) -> float:
        if self.config.cost_type == "points":
            return self.config.commission_per_side * 2.0 * qty
        return notional * (self.config.commission_per_side / 10_000.0) * 2.0

    def _spread_cost(self, ref_price: float, qty: float) -> float:
        if self.config.cost_type == "points":
            return self.config.spread * self.config.contract_multiplier * qty
        notional = ref_price * self.config.contract_multiplier * qty
        return notional * (self.config.spread / 10_000.0)

    def _slippage_cost(self, ref_price: float, qty: float) -> float:
        if self.config.cost_type == "points":
            return self.config.slippage * self.config.contract_multiplier * qty * 2.0
        notional = ref_price * self.config.contract_multiplier * qty
        return notional * (self.config.slippage / 10_000.0) * 2.0


def _validate_side(side: TradeSide) -> None:
    if side not in {"long", "short"}:
        raise ValueError(f"Invalid side '{side}'. Expected 'long' or 'short'.")


def _coerce_scalar(value: str) -> float | str:
    if value.startswith("\"") and value.endswith("\""):
        value = value[1:-1]
    try:
        return float(value)
    except ValueError:
        return value


def _parse_basic_yaml(content: str) -> dict:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, value_part = stripped.partition(":")
        value_part = value_part.strip()

        while indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if value_part == "":
            new_map: dict[str, object] = {}
            current[key] = new_map
            stack.append((indent, new_map))
        else:
            current[key] = _coerce_scalar(value_part)

    return root


def _load_yaml_file(path: Path) -> dict:
    content = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(content) or {}
    return _parse_basic_yaml(content)
