"""Unit tests for the centralized CostModel."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.costs import CostModel


FIXTURE_PATH = Path("config/costs/costs.yaml")


@pytest.mark.parametrize("instrument", ["NDX", "NDXm"])
def test_from_yaml_reads_instrument(instrument: str) -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), instrument)
    assert model.config.instrument == instrument
    assert model.config.contract_multiplier == 1.0


def test_from_yaml_resolves_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root / "research")
    model = CostModel.from_yaml("config/costs/costs.yaml", "NDX")
    assert model.config.instrument == "NDX"


def test_breakdown_points_long_vs_short_equivalence() -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), "NDXm")
    long_cost = model.breakdown(100.0, 105.0, "long", qty=1.0)["total_cost"]
    short_cost = model.breakdown(105.0, 100.0, "short", qty=1.0)["total_cost"]
    assert math.isclose(long_cost, short_cost)


def test_breakdown_points_components() -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), "NDX")
    breakdown = model.breakdown(100.0, 101.0, "long", qty=2.0)
    assert breakdown["commission"] == pytest.approx(1.0 * 2.0 * 2.0)
    assert breakdown["spread"] == pytest.approx(1.0 * 1.0 * 2.0)
    assert breakdown["slippage"] == pytest.approx(0.5 * 1.0 * 2.0 * 2.0)


def test_zero_costs_no_effect_on_returns() -> None:
    zero_model = CostModel.from_yaml(str(FIXTURE_PATH), "NDXm")
    zero_model.config.commission_per_side = 0.0
    zero_model.config.slippage = 0.0
    zero_model.config.spread = 0.0
    net_return = zero_model.apply_to_gross_return(0.02, 100.0, 102.0, "long")
    assert net_return == pytest.approx(0.02)


def test_qty_multiplier() -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), "NDXm")
    cost_single = model.estimate_trade_cost(100.0, 102.0, "long", qty=1.0)
    cost_double = model.estimate_trade_cost(100.0, 102.0, "long", qty=2.0)
    assert cost_double == pytest.approx(cost_single * 2.0)


def test_bps_cost_type() -> None:
    tmp_path = Path("/tmp/test_costs.yaml")
    tmp_path.write_text(
        """
default:
  cost_type: "bps"
  commission_per_side: 10.0
  spread: 5.0
  slippage: 2.5
  contract_multiplier: 1.0
  currency: "USD"

instruments:
  TEST:
    cost_type: "bps"
    commission_per_side: 10.0
    spread: 5.0
    slippage: 2.5
    contract_multiplier: 1.0
    venue: "generic"
        """
    )

    model = CostModel.from_yaml(str(tmp_path), "TEST")
    breakdown = model.breakdown(100.0, 101.0, "long", qty=1.0)
    expected_notional = 100.0
    expected_commission = expected_notional * (10.0 / 10_000.0) * 2.0
    expected_spread = (100.5) * (5.0 / 10_000.0)
    expected_slippage = (100.5) * (2.5 / 10_000.0) * 2.0
    assert breakdown["commission"] == pytest.approx(expected_commission)
    assert breakdown["spread"] == pytest.approx(expected_spread)
    assert breakdown["slippage"] == pytest.approx(expected_slippage)
    assert breakdown["total_cost"] == pytest.approx(
        expected_commission + expected_spread + expected_slippage
    )


def test_entry_equals_exit() -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), "NDX")
    breakdown = model.breakdown(100.0, 100.0, "short", qty=1.0)
    assert breakdown["total_cost"] >= 0.0


def test_invalid_side_raises() -> None:
    model = CostModel.from_yaml(str(FIXTURE_PATH), "NDX")
    with pytest.raises(ValueError):
        model.breakdown(100.0, 101.0, "flat", qty=1.0)  # type: ignore[arg-type]
