# Cost Model

This module centralizes every transaction cost assumption for the project. All
research notebooks, scripts, the backtesting engine, and production runners must
consume costs exclusively through `CostModel` and the single configuration file
`config/costs/costs.yaml`.

## Configuration file

Costs are defined in YAML under `config/costs/costs.yaml` with a `default`
section and an `instruments` map. Example:

```yaml
default:
  cost_type: "points"
  commission_per_side: 0.0
  spread: 0.0
  slippage: 0.0
  contract_multiplier: 1.0
  currency: "USD"

instruments:
  NDX:
    cost_type: "points"
    commission_per_side: 1.0
    spread: 1.0
    slippage: 0.5
    contract_multiplier: 1.0
    venue: "generic"
```

The `cost_type` flag determines whether spread, commission, and slippage are
expressed in **price points** (`points`) or **basis points of notional**
(`bps`). `contract_multiplier` converts price differences into currency PnL.

## API

`CostModel` exposes:

- `CostModel.from_yaml(path, instrument)` – build a model from the YAML file.
- `estimate_trade_cost(entry_price, exit_price, side, qty=1.0)` – total monetary
  round-trip cost.
- `apply_to_gross_return(gross_return, entry_price, exit_price, side, qty=1.0)`
  – gross-to-net conversion for normalized returns.
- `breakdown(entry_price, exit_price, side, qty=1.0)` – dictionary with
  `commission`, `spread`, `slippage`, and `total_cost` plus the normalized
  `cost_return`.

## Usage guidelines

1. **Single source of truth:** never hardcode commission, spread, or slippage in
   scripts or notebooks. Always load the YAML and build a `CostModel`.
2. **Gross vs net:** keep gross metrics (before costs) and net metrics (after
   costs) side by side. Change the YAML to propagate new assumptions everywhere.
3. **Extending to Darwinex:** add new venues/instruments under the `instruments`
   section with their contract specs and fees. No code changes should be
   required as long as the structure remains the same.
