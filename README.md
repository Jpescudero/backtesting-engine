# Backtesting Engine — High-Performance Intraday Backtesting Framework

This project is a **high-performance intraday backtesting engine** built in **Python** and accelerated with **Numba**. It turns **Darwinex BID/ASK market data** (currently focused on *NDXm*) into reproducible research pipelines and realistic execution simulations.

Latest highlights:

- 🔄 **End-to-end data pipeline**: automatically generate tick parquet files, 1-minute OHLCV CSVs, and NPZ arrays ready for Numba. Convenience helpers (`ensure_ticks_and_csv`, `ensure_npz_from_csv`, `prepare_npz_dataset`) make sure all artifacts exist before a run.
- 🧭 **Strategy catalog**: includes the **Microstructure Reversal** strategy with rich parameters (EMA/ATR context, pullback and exhaustion detection, structure-break filter, bullish shift candle) plus the existing opening-sweep example.
- ⚙️ **Configurable backtesting core**: Numba-accelerated engine with stop-loss / take-profit, max duration, commission, slippage, entry thresholds, and position sizing via `BacktestConfig`.
- 📈 **Analytics & reports**: automatic conversion to pandas structures, equity/trade metrics, and lightweight Excel + JSON export with size-safe helpers. Generates equity + monthly trades plots and best/worst trade snapshots.
- 🎛️ **CLI orchestration**: `main.py` wraps a full pipeline (data prep → signals → backtest → reports) with flags for symbol, timeframe, strategy tuning, SL/TP, slippage/commission, train/test years, and headless plotting.
- 🗂️ **Train/test slicing**: load specific years from NPZ feeds to isolate training vs testing periods without touching raw files.

The framework aims to be a **research-ready, production-oriented foundation** for exploring intraday pattern-based strategies—especially those relying on liquidity events and high-volume reversal behavior suitable for DARWIN-style systematic trading.

---

## 📁 1. Project Structure

### 1.1 Table View

| Path                                   | Description |
|----------------------------------------|-------------|
| `main.py`                              | CLI entry point orchestrating full pipeline |
| `src/config/paths.py`                  | Global paths and directory management |
| `src/data/data_utils.py`               | Helpers for file discovery & aggregation |
| `src/data/data_to_parquet.py`          | Convert Darwinex BID/ASK logs → parquet |
| `src/data/bars1m_to_excel.py`          | Build 1-minute OHLCV CSV files |
| `src/data/csv_1m_to_npz.py`            | Convert OHLCV CSV → NPZ (Numba-ready) |
| `src/data/parquet_to_npz.py`           | Convert parquet bars → NPZ arrays |
| `src/data/feeds.py`                    | Data feed (NPZ loader with year slicing) |
| `src/pipeline/data_pipeline.py`        | High-level helpers to ensure CSV/NPZ artifacts exist |
| `src/pipeline/backtest_runner.py`      | Orchestrates data prep, signals, backtests, and reports |
| `src/pipeline/reporting.py`            | Analytics, plots, and export helpers |
| `src/engine/core.py`                   | Numba-powered backtesting engine |
| `src/analytics/metrics.py`             | Performance metrics |
| `src/analytics/reporting.py`           | Convert results → pandas structures |
| `src/analytics/plots.py`               | Equity/trade visualization tools |
| `src/analytics/trade_plots.py`         | Best/worst trade charting |
| `src/strategies/`                      | Strategy modules |
| `src/utils/`                           | Timing utilities and shared helpers |
| `data/raw/darwinex/`                   | Raw tick logs (BID/ASK) |
| `data/parquet/ticks/`                  | Parquet files of cleaned ticks |
| `data/parquet/bars_1m/`                | Parquet 1-minute bars |
| `data/npz/`                            | Numba-ready NPZ files |
| `notebooks/`                           | Research notebooks |
| `reports/`                             | Generated reports and plots |

### 1.2 Codeblock

- **main.py** — Main entry point of the entire workflow
- **src/**
  - **config/**
    - `paths.py` — Defines all project directory paths
  - **data/**
    - `data_utils.py` — Helper utilities
    - `data_to_parquet.py` — Convert Darwinex logs → parquet
    - `bars1m_to_excel.py` — Build 1m OHLCV bars
    - `csv_1m_to_npz.py` — Convert 1m bars → NPZ
    - `parquet_to_npz.py` — Convert parquet bars → NPZ
    - `feeds.py` — Data feed loader for NPZ files (with `load_all` / `load_years`)
  - **pipeline/**
    - `data_pipeline.py` — Ensure ticks → CSV → NPZ artifacts exist
    - `backtest_runner.py` — Run strategies + engine + reports in one call
    - `reporting.py` — Analytics + plots + Excel/JSON export helpers
  - **engine/**
    - `core.py` — Backtesting engine (Numba)
  - **analytics/**
    - `metrics.py` — Performance metrics
    - `reporting.py` — Reporting utilities
    - `plots.py` — Visualization tools
    - `trade_plots.py` — Best/worst trade snapshots
  - **strategies/** — Strategy implementations
  - **utils/** — Timing and misc helpers
- **data/**
  - `raw/darwinex/` — Original tick logs
  - `parquet/ticks/` — Cleaned tick files
  - `parquet/bars_1m/` — OHLCV bars
  - `npz/` — Numba-ready arrays
- **notebooks/** — Research & experiments
- **reports/** — Generated reports

---

## 🚀 2. Workflow Overview

### 2.1 Data ingestion (Darwinex tick logs → parquet → CSV → NPZ)

1. **Tick parquet generation** — `data_to_parquet.py` converts BID/ASK logs into hourly parquet shards under `data/parquet/ticks/<symbol>/<year>/`.
2. **1-minute bars** — `bars1m_to_excel.py` aggregates all ticks into a unified CSV (`barras_1min_<symbol>_all_years.csv`).
3. **NPZ arrays (Numba-ready)** — `csv_1m_to_npz.py` turns the CSV into compact NumPy arrays (`timestamps, open, high, low, close, volume`).
4. **Automation helpers** — `prepare_npz_dataset` chains the previous steps and guarantees both the CSV and NPZ exist, so pipelines can be started with a single call.

### 2.2 Loading data for backtests

`NPZOHLCVFeed` (src/data/feeds.py) streams NPZ arrays and can slice specific years via `load_years` or load the full history with `load_all`. This makes it easy to separate **training** and **testing** windows without duplicating files.

### 2.3 Strategies

All strategies live in `src/strategies/` and return **signal arrays** aligned with OHLCV data.

- **Microstructure Reversal** (`StrategyMicrostructureReversal`): long-only, trend-filtered setup that detects pullbacks, exhaustion bars, a bullish shift candle, and a break of short-term structure. Parameters cover EMA/ATR configuration, pullback depth/length, exhaustion candle shape, shift candle strength, and structure-break lookback.
- **Opening Sweep / Liquidity Grab** (`StrategyBarridaApertura`): example strategy for high-volume morning sweeps.

Example:

```python
from src.strategies.microstructure_reversal import StrategyMicrostructureReversal

strategy = StrategyMicrostructureReversal(
    ema_short=20,
    ema_long=50,
    atr_period=20,
    min_pullback_atr=0.3,
    max_pullback_atr=1.3,
    max_pullback_bars=12,
)

signals = strategy.generate_signals(data).signals
```

### 2.4 Backtesting engine

Implemented in `src/engine/core.py`. Key features:

- Vectorized, Numba-accelerated execution for thousands of trades.
- Stop-loss / take-profit, maximum bars in trade, entry thresholds, commission, slippage, and position sizing via `BacktestConfig`.
- Returns a structured `BacktestResult` with cash, position, equity curve, trade logs, and metadata.

```python
import numpy as np

from src.engine.core import BacktestConfig, run_backtest_with_signals
from src.utils import compute_position_size

config = BacktestConfig(
    initial_cash=100_000,
    commission_per_trade=1.0,
    slippage=0.0,
    trade_size=1.0,
    sl_pct=0.01,
    tp_pct=0.02,
    max_bars_in_trade=60,
)

result = run_backtest_with_signals(data, signals, config)
```

To run strategies with their own risk parameters, pre-compute position sizes per bar and
feed them into the engine. This keeps SL/TP logic under the strategy while reusing the
common executor:

```python
sizes = np.array(
    [
        compute_position_size(
            equity=current_equity[i],
            entry_price=entries[i],
            stop_loss=stops[i],
            take_profit=tps[i],
            risk_pct=0.0075,
            point_value=1.0,
        )
        for i in range(len(signals))
    ],
    dtype=float,
)

result = run_backtest_with_signals(data, signals, position_sizes=sizes, config=config)
```

### 2.5 Analytics, reporting & visualization

Located under `src/analytics/` and `src/pipeline/reporting.py`:

- **Metric calculators** (`equity_curve_metrics`, `trades_metrics`) for returns, drawdowns, Sharpe-like ratios, expectancy, win rate, trade duration, and more.
- **Pandas converters** (`equity_to_series`, `trades_to_dataframe`) to turn engine outputs into analysis-ready structures.
- **Plots** (`plot_equity_curve`, `plot_trades_per_month`, `plot_best_and_worst_trades`) for equity curves, monthly trade distributions, and best/worst trade breakdowns.
- **Report artifacts** (`generate_report_files`) save lightweight Excel + JSON summaries with size guards to keep exports and load times manageable.

### 2.6 Orchestrated pipeline (CLI)

`main.py` wires everything together. It uses `run_single_backtest` to:

1. Prepare data (generate parquet/CSV/NPZ if missing).
2. Load NPZ feeds (optionally selecting train/test years).
3. Generate strategy signals (Microstructure Reversal by default).
4. Run the Numba engine with the chosen `BacktestConfig`.
5. Compute metrics, export Excel/JSON, and render equity + trade plots (headless-friendly).

Common flags:

- `--symbol`, `--timeframe`: select instrument and bar size.
- Strategy tuning: `--ema-short`, `--ema-long`, `--atr-period`, `--min-pullback-atr`, `--max-pullback-atr`, `--max-pullback-bars`, `--exhaustion-close-min`, `--exhaustion-close-max`, `--exhaustion-body-max-ratio`, `--shift-body-atr`, `--structure-break-lookback`.
- Risk/execution: `--initial-cash`, `--commission`, `--trade-size`, `--slippage`, `--sl-pct`, `--tp-pct`, `--max-bars`, `--entry-threshold`.
- Data splits: `--train-years 2019,2020`, `--test-years 2021,2022`, `--use-test-years`.
- Reporting: `--no-report-files`, `--no-main-plots`, `--no-trade-plots`, `--headless`.

Example CLI run:

```bash
python main.py --symbol NDXm --timeframe 1m --ema-short 20 --ema-long 50 \
  --atr-period 20 --min-pullback-atr 0.3 --max-pullback-atr 1.3 --max-pullback-bars 12 \
  --sl-pct 0.01 --tp-pct 0.02 --commission 1.0 --slippage 0.0 --trade-size 1.0 \
  --train-years 2019,2020 --test-years 2021,2022 --headless
```

---

📄 License

MIT License — fully open for research and personal trading.
