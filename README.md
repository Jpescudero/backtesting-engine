# Backtesting Engine — High-Performance Intraday Backtesting Framework

This project is a **high-performance intraday backtesting engine** built in **Python** and accelerated with **Numba**. It turns **Darwinex BID/ASK market data** (currently focused on *NDXm*) into reproducible research pipelines and realistic execution simulations.

Latest highlights:

- 🔄 **End-to-end data pipeline**: automatically generate tick parquet files, 1-minute OHLCV CSVs, and NPZ arrays ready for Numba. Convenience helpers (`ensure_ticks_and_csv`, `ensure_npz_from_csv`, `prepare_npz_dataset`) make sure all artifacts exist before a run.
- 🧭 **Strategy catalog**: includes the **Microstructure Reversal** strategy (EMA/ATR context, pullback + exhaustion detection, structure-break filter, bullish shift candle), the **Microstructure Sweep** (stop-hunt + absorption with volume/volatility filters), and the existing opening-sweep example.
- ⚙️ **Cost-model aware backtesting core**: Numba-accelerated engine with stop-loss / take-profit, max duration, centralized transaction costs (via `config/costs/costs.yaml` + `CostModel`), gross/net PnL, entry thresholds, and position sizing through `BacktestConfig`.
- 📈 **Analytics & reports**: automatic conversion to pandas structures, equity/trade metrics, and lightweight Excel + JSON export with size-safe helpers. Generates equity + monthly trades plots and best/worst trade snapshots.
- 🎛️ **CLI orchestration**: `main.py` wraps a full pipeline (data prep → signals → backtest → reports) with flags for symbol, timeframe, strategy tuning, SL/TP, train/test years, and headless plotting.
- 🧾 **Centralized costs**: a single YAML file (`config/costs/costs.yaml`) drives all spread/commission/slippage assumptions via `src.costs.CostModel` for both research and engine workflows.
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
| `src/engine/core.py`                   | Numba-powered backtesting engine (gross + net PnL via centralized `CostModel`) |
| `src/analytics/metrics.py`             | Performance metrics |
| `src/analytics/reporting.py`           | Convert results → pandas structures |
| `src/analytics/plots.py`               | Equity/trade visualization tools |
| `src/analytics/trade_plots.py`         | Best/worst trade charting |
| `src/strategies/`                      | Strategy modules |
| `src/strategies/microstructure_reversal.py` | Microstructure Reversal strategy (pullback + structure break) |
| `src/strategies/microstructure_sweep.py` | Microstructure Sweep strategy (stop-hunt + absorption) |
| `src/utils/`                           | Timing utilities and shared helpers |
| `data/raw/darwinex/`                   | Raw tick logs (BID/ASK) |
| `data/parquet/ticks/`                  | Parquet files of cleaned ticks |
| `data/parquet/bars_1m/`                | Parquet 1-minute bars |
| `data/npz/`                            | Numba-ready NPZ files |
| `notebooks/`                           | Research notebooks |
| `reports/`                             | Generated reports and plots |
| `research/ml/build_ml_dataset_ndxm.py` | Build ML dataset (features + labels) from NDXm 1m bars |
| `research/ml/train_directional_model_ndxm.py` | Train baseline directional Gradient Boosting model |
| `research/microstructure/study_opening_sweeps.py` | Opening sweep stats study (metrics + plots) |

### 1.1.1 Script-by-script quick reference

This table expands on the individual scripts and modules so you can quickly understand what each one does when wiring the pipeline together or reusing parts in notebooks:

| Path / script | Purpose |
| --- | --- |
| `main.py` | CLI entry point that orchestrates the full pipeline (data prep → strategy signals → backtest → reports). Generates Excel/JSON summaries plus trade/equity PNGs, supports Microstructure Reversal (default) or Microstructure Sweep, and lets you pick train/test years, risk params, and advanced ATR/vol filters via flags. |
| `run_settings.example.txt` | Key-value configuration template for `main.py`. Copy it, fill in `initial_cash`, `train_years`, `test_years`, `use_test_years`, `strategy`, and optional sweep/reversal parameters, then pass it through `--config-file`. |
| `src/config/data_roots.example.json` | Suggested list of local/cloud data hubs (e.g., `C:/Users/JorgeP/Market Data`, `C:/Users/Jorge/Market Data`, `C:/Users/jorge/Market Data` + OneDrive mirror). Copy to `data_roots.json` to customize more machines and hubs. |
| `src/config/paths.py` | Centralized “GPS” for directories (project root, data, reports, etc.). |
| `src/pipeline/__init__.py` | Convenience imports for the pipeline package. |
| `src/pipeline/data_pipeline.py` | Ensures downstream artifacts exist by chaining ticks → CSV → NPZ generation. Ideal when you want a single call to prepare datasets before backtests. |
| `src/pipeline/backtest_runner.py` | High-level runner that combines strategy generation, Numba engine execution, and reporting in one call. |
| `src/pipeline/reporting.py` | Helpers to export metrics, plots, and final files (Excel/JSON + charts) in a backtest-friendly bundle. |
| `src/data/data_utils.py` | Utility layer for file discovery and OHLCV helpers such as `iter_ticks_by_year`, `make_ohlcv`, and `list_tick_files`. |
| `src/data/data_to_parquet.py` | Converts Darwinex BID/ASK logs into cleaned hourly parquet shards per symbol/year. |
| `src/data/bars1m_to_excel.py` | Aggregates parquet ticks into a consolidated 1-minute OHLCV CSV covering all years. |
| `src/data/csv_1m_to_npz.py` | Turns the 1-minute CSV into compact NPZ arrays (`timestamps, open, high, low, close, volume`). |
| `src/data/parquet_to_npz.py` | Generic parquet-to-NPZ converter with `bars_df_to_npz_arrays` for alternate bar sources. |
| `src/data/feeds.py` | `NPZOHLCVFeed` loader that returns Numba-friendly OHLCV arrays and can slice by year (`load_years`) or load all data. |
| `src/engine/core.py` | Numba-powered backtesting engine with `BacktestConfig`, `BacktestResult`, and `run_backtest_with_signals` for applying SL/TP, max duration, and execution costs. |
| `src/analytics/metrics.py` | Computes equity/trade metrics (returns, drawdown, expectancy, win rate, etc.) in a Darwinex-like style. |
| `src/analytics/reporting.py` | Converts raw engine outputs into pandas structures via `equity_to_series` and `trades_to_dataframe`. |
| `src/analytics/plots.py` | Plotting helpers for equity curves, monthly trades, and related visuals. |
| `src/analytics/backtest_output.py` | Exports lightweight Excel/JSON summaries while keeping file sizes manageable. |
| `src/analytics/trade_plots.py` | Renders best/worst trade charts over 1m bars with annotated entries/exits. |
| `src/strategies/base.py` | Base classes and shared types for strategy modules. |
| `src/strategies/barrida_apertura.py` | Opening “sweep” strategy around 09:00/15:00 Europe/Madrid sessions. |
| `src/strategies/microstructure_reversal.py` | Microstructure Reversal strategy (pullback + structure break + bullish shift candle) with EMA/ATR context filters. |
| `src/strategies/microstructure_sweep.py` | Microstructure Sweep strategy (sweep + absorption) with ATR/volume filters, intraday windows, and trade caps. |
| `src/utils/timing.py` | Context manager to time blocks and log duration. |
| `src/utils/risk.py` | Dynamic position sizing given SL/TP and risk allowance. |
| `src/visualization/visualizacion.py` | Early-stage notebook-friendly plotting ideas for future dashboards. |
| `notebooks/` | Research notebooks for experiments. |
| `reports/` | Output folder for generated reports, plots, and exports. |
| `research/ml/build_ml_dataset_ndxm.py` | CLI script to build ML features/labels from NDXm 1m bars and store parquet datasets under `data/ml/`. |
| `research/ml/train_directional_model_ndxm.py` | Trains a baseline Gradient Boosting classifier on the ML dataset and saves model/scaler artifacts plus metrics. |
| `research/microstructure/study_opening_sweeps.py` | Opening sweep statistical study (session-filtered signals, forward returns, CSV exports, and plots under `reports/research/microstructure/`). |

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
- **research/** — ML datasets/models and microstructure studies (`ml/`, `microstructure/`)

### 1.3 Guía de enlaces en GitHub

Para compartir enlaces rápidos del repositorio (README, scripts principales, estrategias, pruebas, notebooks, etc.) con ChatGPT u otros colaboradores, consulta la [guía de links en GitHub](docs/github_links_guide.md) y sustituye `<OWNER>/<REPO>` por el nombre real del repositorio.

### 1.5 Calidad y flujo de desarrollo

- **Formateo y linting**: configuración centralizada en `pyproject.toml` para **Black** (100 columnas), **Ruff** (incluye reglas de importación isort) y **isort** con `src` como primer partido.
- **Tipado**: **mypy** se ejecuta sobre `src` y `tests` con Python 3.10.
- **Cobertura**: se exige un mínimo del 80 % (`fail_under = 80`) y se genera `coverage.xml`.
- **CI (GitHub Actions)**: el workflow `.github/workflows/ci.yml` instala `requirements-dev.txt` y ejecuta Ruff, Black, isort, mypy y `pytest --cov` subiendo el XML como artefacto.
- **Pre-commit**: `.pre-commit-config.yaml` añade hooks para Ruff (lint + format), Black, isort y comprobaciones básicas de espacio en blanco. Ejecuta `pre-commit install` tras clonar para activar los hooks locales.
- **Resumen en lenguaje llano**: si quieres una versión corta y sin tecnicismos sobre por qué se añadió todo esto, mira `docs/tooling_explicacion.md`.

### 1.4 Project layout reference from `backtesting_project_structure.txt`

For a quick orientation, this mirrors the annotated structure documented in `backtesting_project_structure.txt` (useful when browsing the repo or mapping local folders):

- **Executable & configs**
  - `main.py` — entry point for running Microstructure Reversal/Sweep backtests, generating Excel/JSON summaries and trade PNGs, and selecting train/test years and capital.
  - `run_settings.example.txt` — template of `key=value` settings for `main.py` (capital, train/test years, strategy selection, sweep parameters, etc.). Copy it and provide via `--config-file`.
- **Source tree (`src/`)**
  - `config/paths.py` — defines project/data/report directories.
  - `pipeline/` — orchestration helpers: `data_pipeline.py` (ensure ticks → CSV → NPZ), `backtest_runner.py` (strategy + engine + reporting), `reporting.py` (export metrics/plots/files), and `__init__.py` for package imports.
  - `data/` — ingestion and transformation scripts: `data_utils.py`, `data_to_parquet.py` (Darwinex logs → parquet), `bars1m_to_excel.py` (parquet ticks → 1m CSV), `csv_1m_to_npz.py` (CSV → NPZ), `parquet_to_npz.py` (generic parquet → NPZ), and `feeds.py` (NPZ loader for Numba).
  - `engine/core.py` — Numba backtesting core (`BacktestConfig`, `BacktestResult`, `run_backtest_with_signals`).
  - `analytics/` — metrics (`metrics.py`), pandas converters (`reporting.py`), plots (`plots.py`), exports (`backtest_output.py`), and best/worst trade visuals (`trade_plots.py`).
  - `strategies/` — concrete strategies: `barrida_apertura.py`, `microstructure_reversal.py`, `microstructure_sweep.py`, plus `base.py` for shared types.
  - `utils/` — shared utilities: `timing.py` (timers/logging) and `risk.py` (position sizing).
  - `visualization/visualizacion.py` — early notebook-friendly visualization draft.
- **Data folders (`data/`)**
  - `raw/darwinex/<SYMBOL>/BID|ASK` — compressed Darwinex BID/ASK logs (e.g., `NDXm_BID_2021-04-09_14.log.gz`).
  - `parquet/ticks/<SYMBOL>/<YEAR>/` — cleaned tick parquet shards (e.g., `NDXm_2021-04-09_14.parquet`).
  - `parquet/bars_1m/<SYMBOL>/` — optional parquet 1-minute bars per year.
  - `npz/<SYMBOL>/` — NPZ bundles (`<symbol>_all_1m.npz`) used by the engine.
- **Outputs**
  - `reports/` — generated plots and Excel/JSON summaries.

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
- **Microstructure Sweep** (`StrategyMicrostructureSweep`): captures stop-hunts followed by absorption and confirmation. Uses dual ATR timeframes, minimum sweep breaks of previous lows, wick/body and range filters, confirmation candle requirements, session windows, per-day trade caps, and volume/ATR guards (relative volume + intraday ATR median).
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

Microstructure Sweep example (external ATR for a dedicated timeframe):

```python
from src.strategies.microstructure_sweep import StrategyMicrostructureSweep

sweep = StrategyMicrostructureSweep(
    sweep_lookback=15,
    min_sweep_break_atr=0.35,
    min_lower_wick_body_ratio=1.5,
    confirm_body_atr=0.35,
)

signals = sweep.generate_signals(data, external_atr=lower_tf_atr).signals
```

### 2.4 Backtesting engine

Implemented in `src/engine/core.py`. Key features:

- Vectorized, Numba-accelerated execution for thousands of trades.
- Stop-loss / take-profit, maximum bars in trade, entry thresholds, and centralized transaction costs (gross + net) via `BacktestConfig` + `CostModel`.
- Returns a structured `BacktestResult` with cash, position, equity curve, trade logs, and metadata.

```python
import numpy as np

from src.engine.core import BacktestConfig, run_backtest_with_signals
from src.utils import compute_position_size

config = BacktestConfig(
    initial_cash=100_000,
    trade_size=1.0,
    sl_pct=0.01,
    tp_pct=0.02,
    max_bars_in_trade=60,
    cost_config_path="config/costs/costs.yaml",
    cost_instrument="NDX",
)

result = run_backtest_with_signals(data, signals, config, cost_model=CostModel.from_yaml("config/costs/costs.yaml", "NDX"))
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
- Strategy selection: `--strategy microstructure_reversal` (default) or `--strategy microstructure_sweep`.
- Microstructure Reversal tuning: `--ema-short`, `--ema-long`, `--atr-period`, `--min-pullback-atr`, `--max-pullback-atr`, `--max-pullback-bars`, `--exhaustion-close-min`, `--exhaustion-close-max`, `--exhaustion-body-max-ratio`, `--shift-body-atr`, `--structure-break-lookback`.
- Microstructure Sweep tuning: `--atr-timeframe`, `--atr-timeframe-period`, `--sweep-lookback`, `--min-sweep-break-atr`, `--min-lower-wick-body-ratio`, `--min-sweep-range-atr`, `--confirm-body-atr`, `--no-confirm-close-above-mid`, `--volume-period`, `--min-rvol`, `--vol-percentile-min`, `--vol-percentile-max`, `--max-atr-mult-intraday`, `--max-trades-per-day`, `--sweep-max-holding-bars`, `--atr-stop-mult`, `--rr-multiple`.
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

Microstructure Sweep quick start:

```bash
python main.py --strategy microstructure_sweep --symbol NDXm --timeframe 1m \
  --atr-timeframe 1m --atr-timeframe-period 10 --sweep-lookback 15 --min-sweep-break-atr 0.35 \
  --min-lower-wick-body-ratio 1.5 --min-sweep-range-atr 0.5 --confirm-body-atr 0.35 \
  --volume-period 20 --min-rvol 1.0 --vol-percentile-min 0.8 --vol-percentile-max 1.0 \
  --max-trades-per-day 4 --sweep-max-holding-bars 60 --atr-stop-mult 0.2 --rr-multiple 2.5 \
  --sl-pct 0.01 --tp-pct 0.02 --commission 1.0 --slippage 0.0 --trade-size 1.0 --headless
```

---

📄 License

MIT License — fully open for research and personal trading.
