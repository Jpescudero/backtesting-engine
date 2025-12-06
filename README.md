# Backtesting Engine — High-Performance Intraday Backtesting Framework


This project is a **high-performance intraday backtesting engine** built in **Python** and accelerated with **Numba**, designed to process **Darwinex BID/ASK market data** (currently focused on the *NDXm* index) and evaluate algorithmic trading strategies under realistic execution constraints.

It provides a **complete end-to-end data pipeline**—from raw Darwinex tick logs to aggregated 1-minute OHLCV bars, NumPy-optimized NPZ files, and finally a fully vectorized backtesting engine capable of handling thousands of trades efficiently.  
The framework includes:

- A robust **Darwinex data ingestion and cleaning pipeline**
- **1-minute bar generation** and NPZ conversion for ultra-fast backtests
- A **Numba-accelerated backtesting core** supporting SL/TP, max-duration rules, commission, slippage, and position sizing
- A modular **strategy layer** with an example “Opening Sweep / Liquidity Grab” strategy
- Comprehensive **reporting modules**: equity curves, performance metrics, trade analytics, and export to Excel/JSON
- Visualization tools for **best/worst trades**, equity curve plotting, and trade distributions

The goal of the framework is to serve as a **research-ready, production-oriented foundation** for exploring intraday pattern-based strategies—especially those relying on liquidity events (e.g., morning or afternoon opening sweeps) and high-volume reversal behavior that may be suitable for DARWIN-style systematic trading approaches.


---

## 📁 1.Project Structure

### 1.1 Table View

| Path                                 | Description |
|-------------------------------------|-------------|
| `main.py`                           | Entry point orchestrating full pipeline |
| `src/config/paths.py`               | Global paths and directory management |
| `src/data/data_utils.py`            | Helpers for file discovery & aggregation |
| `src/data/data_to_parquet.py`       | Convert Darwinex BID/ASK logs → parquet |
| `src/data/bars1m_to_excel.py`       | Build 1-minute OHLCV CSV files |
| `src/data/csv_1m_to_npz.py`         | Convert OHLCV CSV → NPZ (Numba-ready) |
| `src/data/parquet_to_npz.py`        | Convert parquet bars → NPZ arrays |
| `src/data/feeds.py`                 | Data feed (NPZ loader for the engine) |
| `src/data/organize_darwinex_files.py` | (Pending) Organize raw tick logs |
| `src/engine/core.py`                | Numba-powered backtesting engine |
| `src/analytics/metrics.py`          | Performance metrics |
| `src/analytics/reporting.py`        | Convert results → pandas structures |
| `src/analytics/plots.py`            | Equity/trade visualization tools |
| `src/strategies/`                   | Strategy modules |
| `src/visualization/`                | Future dashboards |
| `data/raw/darwinex/`                | Raw tick logs (BID/ASK) |
| `data/parquet/ticks/`               | Parquet files of cleaned ticks |
| `data/parquet/bars_1m/`             | Parquet 1-minute bars |
| `data/npz/`                         | Numba-ready NPZ files |
| `data/other/`                       | Misc additional files |
| `notebooks/`                        | Research notebooks |
| `reports/`                          | HTML/PDF reports |

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
    - `feeds.py` — Data feed loader for NPZ files
    - `organize_darwinex_files.py` — Pending
  - **engine/**
    - `core.py` — Backtesting engine (Numba)
  - **analytics/**
    - `metrics.py` — Performance metrics
    - `reporting.py` — Reporting utilities
    - `plots.py` — Visualization tools
  - **strategies/** — Strategy implementations
  - **visualization/** — Dashboard modules
- **data/**
  - `raw/darwinex/` — Original tick logs
  - `parquet/ticks/` — Cleaned tick files
  - `parquet/bars_1m/` — OHLCV bars
  - `npz/` — Numba-ready arrays
  - `other/`
- **notebooks/** — Research & experiments
- **reports/** — Generated reports

---

## 🚀 2. Workflow Overview

### **2.1. Darwinex Tick Logs → Parquet**

The system processes raw BID/ASK logs into structured parquet files, organized by:
parquet/ticks/<symbol>/<year>/<symbol>_YYYY-MM-DD_HH.parquet
Handled by: `data_to_parquet.py`t

---

### **2.2. Parquet → 1-Minute OHLCV Bars**
All ticks are aggregated into a single unified CSV:
barras_1min_<symbol>_all_years.csv
Handled by: `bars1m_to_excel.py`

---

### **2.3. 1-Minute Bars → NPZ Arrays (Numba-Ready)**
Numerical arrays include:
timestamps, open, high, low, close, volume
Handled by: `csv_1m_to_npz.py`

---

### **2.4. Data Feed Loader**
Loads the NPZ arrays into structured objects

---

### **2.5. Strategies**

All trading strategies reside inside:
src/strategies/


Each strategy is responsible for generating a **signal array** aligned with the OHLCV data:

- `+1` → long entry  
- `-1` → short entry (if enabled)  
- `0` → no trade  

Example strategy included:  
**Sweep-Reversal Opening Strategy (`StrategyBarridaApertura`)**

This strategy detects:

- High-volume washout candles near market open  
- One or two consecutive bearish bars  
- A local reversal trigger  

Usage:

```python
from src.strategies.barrida_apertura import StrategyBarridaApertura

strategy = StrategyBarridaApertura(
    volume_percentile=80,
    use_two_bearish_bars=True
)

signals = strategy.generate_signals(data).signals
```

---

### **2.6. Backtesting Engine**

The backtesting engine is implemented in:
src/engine/core.py

Its main responsibilities are:

- Execute strategies using precomputed **signals**
- Simulate order execution (market-style)
- Apply **commission**, **slippage**, and **trade sizing**
- Enforce risk rules:
  - Stop-Loss (percentage-based)
  - Take-Profit (percentage-based)
  - Maximum duration in bars
- Manage cash, equity, and portfolio state
- Produce a structured result object (`BacktestResult`)

Example usage:

```python
from src.engine.core import BacktestConfig, run_backtest_with_signals

config = BacktestConfig(
    initial_cash=100_000,
    commission_per_trade=1.0,
    slippage=0.0,
    trade_size=1.0,
    sl_pct=0.01,          # 1% stop-loss
    tp_pct=0.02,          # 2% take-profit
    max_bars_in_trade=60  # 1-hour limit on 1m bars
)

result = run_backtest_with_signals(data, signals, config)
```

Output includes:

Final cash and equity
Full bar-by-bar equity curve
Trade logs (entries, exits, PnL, duration)
Extra metadata (e.g., number of trades)

---

### **2.7. Metrics, Reporting & Visualization**

All analytical and visualization tools are located in: src/analytics/


This module provides three main components:

#### **2.7.1 Performance Metrics**

File:src/analytics/metrics.py

This module computes detailed performance statistics for both the equity curve and trade logs.

Available functions include:

- **`equity_curve_metrics(series)`**
  - Total Return  
  - Annualized Return  
  - Max Drawdown  
  - Volatility  
  - Sharpe-like ratios  
  - Return distribution metrics  

- **`trades_metrics(trades_df)`**
  - Win rate  
  - Average win / loss  
  - Expectancy  
  - Profit factor  
  - Average trade duration  
  - Best/worst trades  

Example usage:

```python
from src.analytics.metrics import equity_curve_metrics

metrics = equity_curve_metrics(eq_series)
print(metrics)
```

#### **2.7.2 Reporting Utilities**

The reporting utilities are located in:
src/analytics/reporting.py

This module provides functions to convert raw backtest results into clean, analysis-ready pandas structures.

#### **2.7.3 Visualization Tools**

All visualization utilities are located in:
src/analytics/plots.py

---


📄 License

MIT License — fully open for research and personal trading.
