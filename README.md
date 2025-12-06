# Backtesting Engine — High-Performance Intraday Backtesting Framework

This project implements a **fast, modular, and reproducible backtesting engine** designed for **intraday algorithmic trading research**.  
It is built around a full data pipeline (Darwinex BID/ASK → parquet → OHLCV → NPZ) and a **Numba-accelerated backtesting core**, allowing realistic simulations with dynamic SL/TP, execution constraints, and strategy-driven signals.

---

## 📁 Project Structure

backtesting-engine/
│
├── main.py # Entry point orchestrating the entire pipeline
├── src/
│ ├── config/
│ │ └── paths.py # Global project paths and directory management
│ │
│ ├── data/ # Data ingestion, cleaning, and transformation
│ │ ├── data_utils.py
│ │ ├── data_to_parquet.py
│ │ ├── bars1m_to_excel.py
│ │ ├── csv_1m_to_npz.py
│ │ ├── parquet_to_npz.py
│ │ ├── feeds.py
│ │ └── organize_darwinex_files.py # (pending)
│ │
│ ├── engine/ # Numba-powered backtesting engine
│ │ └── core.py
│ │
│ ├── analytics/ # Metrics, reporting, visualizations
│ │ ├── metrics.py
│ │ ├── reporting.py
│ │ └── plots.py
│ │
│ ├── strategies/ # Strategy implementations
│ │ # e.g., momentum_opening.py, sweep_reversal.py, etc.
│ │
│ └── visualization/ # Future dashboard components
│
├── data/ # All datasets and generated artifacts
│ ├── raw/
│ │ └── darwinex/ # Original BID/ASK compressed logs
│ ├── parquet/
│ │ ├── ticks/
│ │ └── bars_1m/
│ ├── npz/
│ └── other/
│
├── notebooks/ # Exploratory research notebooks
└── reports/ # Generated reports (PDF/HTML)



---

## 🚀 Workflow Overview

### **1. Darwinex Tick Logs → Parquet**
The system processes raw BID/ASK logs into structured parquet files, organized by:
parquet/ticks/<symbol>/<year>/<symbol>_YYYY-MM-DD_HH.parquet
Handled by: `data_to_parquet.py`t
---

### **2. Parquet → 1-Minute OHLCV Bars**
All ticks are aggregated into a single unified CSV:
barras_1min_<symbol>_all_years.csv
Handled by: `bars1m_to_excel.py`

---

### **3. 1-Minute Bars → NPZ Arrays (Numba-Ready)**
Numerical arrays include:
timestamps, open, high, low, close, volume
Handled by: `csv_1m_to_npz.py`

---

### **4. Data Feed Loader**
Loads the NPZ arrays into structured objects:

```python
from src.data.feeds import NPZOHLCVFeed
data = NPZOHLCVFeed("NDXm", "1m").load_all()

### **5. Strategies**

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

### **6. Backtesting Engine**

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

Output includes:

Final cash and equity
Full bar-by-bar equity curve
Trade logs (entries, exits, PnL, duration)
Extra metadata (e.g., number of trades)


### **7. Metrics, Reporting & Visualization**

All analytical and visualization tools are located in: src/analytics/


This module provides three main components:

---

### **7.1 Performance Metrics**

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


### **7.2 Reporting Utilities**

The reporting utilities are located in:
src/analytics/reporting.py


This module provides functions to convert raw backtest results into clean, analysis-ready pandas structures.

#### **Available Functions**

---

#### **`equity_to_series(result, data)`**

Converts the engine’s internal equity array into a pandas `Series`, indexed by timestamps.

Outputs:

- Equity curve over time  
- Fully aligned with the OHLCV feed  
- Suitable for plotting and metric computation  

Example:

```python
from src.analytics.reporting import equity_to_series

eq_series = equity_to_series(result, data)
print(eq_series.tail())

trades_to_dataframe(result, data)

Creates a detailed trade log as a pandas DataFrame.

Columns typically include:

entry_time
exit_time
entry_price
exit_price
pnl_abs (absolute profit/loss)
pnl_pct (percentage return)
bars_in_trade
direction (long/short)

Example:

from src.analytics.reporting import trades_to_dataframe

trades_df = trades_to_dataframe(result, data)
print(trades_df.head())

### **7.3 Visualization Tools**

All visualization utilities are located in:
src/analytics/plots.py

These tools provide graphical analysis of strategy performance, including equity evolution, trade distribution, and behavioral patterns over time.

---

#### **Available Plotting Functions**

---

### **`plot_equity_curve(result, data, ax=None)`**

Plots the full **equity curve** using timestamps from the loaded OHLCV feed.

Features:

- Smooth equity line  
- Starting and ending capital visualization  
- Optional axis injection for subplot integration  

Example:

```python
from src.analytics.plots import plot_equity_curve
plot_equity_curve(result, data)


📄 License

MIT License — fully open for research and personal trading.
