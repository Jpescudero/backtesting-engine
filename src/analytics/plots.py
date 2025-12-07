# src/analytics/plots.py

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.engine.core import BacktestResult
from src.data.feeds import OHLCVArrays
from src.analytics.reporting import equity_to_series, trades_to_dataframe


def plot_equity_curve(
    result: BacktestResult,
    data: OHLCVArrays,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Dibuja la curva de equity en un gráfico.
    """
    if ax is None:
        _, ax = plt.subplots()

    eq = equity_to_series(result, data)
    ax.plot(eq.index, eq.values)
    ax.set_title("Curva de Equity")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Equity")

    return ax


def plot_trades_per_month(
    result: BacktestResult,
    data: OHLCVArrays,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Dibuja un gráfico de barras con el número de trades por mes,
    usando la fecha de entrada de cada trade.
    """
    if ax is None:
        _, ax = plt.subplots()

    trades_df = trades_to_dataframe(result, data)
    if trades_df.empty:
        ax.set_title("Número de trades por mes (sin trades)")
        return ax

    # Serie con 1 por trade indexada por fecha de entrada
    s = pd.Series(1, index=trades_df["entry_time"])

    # Recuento mensual
    trades_per_month = s.resample("ME").sum()

    ax.bar(trades_per_month.index, trades_per_month.values)
    ax.set_title("Número de trades por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Nº de trades")

    return ax
