# src/analytics/plots.py

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.analytics.reporting import equity_to_series, trades_to_dataframe
from src.data.feeds import OHLCVArrays
from src.engine.core import BacktestResult


def plot_equity_curve(
    result: BacktestResult,
    data: OHLCVArrays,
    ax: Optional[plt.Axes] = None,
    strategy_name: Optional[str] = None,
) -> plt.Axes:
    """
    Dibuja la curva de equity en un gráfico.
    """
    if ax is None:
        _, ax = plt.subplots()

    eq_gross = equity_to_series(result, data, equity_field="equity")
    eq_net = None
    if getattr(result, "equity_net", None) is not None:
        eq_net = equity_to_series(result, data, equity_field="equity_net")

    ax.plot(eq_gross.index, eq_gross.values, label="Gross")
    if eq_net is not None and not eq_net.empty:
        ax.plot(eq_net.index, eq_net.values, label="Net")

    title = "Curva de Equity"
    if strategy_name:
        title = f"{title} - {strategy_name}"
    ax.set_title(title)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Equity")
    ax.legend()

    return ax


def plot_trades_per_month(
    result: BacktestResult,
    data: OHLCVArrays,
    ax: Optional[plt.Axes] = None,
    strategy_name: Optional[str] = None,
) -> plt.Axes:
    """
    Dibuja un gráfico de barras con el número de trades por mes,
    usando la fecha de entrada de cada trade.
    """
    if ax is None:
        _, ax = plt.subplots()

    trades_df = trades_to_dataframe(result, data)
    if trades_df.empty:
        base_title = "Número de trades por mes (sin trades)"
        if strategy_name:
            base_title = f"{base_title} - {strategy_name}"
        ax.set_title(base_title)
        return ax

    # Serie con 1 por trade indexada por fecha de entrada
    s = pd.Series(1, index=trades_df["entry_time"])

    # Recuento mensual
    trades_per_month = s.resample("ME").sum()

    ax.bar(trades_per_month.index, trades_per_month.values)
    base_title = "Número de trades por mes"
    if strategy_name:
        base_title = f"{base_title} - {strategy_name}"
    ax.set_title(base_title)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Nº de trades")

    return ax
