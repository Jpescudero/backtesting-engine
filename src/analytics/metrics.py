# src/analytics/metrics.py

from __future__ import annotations

from math import sqrt
from typing import Any, Dict

import numpy as np
import pandas as pd


def equity_curve_metrics(equity: pd.Series) -> Dict[str, Any]:
    """
    Calcula métricas tipo 'Darwinex style' sobre una curva de equity.

    equity: pd.Series con índice datetime y valores de equity en dinero.

    Devuelve cosas como:
      - total_return
      - annualized_return
      - max_drawdown
      - return_drawdown_ratio
      - sharpe_ratio
      - sortino_ratio
      - volatility_annual
      - var_95_monthly (histórico, aproximado)
    """
    if equity.empty:
        return {}

    # Aseguramos orden temporal
    equity = equity.sort_index()

    # --- Retornos simples ---
    equity = equity[equity != 0]
    if equity.empty:
        return {}

    start_eq = float(equity.iloc[0])
    end_eq = float(equity.iloc[-1])

    if start_eq != 0:
        total_return = end_eq / start_eq - 1.0
    else:
        total_return = np.nan

    # Horizonte en años
    delta_days = (equity.index[-1] - equity.index[0]).days
    years = delta_days / 365.25 if delta_days > 0 else 0.0

    if years > 0:
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        annualized_return = np.nan

    # --- Drawdown (estilo Darwinex: sobre curva de retorno/equity) ---
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_drawdown = float(dd.min())

    if max_drawdown < 0:
        return_dd_ratio = total_return / abs(max_drawdown)
    else:
        return_dd_ratio = np.nan

    # --- Retornos diarios para Sharpe / Sortino ---
    # Cogemos último valor de equity por día
    eq_daily = equity.resample("1D").last().dropna()
    ret_daily = eq_daily.pct_change().dropna()

    if len(ret_daily) > 1:
        mean_daily = float(ret_daily.mean())
        std_daily = float(ret_daily.std(ddof=1))  # ddof=1 como en estadística clásica

        # Vol annualizada (suponiendo 252 días de mercado)
        volatility_annual = std_daily * sqrt(252.0)

        if volatility_annual > 0:
            sharpe_ratio = (mean_daily * 252.0) / volatility_annual  # rf ~ 0
        else:
            sharpe_ratio = np.nan

        # Sortino: solo desviación de retornos negativos
        neg = ret_daily[ret_daily < 0]
        if len(neg) > 0:
            std_down = float(neg.std(ddof=1))
            if std_down > 0:
                sortino_ratio = (mean_daily * 252.0) / (std_down * sqrt(252.0))
            else:
                sortino_ratio = np.nan
        else:
            sortino_ratio = np.nan
    else:
        volatility_annual = np.nan
        sharpe_ratio = np.nan
        sortino_ratio = np.nan

    # --- Var mensual histórica aproximada (95%) ---
    # Proyección muy simple: coger retornos mensuales de equity
    eq_monthly = equity.resample("ME").last().dropna()
    monthly_returns = eq_monthly.pct_change().dropna()

    if len(monthly_returns) > 0:
        # percentil 5 (5% peor caso) -> VaR 95%
        var_95_monthly = float(np.percentile(monthly_returns, 5) * -1.0)
    else:
        var_95_monthly = np.nan

    return {
        "start_equity": start_eq,
        "end_equity": end_eq,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "return_drawdown_ratio": return_dd_ratio,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "volatility_annual": volatility_annual,
        "var_95_monthly": var_95_monthly,
        "n_days": len(eq_daily),
        "n_months": len(monthly_returns),
    }


def trades_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Métricas sobre el conjunto de trades:
      - número de trades
      - winrate
      - payoff ratio (media ganadores / media perdedores)
      - expectancy media por trade
      - duración media (en barras)
      - porcentaje de salidas por SL/TP/time/signal
    """
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "winrate": np.nan,
            "avg_pnl": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "expectancy_per_trade": np.nan,
            "avg_holding_bars": np.nan,
            "exit_reason_counts": {},
        }

    n_trades = len(trades)
    pnl = trades["pnl"]

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    winrate = len(wins) / n_trades if n_trades > 0 else np.nan
    avg_pnl = float(pnl.mean())

    avg_win = float(wins.mean()) if len(wins) > 0 else np.nan
    avg_loss = float(losses.mean()) if len(losses) > 0 else np.nan

    if not np.isnan(avg_win) and not np.isnan(avg_loss) and avg_loss != 0:
        payoff_ratio = abs(avg_win / avg_loss)
    else:
        payoff_ratio = np.nan

    # Expectancy sencilla en dinero por trade (ya es la media)
    expectancy_per_trade = avg_pnl

    # Duración media
    if "holding_bars" in trades.columns:
        avg_holding_bars = float(trades["holding_bars"].mean())
    else:
        avg_holding_bars = np.nan

    # Distribución de motivos de salida (si está la columna exit_reason)
    reason_counts = {}
    if "exit_reason" in trades.columns:
        reason_counts = trades["exit_reason"].value_counts().to_dict()

    return {
        "n_trades": n_trades,
        "winrate": winrate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_per_trade": expectancy_per_trade,
        "avg_holding_bars": avg_holding_bars,
        "exit_reason_counts": reason_counts,
    }


def trade_level_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    """Estadísticas sobre la amplitud de SL/TP y frecuencia de break-even."""

    if trades is None or trades.empty:
        return {
            "sl_distance_pct": {},
            "tp_distance_pct": {},
            "sl_distance_abs": {},
            "tp_distance_abs": {},
            "breakeven_count": 0,
            "breakeven_rate": 0.0,
        }

    entry_price = trades["entry_price"].astype(float)
    side = np.sign(trades.get("qty", 1.0)).replace(0, 1.0)

    sl_level = trades.get("stop_loss")
    tp_level = trades.get("take_profit")

    sl_distance = _signed_distance(entry_price, sl_level, side)
    tp_distance = _signed_distance(tp_level, entry_price, side)

    stats_sl_abs = _distance_stats(sl_distance)
    stats_tp_abs = _distance_stats(tp_distance)

    stats_sl_pct = _distance_stats(sl_distance / entry_price)
    stats_tp_pct = _distance_stats(tp_distance / entry_price)

    pnl = trades.get("pnl", pd.Series(dtype=float)).astype(float)
    tolerance = (entry_price * 1e-4).abs()
    breakeven_mask = pnl.abs() <= tolerance
    breakeven_count = int(breakeven_mask.sum())
    breakeven_rate = float(breakeven_count / len(trades)) if len(trades) else 0.0

    return {
        "sl_distance_pct": stats_sl_pct,
        "tp_distance_pct": stats_tp_pct,
        "sl_distance_abs": stats_sl_abs,
        "tp_distance_abs": stats_tp_abs,
        "breakeven_count": breakeven_count,
        "breakeven_rate": breakeven_rate,
    }


def _signed_distance(
    upper: pd.Series | Any, lower: pd.Series | Any, side: pd.Series | Any
) -> pd.Series:
    upper_series = pd.Series(upper, dtype=float)
    lower_series = pd.Series(lower, dtype=float)
    side_series = pd.Series(side, dtype=float).replace(0, 1.0)

    distance = (upper_series - lower_series) * side_series
    return distance


def _distance_stats(distance: pd.Series) -> Dict[str, float]:
    distance = pd.Series(distance, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if distance.empty:
        return {}

    return {
        "mean": float(distance.mean()),
        "median": float(distance.median()),
        "min": float(distance.min()),
        "max": float(distance.max()),
        "p25": float(distance.quantile(0.25)),
        "p75": float(distance.quantile(0.75)),
        "std": float(distance.std(ddof=1)) if len(distance) > 1 else 0.0,
        "count": int(len(distance)),
    }
