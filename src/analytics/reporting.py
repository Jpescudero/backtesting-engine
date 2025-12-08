# src/analytics/reporting.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.core import BacktestResult
from src.data.feeds import OHLCVArrays


def equity_to_series(
    result: BacktestResult,
    data: OHLCVArrays,
) -> pd.Series:
    """
    Convierte la equity (np.ndarray) en una serie de pandas indexada por timestamp.
    """
    ts = pd.to_datetime(data.ts, unit="ns", utc=True)

    # Hay feeds con timestamps duplicados (p. ej. al concatenar ficheros), lo que
    # provoca trazos verticales en la curva de equity. Nos quedamos con el valor
    # más reciente de cada timestamp y aseguramos orden creciente.
    equity_series = pd.Series(result.equity, index=ts, name="equity")
    equity_series = equity_series.groupby(level=0).last().sort_index()

    return equity_series


def trades_to_dataframe(
    result: BacktestResult,
    data: OHLCVArrays,
) -> pd.DataFrame:
    """
    Convierte el log de trades (arrays NumPy) en un DataFrame amigable.
    """
    log = result.trade_log
    if not log:
        # Sin trades
        return pd.DataFrame(
            columns=[
                "entry_time", "exit_time", "entry_idx", "exit_idx",
                "entry_price", "exit_price", "qty", "pnl",
                "holding_bars", "exit_reason_code", "exit_reason"
            ]
        )

    ts = pd.to_datetime(data.ts, unit="ns", utc=True)

    entry_idx = log["entry_idx"]
    exit_idx = log["exit_idx"]

    entry_time = ts[entry_idx]
    exit_time = ts[exit_idx]

    df = pd.DataFrame({
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "entry_price": log["entry_price"],
        "exit_price": log["exit_price"],
        "qty": log["qty"],
        "pnl": log["pnl"],
        "holding_bars": log["holding_bars"],
        "exit_reason_code": log["exit_reason"],
    })

    stop_losses = result.extra.get("stop_losses") if result.extra else None
    take_profits = result.extra.get("take_profits") if result.extra else None
    sl_pct = float(result.extra.get("sl_pct", np.nan)) if result.extra else np.nan
    tp_pct = float(result.extra.get("tp_pct", np.nan)) if result.extra else np.nan

    if isinstance(stop_losses, np.ndarray) and stop_losses.shape[0] == data.c.shape[0]:
        df["stop_loss"] = stop_losses[entry_idx]
    else:
        df["stop_loss"] = np.nan

    if isinstance(take_profits, np.ndarray) and take_profits.shape[0] == data.c.shape[0]:
        df["take_profit"] = take_profits[entry_idx]
    else:
        df["take_profit"] = np.nan

    # Si no hay SL/TP explícitos, derivamos los niveles desde el % configurado
    side = np.sign(df["qty"]).replace(0, 1.0)
    if df["stop_loss"].isna().any() and np.isfinite(sl_pct):
        df.loc[df["stop_loss"].isna(), "stop_loss"] = df.loc[
            df["stop_loss"].isna(), "entry_price"
        ] * (1.0 - sl_pct * side[df["stop_loss"].isna()])

    if df["take_profit"].isna().any() and np.isfinite(tp_pct):
        df.loc[df["take_profit"].isna(), "take_profit"] = df.loc[
            df["take_profit"].isna(), "entry_price"
        ] * (1.0 + tp_pct * side[df["take_profit"].isna()])

    df["sl_pct"] = sl_pct
    df["tp_pct"] = tp_pct

    # Mapear códigos a texto
    reason_map = {
        1: "stop_loss",
        2: "take_profit",
        3: "time_stop",
        4: "signal_exit",
    }
    df["exit_reason"] = df["exit_reason_code"].map(reason_map).fillna("unknown")

    return df.sort_values("entry_time")
