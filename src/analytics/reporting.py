# src/analytics/reporting.py

from __future__ import annotations

from typing import Optional

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
    return pd.Series(result.equity, index=ts, name="equity")


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

    # Mapear códigos a texto
    reason_map = {
        1: "stop_loss",
        2: "take_profit",
        3: "time_stop",
        4: "signal_exit",
    }
    df["exit_reason"] = df["exit_reason_code"].map(reason_map).fillna("unknown")

    return df.sort_values("entry_time")
