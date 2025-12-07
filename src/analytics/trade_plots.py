# src/analytics/trade_plots.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    # mplfinance es ligero y cómodo para velas
    from mplfinance.original_flavor import candlestick_ohlc
except ImportError:  # pragma: no cover
    candlestick_ohlc = None

from src.data.feeds import NPZOHLCVFeed, OHLCVArrays


@dataclass
class BestWorstConfig:
    top_n: int = 3
    window_bars_before: int = 30
    window_bars_after: int = 60
    timeframe: str = "1m"
    symbol: str = "NDXm"
    figsize: Tuple[int, int] = (16, 10)


def _ts_to_datetime(ts: np.ndarray) -> np.ndarray:
    """Convierte array de timestamps a datetime64[ns] de forma robusta."""
    if np.issubdtype(ts.dtype, np.datetime64):
        return ts.astype("datetime64[ns]")

    ts0 = int(ts[0])
    if ts0 > 10 ** 14:
        unit = "ns"
    elif ts0 > 10 ** 11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(ts, unit=unit).values.astype("datetime64[ns]")


def _build_window_indices(
    entry_idx: int,
    exit_idx: int,
    n_bars: int,
    before: int,
    after: int,
) -> slice:
    start = max(0, entry_idx - before)
    end = min(n_bars - 1, exit_idx + after)
    return slice(start, end + 1)


def _maybe_get_level(row: pd.Series, candidates: Iterable[str]) -> Optional[float]:
    """Devuelve el primer nivel disponible en la fila entre varios posibles nombres."""
    for name in candidates:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                continue
    return None


def _plot_single_trade(
    ax: plt.Axes,
    trade: pd.Series,
    data: OHLCVArrays,
    cfg: BestWorstConfig,
    side_label: str,
    rank: int,
) -> None:
    """Dibuja una única operación (velas + entrada/salida + SL/TP)."""
    n_bars = data.c.shape[0]
    entry_idx = int(trade["entry_idx"])
    exit_idx = int(trade["exit_idx"])

    window = _build_window_indices(
        entry_idx=entry_idx,
        exit_idx=exit_idx,
        n_bars=n_bars,
        before=cfg.window_bars_before,
        after=cfg.window_bars_after,
    )

    ts_dt = _ts_to_datetime(data.ts[window])
    o = data.o[window]
    h = data.h[window]
    l = data.l[window]
    c = data.c[window]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts_dt),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
        }
    )
    # Evitar FutureWarning: convertimos la serie a numpy datetime64 directamente
    df["mdates"] = mdates.date2num(df["timestamp"].to_numpy())

    if candlestick_ohlc is None:
        # fallback sencillo: pseudo-velas
        for _, row_b in df.iterrows():
            ax.vlines(row_b["mdates"], row_b["low"], row_b["high"], linewidth=0.6)
            color = "green" if row_b["close"] >= row_b["open"] else "red"
            ax.vlines(
                row_b["mdates"],
                row_b["open"],
                row_b["close"],
                linewidth=3,
                color=color,
            )
    else:
        ohlc = df[["mdates", "open", "high", "low", "close"]].values
        candlestick_ohlc(
            ax,
            ohlc,
            width=0.0008,
            colorup="green",
            colordown="red",
            alpha=0.8,
        )

    # Marcar entrada y salida
    entry_time = pd.to_datetime(trade["entry_time"])
    exit_time = pd.to_datetime(trade["exit_time"])
    entry_num = mdates.date2num(entry_time.to_pydatetime())
    exit_num = mdates.date2num(exit_time.to_pydatetime())

    entry_price = float(trade["entry_price"])
    exit_price = float(trade["exit_price"])

    ax.scatter(entry_num, entry_price, marker="^", color="blue", s=50, zorder=5)
    ax.scatter(exit_num, exit_price, marker="v", color="black", s=50, zorder=5)

    # SL / TP: primero intentamos leer columnas (por si en el futuro las añades)
    sl = _maybe_get_level(
        trade,
        ["stop_loss", "sl_price", "sl", "stop_loss_price"],
    )
    tp = _maybe_get_level(
        trade,
        ["take_profit", "tp_price", "tp", "take_profit_price"],
    )

    # Si no vienen en el DataFrame, los calculamos a partir de % y side
    # (replica BacktestConfig del main.py: sl_pct=0.01, tp_pct=0.02)
    sl_pct = float(trade.get("sl_pct", 0.01))
    tp_pct = float(trade.get("tp_pct", 0.02))
    side = float(trade.get("side", 1.0))  # StrategyBarridaApertura es long-only (=1)

    if sl is None:
        if side >= 0:
            sl = entry_price * (1.0 - sl_pct)
        else:
            sl = entry_price * (1.0 + sl_pct)

    if tp is None:
        if side >= 0:
            tp = entry_price * (1.0 + tp_pct)
        else:
            tp = entry_price * (1.0 - tp_pct)

    # Rango completo del gráfico para que las líneas sean "como órdenes" en MT
    xmin = df["mdates"].min()
    xmax = df["mdates"].max()

    if sl is not None:
        ax.hlines(
            sl,
            xmin=xmin,
            xmax=xmax,
            colors="red",
            linestyles="--",
            linewidth=1.2,
            label="SL",
        )
    if tp is not None:
        ax.hlines(
            tp,
            xmin=xmin,
            xmax=xmax,
            colors="green",
            linestyles="--",
            linewidth=1.2,
            label="TP",
        )

    # Eje X sólo hora
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", rotation=45)

    # Título con día
    trade_day = entry_time.strftime("%Y-%m-%d")
    trade_id = int(trade["trade_id"]) if "trade_id" in trade else int(trade.name)
    pnl = float(trade["pnl"])

    title = f"{side_label} #{rank} | {trade_day} | trade_id={trade_id} | PnL={pnl:.2f}"
    ax.set_title(title, fontsize=9)

    ax.set_ylabel("Precio")


def plot_best_and_worst_trades(
    trades_df: pd.DataFrame,
    data: OHLCVArrays,
    n_best: int = 3,
    cfg: Optional[BestWorstConfig] = None,
    save_path: Optional[Path] = None,
    **kwargs,
) -> plt.Figure:
    """Genera figura con las N mejores y N peores operaciones.

    Compatible con la llamada de main.py:
        plot_best_and_worst_trades(trades_df=..., data=..., n_best=..., ...)
    """
    trades = trades_df

    if trades.empty:
        raise ValueError("No hay trades para plotear mejores/peores.")

    n_rows = min(n_best, len(trades))

    if cfg is None:
        cfg = BestWorstConfig(top_n=n_rows)
    else:
        cfg.top_n = n_rows

    required_cols = {
        "entry_idx",
        "exit_idx",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "pnl",
    }
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"Faltan columnas en trades para poder plotear: {missing}")

    best = trades.nlargest(cfg.top_n, "pnl")
    worst = trades.nsmallest(cfg.top_n, "pnl")

    fig, axes = plt.subplots(
        nrows=cfg.top_n,
        ncols=2,
        figsize=cfg.figsize,
        sharex=False,
    )

    if cfg.top_n == 1:
        axes = np.array([[axes[0], axes[1]]])  # type: ignore[index]

    for i in range(cfg.top_n):
        _plot_single_trade(
            ax=axes[i, 0],
            trade=best.iloc[i],
            data=data,
            cfg=cfg,
            side_label="BEST",
            rank=i + 1,
        )
        _plot_single_trade(
            ax=axes[i, 1],
            trade=worst.iloc[i],
            data=data,
            cfg=cfg,
            side_label="WORST",
            rank=i + 1,
        )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[trade_plots] Figura guardada en: {save_path}")

    return fig


def load_trades_from_excel(excel_path: Path | str) -> pd.DataFrame:
    """Helper para leer la hoja 'trades' del Excel de backtest."""
    excel_path = Path(excel_path)
    df = pd.read_excel(excel_path, sheet_name="trades")
    if "entry_time" in df:
        df["entry_time"] = pd.to_datetime(df["entry_time"])
    if "exit_time" in df:
        df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df


def plot_best_and_worst_from_files(
    excel_path: Path | str,
    symbol: str,
    timeframe: str = "1m",
    cfg: Optional[BestWorstConfig] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Convenience: carga datos NPZ + Excel y genera el plot."""
    if cfg is None:
        cfg = BestWorstConfig(symbol=symbol, timeframe=timeframe)

    trades = load_trades_from_excel(excel_path)
    feed = NPZOHLCVFeed(symbol=symbol, timeframe=timeframe)
    data = feed.load_all()

    return plot_best_and_worst_trades(
        trades_df=trades,
        data=data,
        n_best=cfg.top_n,
        cfg=cfg,
        save_path=save_path,
    )


if __name__ == "__main__":
    # Ejemplo rápido (ajusta rutas a tu entorno)
    default_excel = Path("data/backtests/backtest_NDXm_barrida_apertura.xlsx")
    out_png = default_excel.with_name("best_worst_trades.png")

    if default_excel.exists():
        plot_best_and_worst_from_files(
            excel_path=default_excel,
            symbol="NDXm",
            timeframe="1m",
            save_path=out_png,
        )
        plt.show()
    else:
        print(f"[trade_plots] No se encontró el Excel por defecto: {default_excel}")
