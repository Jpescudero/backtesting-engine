# src/analytics/trade_plots.py
"""
Plots específicos para analizar trades individuales.

Incluye una función para dibujar las mejores y peores
operaciones (por PnL) sobre barras de 1 minuto, marcando
entrada y salida.

Mejoras:
- Eje X con timestamps legibles (si data.ts está disponible).
- Por defecto, dibuja 3 mejores + 3 peores trades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _extract_arrays(
    data,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
]:
    """
    Extrae arrays (o, h, l, c, v, ts) de la estructura `data`
    devuelta por NPZOHLCVFeed.

    Soporta:
      - dataclass/objeto con atributos .o, .h, .l, .c, .v, .ts
      - dict con claves 'o','h','l','c','v','ts'
      - dict con claves 'open','high','low','close','volume','timestamp'
    """
    ts = None

    # Objeto con atributos
    for keys in (("o", "h", "l", "c"), ("open", "high", "low", "close")):
        if all(hasattr(data, k) for k in keys):
            o = np.asarray(getattr(data, keys[0]))
            h = np.asarray(getattr(data, keys[1]))
            l = np.asarray(getattr(data, keys[2]))
            c = np.asarray(getattr(data, keys[3]))
            v = np.asarray(getattr(data, "v", np.zeros_like(o)))

            if hasattr(data, "ts"):
                ts = np.asarray(getattr(data, "ts"))
            elif hasattr(data, "timestamp"):
                ts = np.asarray(getattr(data, "timestamp"))
            elif hasattr(data, "time"):
                ts = np.asarray(getattr(data, "time"))

            return o, h, l, c, v, ts

    # Diccionario
    if isinstance(data, dict):
        for keys in (("o", "h", "l", "c"), ("open", "high", "low", "close")):
            if all(k in data for k in keys):
                o = np.asarray(data[keys[0]])
                h = np.asarray(data[keys[1]])
                l = np.asarray(data[keys[2]])
                c = np.asarray(data[keys[3]])
                v = np.asarray(data.get("v") or data.get("volume") or np.zeros_like(o))

                for ts_key in ("ts", "timestamp", "time"):
                    if ts_key in data:
                        ts = np.asarray(data[ts_key])
                        break

                return o, h, l, c, v, ts

    raise TypeError("No se han podido extraer arrays OHLC de `data`.")


def _build_timestamps(ts_array: Optional[np.ndarray], n: int) -> Optional[pd.DatetimeIndex]:
    """
    Intenta construir un DatetimeIndex legible a partir de ts_array.
    Si no se puede, devuelve None y se usará el índice de barras.
    """
    if ts_array is None:
        return None

    ts_array = np.asarray(ts_array)
    try:
        # Si ya es datetime64, esto funciona directo
        ts = pd.to_datetime(ts_array)
    except Exception:
        # Como fallback, probamos con unidad "ns" (típico epoch_ns)
        try:
            ts = pd.to_datetime(ts_array, unit="ns", origin="unix")
        except Exception:
            return None

    if len(ts) != n:
        # Inconsistencia de longitud; mejor no usarlo
        return None

    return pd.DatetimeIndex(ts)


def select_best_and_worst_trades(
    trades_df: pd.DataFrame,
    n_best: int = 3,
    n_worst: int = 3,
    pnl_col: str = "pnl",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve dos DataFrames:
      - best_trades: top n_best por PnL descendente
      - worst_trades: top n_worst por PnL ascendente
    """
    if pnl_col not in trades_df.columns:
        raise KeyError(f"No se encuentra la columna de PnL '{pnl_col}' en trades_df.columns.")

    sorted_trades = trades_df.sort_values(pnl_col, ascending=False)
    best = sorted_trades.head(n_best)
    worst = sorted_trades.tail(n_worst).sort_values(pnl_col, ascending=True)
    return best, worst


def _plot_single_trade_bars(
    ax: plt.Axes,
    *,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    timestamps: Optional[pd.DatetimeIndex],
    window: int = 30,
    title: str = "",
    direction: Optional[int] = None,
) -> None:
    """
    Dibuja barras OHLC simplificadas alrededor de un trade concreto.

    Si `timestamps` no es None, el eje X usa fechas/horas legibles.
    Si es None, usa el índice de barras (0, 1, 2, ...).
    """
    n = len(c)
    entry_idx = int(entry_idx)
    exit_idx = int(exit_idx)

    left = max(0, min(entry_idx, exit_idx) - window)
    right = min(n - 1, max(entry_idx, exit_idx) + window)

    idx_range = np.arange(left, right + 1, dtype=int)

    if timestamps is not None:
        x_vals = timestamps[idx_range]
        x_entry = timestamps[entry_idx]
        x_exit = timestamps[exit_idx]
    else:
        x_vals = idx_range
        x_entry = entry_idx
        x_exit = exit_idx

    # Dibujamos barras tipo "OHLC" simplificadas
    for pos, i in enumerate(idx_range):
        x = x_vals[pos]
        color = "tab:green" if c[i] >= o[i] else "tab:red"
        # línea vertical low-high
        ax.vlines(x, l[i], h[i], linewidth=1, alpha=0.8, color=color)
        # ticks horizontales en open/close (ligeramente desplazados en X)
        if timestamps is not None:
            # pequeño desplazamiento en tiempo
            delta = (timestamps[1] - timestamps[0]) if len(timestamps) > 1 else pd.Timedelta(minutes=1)
            ax.hlines(o[i], x - 0.2 * delta, x, linewidth=1.5, color=color)
            ax.hlines(c[i], x, x + 0.2 * delta, linewidth=1.5, color=color)
        else:
            ax.hlines(o[i], x - 0.2, x, linewidth=1.5, color=color)
            ax.hlines(c[i], x, x + 0.2, linewidth=1.5, color=color)

    # Señales de entrada/salida
    entry_price = c[entry_idx]
    exit_price = c[exit_idx]

    ax.axvline(x_entry, linestyle="--", linewidth=1.0, color="blue", alpha=0.9)
    ax.axvline(x_exit, linestyle="--", linewidth=1.0, color="black", alpha=0.9)

    ax.scatter([x_entry], [entry_price], marker="^", s=40, color="blue", zorder=5)
    ax.scatter([x_exit], [exit_price], marker="v", s=40, color="black", zorder=5)

    if direction is not None:
        if direction > 0:
            dir_txt = "LONG"
        elif direction < 0:
            dir_txt = "SHORT"
        else:
            dir_txt = "FLAT"
        ax.text(
            0.01,
            0.97,
            dir_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
        )

    ax.set_title(title, fontsize=9)

    if timestamps is not None:
        ax.set_xlabel("Hora")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.tick_params(axis="x", rotation=0)
    else:
        ax.set_xlabel("Barra (1m)")

    ax.grid(True, alpha=0.2)


def plot_best_and_worst_trades(
    *,
    trades_df: pd.DataFrame,
    data,
    n_best: int = 3,
    n_worst: int = 3,
    pnl_col: str = "pnl",
    entry_col: str = "entry_idx",
    exit_col: str = "exit_idx",
    direction_col: str = "direction",
    window: int = 30,
    figsize: Tuple[float, float] = (14.0, 10.0),
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Dibuja las mejores y peores operaciones en una cuadrícula de 3 x 2 (por defecto).

    Parámetros clave:
        trades_df: resultado de trades_to_dataframe(...)
        data: objeto devuelto por NPZOHLCVFeed.load_all()
        n_best, n_worst: límite de trades buenos/malos a mostrar
        pnl_col: columna con el PnL de cada trade
        entry_col/exit_col: columnas con los índices de barra de entrada/salida
        direction_col: opcional, si existe indica LONG/SHORT con +1/-1
        window: nº de barras antes y después del trade a dibujar
        figsize: tamaño total de la figura
        save_path: si se indica, se guarda la figura en esa ruta (PNG)
    """
    if entry_col not in trades_df.columns or exit_col not in trades_df.columns:
        raise KeyError(
            f"Se requieren las columnas '{entry_col}' y '{exit_col}' en trades_df para poder dibujar los trades."
        )

    if len(trades_df) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No hay trades para representar", ha="center", va="center")
        ax.axis("off")
        if save_path is not None:
            save_path = Path(save_path).resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    o, h, l, c, _, ts = _extract_arrays(data)
    timestamps = _build_timestamps(ts, len(c))

    best_trades, worst_trades = select_best_and_worst_trades(
        trades_df=trades_df,
        n_best=n_best,
        n_worst=n_worst,
        pnl_col=pnl_col,
    )

    n_best_eff = min(len(best_trades), n_best)
    n_worst_eff = min(len(worst_trades), n_worst)
    n_rows = max(n_best_eff, n_worst_eff, 1)
    n_cols = 2  # izquierda: mejores, derecha: peores

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False, sharey=False)
    if n_rows == 1:
        axes = np.array([axes])  # shape (1, 2)

    axes = np.asarray(axes)

    # Mejores trades
    for row_idx in range(n_rows):
        ax = axes[row_idx, 0]
        if row_idx >= n_best_eff:
            ax.axis("off")
            continue

        row = best_trades.iloc[row_idx]
        entry_idx = int(row[entry_col])
        exit_idx = int(row[exit_col])
        direction = None
        if direction_col in trades_df.columns:
            try:
                direction = int(row[direction_col])
            except Exception:
                direction = None

        title = f"BEST #{row_idx+1} | trade_id={row.get('trade_id', row.name)} | PnL={row[pnl_col]:.2f}"
        _plot_single_trade_bars(
            ax,
            o=o,
            h=h,
            l=l,
            c=c,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            timestamps=timestamps,
            window=window,
            title=title,
            direction=direction,
        )

    # Peores trades
    for row_idx in range(n_rows):
        ax = axes[row_idx, 1]
        if row_idx >= n_worst_eff:
            ax.axis("off")
            continue

        row = worst_trades.iloc[row_idx]
        entry_idx = int(row[entry_col])
        exit_idx = int(row[exit_col])
        direction = None
        if direction_col in trades_df.columns:
            try:
                direction = int(row[direction_col])
            except Exception:
                direction = None

        title = f"WORST #{row_idx+1} | trade_id={row.get('trade_id', row.name)} | PnL={row[pnl_col]:.2f}"
        _plot_single_trade_bars(
            ax,
            o=o,
            h=h,
            l=l,
            c=c,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            timestamps=timestamps,
            window=window,
            title=title,
            direction=direction,
        )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
