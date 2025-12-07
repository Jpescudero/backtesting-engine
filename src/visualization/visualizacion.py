# visualizacion.py
from __future__ import annotations

from pathlib import Path

import mplfinance as mpf

from data_utils import load_all_ticks, make_ohlcv


def plot_year_candles(
    parquet_root: str | Path,
    symbol: str,
    year: int,
    timeframe: str = "5min",
    max_bars: int | None = 5000,
) -> None:
    """
    Plotea un año concreto en velas OHLC a partir de los parquet intradiarios.

    - parquet_root: carpeta raíz donde guardaste los parquet (la misma que en main.py)
    - symbol: por ejemplo "NDXm"
    - year: por ejemplo 2021
    - timeframe: "1min", "5min", "15min", "1H", etc. (formato pandas)
    - max_bars: si hay demasiadas barras, recorta para no petar la memoria/gráfico
    """
    parquet_root = Path(parquet_root)

    # 1) Cargar ticks del año
    df_ticks = load_all_ticks(parquet_root, symbol=symbol, year=year)
    if df_ticks.empty:
        raise ValueError(f"No hay ticks para {symbol} en {year}")

    # 2) Agregar a OHLCV con el timeframe deseado
    ohlcv = make_ohlcv(
        df_ticks,
        timeframe=timeframe,
        price_col="mid",
        volume_col=None,          # si tienes volumen, pon aquí el nombre
        include_n_ticks=True,     # opcional: nº de ticks por barra
    )

    if ohlcv.empty:
        raise ValueError(f"No se han podido generar barras para {symbol} {year}")

    # Opcional: limitar número de barras para que el gráfico sea legible
    if max_bars is not None and len(ohlcv) > max_bars:
        # Nos quedamos con las últimas max_bars
        ohlcv = ohlcv.iloc[-max_bars:]

    # 3) Adaptar al formato que espera mplfinance
    #    Índice datetime, columnas: Open, High, Low, Close, (Volume opcional)
    df_plot = ohlcv[["open", "high", "low", "close"]].copy()
    df_plot.index.name = "Date"

    title = f"{symbol} {year} - velas {timeframe}"

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        title=title,
        ylabel="Precio",
        figsize=(16, 8),
        tight_layout=True,
    )
