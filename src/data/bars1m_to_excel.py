# src/data/bars1m_to_excel.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config.paths import (
    PARQUET_TICKS_DIR,
    OTHER_DATA_DIR,
    ensure_directories_exist,
)
from src.data.data_utils import iter_ticks_by_year, make_ohlcv

DEFAULT_SYMBOL = "NDXm"
DEFAULT_TIMEFRAME = "1min"


def get_default_output_csv(symbol: str = DEFAULT_SYMBOL, timeframe: str = DEFAULT_TIMEFRAME) -> Path:
    """
    Construye la ruta por defecto del CSV de barras 1m:

        data/other/barras_<timeframe>_<symbol>_todos_anios.csv
    """
    fname = f"barras_{timeframe}_{symbol}_todos_anios.csv"
    return OTHER_DATA_DIR / fname


def generate_1m_bars_csv(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    parquet_root: Path | None = None,
    output_csv: Path | None = None,
) -> Path:
    """
    A partir de los parquet de ticks, genera un CSV con todas las barras
    de 1 minuto (u otro timeframe) de todos los años del símbolo dado.

    Devuelve la ruta del CSV generado.
    """
    ensure_directories_exist()

    # Origen de los parquet de ticks
    parquet_root = Path(parquet_root) if parquet_root is not None else PARQUET_TICKS_DIR
    parquet_root = parquet_root.resolve()

    # Destino del CSV
    output_csv = Path(output_csv) if output_csv is not None else get_default_output_csv(
        symbol=symbol,
        timeframe=timeframe,
    )
    output_csv = output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_bars: list[pd.DataFrame] = []

    # Recorremos año a año usando el helper
    for year, df_ticks in iter_ticks_by_year(parquet_root, symbol=symbol):
        if df_ticks.empty:
            print(f"{year}: sin ticks, se salta.")
            continue

        print(f"{year}: generando barras de {timeframe}...")

        # Generar OHLC al timeframe deseado
        ohlcv = make_ohlcv(
            df_ticks,
            timeframe=timeframe,
            price_col="mid",      # ajusta si tu columna de precio se llama distinto
            volume_col=None,      # pon el nombre si tienes volumen en los ticks
            include_n_ticks=True  # opcional: nº de ticks por barra
        )

        if ohlcv.empty:
            print(f"{year}: no se han generado barras, se salta.")
            continue

        # Añadimos el año y el símbolo como columnas (útil luego para filtrar)
        ohlcv["year"] = year
        ohlcv["symbol"] = symbol

        all_bars.append(ohlcv)

        print(f"{year}: {len(ohlcv)} barras de {timeframe}.")

    if not all_bars:
        print("No se ha generado ninguna barra. Revisa rutas/símbolo.")
        return output_csv

    # Unir todos los años en un solo DataFrame
    df_all = pd.concat(all_bars).sort_index()

    # Guardar a CSV
    # index=True para que la fecha/hora quede como columna
    df_all.to_csv(output_csv, index=True)

    print(f"Guardado fichero CSV con {len(df_all)} barras en:\n{output_csv}")
    return output_csv


def main() -> None:
    generate_1m_bars_csv()


if __name__ == "__main__":
    # Uso autónomo:
    # python -m src.data.bars1m_to_excel
    main()
