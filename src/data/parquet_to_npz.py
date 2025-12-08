# src/data/parquet_to_npz.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config.paths import (
    ensure_directories_exist,
    NPZ_DIR,
    PARQUET_BARS_1M_DIR,
)


def bars_df_to_npz_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Convierte un DataFrame de barras OHLCV en arrays NumPy listos para Numba.

    Requisitos:
      - Índice de tipo datetime.
      - Columnas: 'open', 'high', 'low', 'close'.
      - 'volume' opcional (si no está, se crea como 1.0).

    Además:
      - Elimina filas con NaN en open/high/low/close.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame de barras debe tener un DatetimeIndex.")

    # Aseguramos orden temporal
    df = df.sort_index()

    # Comprobamos columnas básicas
    for col in ("open", "high", "low", "close"):
        if col not in df:
            raise ValueError(f"El DataFrame debe contener la columna '{col}'.")

    # Eliminamos cualquier barra con OHLC incompleto
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"[bars_df_to_npz_arrays] Eliminadas {dropped} filas con OHLC NaN")

    # Volumen: si no hay, creamos; si hay NaN, los rellenamos con 0.0
    if "volume" not in df:
        df["volume"] = 1.0
    else:
        df["volume"] = df["volume"].fillna(0.0)

    # Índice datetime -> int64 (nanosegundos desde epoch)
    ts = df.index.astype("int64").values

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    return {"ts": ts, "o": o, "h": h, "l": low, "c": c, "v": v}


def convert_parquet_file_to_npz(
    parquet_path: Path,
    npz_path: Path,
    columns: Optional[Sequence[str]] = None,
) -> None:
    """
    Lee un archivo Parquet de barras y guarda un .npz con arrays OHLCV.

    - parquet_path: ruta al archivo Parquet de entrada.
    - npz_path: ruta de salida para el archivo .npz.
    - columns: columnas a leer; si None, se leerán todas.
    """
    parquet_path = parquet_path.resolve()
    npz_path = npz_path.resolve()

    npz_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[parquet_to_npz] Leyendo Parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, columns=columns)

    if not isinstance(df.index, pd.DatetimeIndex):
        # Si el índice no es datetime, intentamos convertir una columna 'timestamp' o similar.
        for candidate in ("timestamp", "datetime", "ts"):
            if candidate in df.columns:
                df[candidate] = pd.to_datetime(df[candidate], utc=True)
                df = df.set_index(candidate)
                break
        else:
            raise ValueError(f"No se encontró índice datetime ni columna de tiempo estándar en {parquet_path}")

    arrays = bars_df_to_npz_arrays(df)

    print(f"[parquet_to_npz] Guardando NPZ: {npz_path}")
    np.savez_compressed(npz_path, **arrays)


def convert_all_parquet_bars_to_npz(
    symbol: str,
    parquet_dir: Optional[Path] = None,
    npz_base_dir: Optional[Path] = None,
    suffix: str = "",
) -> None:
    """
    Convierte todos los archivos Parquet de barras de un símbolo concreto
    a ficheros .npz con el mismo 'stem'.

    Asume estructura:
        PARQUET_BARS_1M_DIR / <symbol> / *.parquet

    Los .npz irán a:
        NPZ_DIR / <symbol> / <stem><suffix>.npz

    - symbol: por ejemplo 'NDXm'.
    - suffix: opcional, por ejemplo '_1m' para indicar el timeframe.
    """
    ensure_directories_exist()

    if parquet_dir is None:
        parquet_dir = PARQUET_BARS_1M_DIR / symbol

    if npz_base_dir is None:
        npz_base_dir = NPZ_DIR / symbol

    parquet_dir = parquet_dir.resolve()
    npz_base_dir = npz_base_dir.resolve()
    npz_base_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[parquet_to_npz] No se encontraron Parquet en {parquet_dir}")
        return

    print(f"[parquet_to_npz] Encontrados {len(parquet_files)} archivos Parquet para {symbol}")

    for pq in parquet_files:
        stem = pq.stem  # p.ej. 'NDXm_2021_1m'
        npz_path = npz_base_dir / f"{stem}{suffix}.npz"
        convert_parquet_file_to_npz(pq, npz_path)


if __name__ == "__main__":
    # Ejemplo de uso directo:
    # python -m src.data.parquet_to_npz
    SYMBOL = "NDXm"
    convert_all_parquet_bars_to_npz(SYMBOL, suffix="_1m")
