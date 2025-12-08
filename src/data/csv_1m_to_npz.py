# src/data/csv_1m_to_npz.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config.paths import DATA_DIR, NPZ_DIR, ensure_directories_exist
from src.data.parquet_to_npz import bars_df_to_npz_arrays


def csv_1m_to_npz(
    symbol: str = "NDXm",
    csv_path: Optional[Path] = None,
    datetime_col_candidates: Sequence[str] = ("timestamp", "datetime", "time", "fecha", "Date", "ts"),
    tz_aware: bool = True,
    chunk_size: int = 250_000,
) -> Path:
    """
    Convierte un CSV de barras 1m en un .npz numba-friendly (ts, o, h, l, c, v).

    - Si existe columna 'ts', la usa directamente como tiempo.
    - Si no, intenta usar alguna de las columnas en datetime_col_candidates.
    - El nombre final será: <symbol>_all_1m.npz en data/npz/<symbol>/
    - El procesamiento se hace en chunks para reducir consumo de memoria.
      Usa chunk_size para ajustar el tamaño del bloque leido de pandas.read_csv.
    """
    ensure_directories_exist()

    # 1) Localizar CSV
    if csv_path is None:
        csv_path = DATA_DIR / "other" / f"barras_1min_{symbol}_todos_anios.csv"

    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")

    print(f"[csv_1m_to_npz] Leyendo CSV en chunks de {chunk_size} filas: {csv_path}")

    def _detect_time_col(columns: list[str]) -> str:
        if "ts" in columns:
            return "ts"
        for col in datetime_col_candidates:
            if col in columns:
                return col
        raise ValueError(
            f"No se encontró columna de tiempo en {datetime_col_candidates}. Columnas disponibles: {columns}"
        )

    arrays_per_chunk: list[dict[str, np.ndarray]] = []
    time_col: str | None = None

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if time_col is None:
            time_col = _detect_time_col(list(chunk.columns))

        if tz_aware:
            chunk[time_col] = pd.to_datetime(chunk[time_col], utc=True)
        else:
            chunk[time_col] = pd.to_datetime(chunk[time_col])

        parsed_chunk = chunk.set_index(time_col).sort_index()
        arrays_per_chunk.append(bars_df_to_npz_arrays(parsed_chunk))

    if not arrays_per_chunk:
        raise ValueError("El CSV no contiene datos para convertir a NPZ")

    ts = np.concatenate([arr["ts"] for arr in arrays_per_chunk])
    order = np.argsort(ts, kind="mergesort")

    def _concat_key(key: str) -> np.ndarray:
        values = np.concatenate([arr[key] for arr in arrays_per_chunk])
        # Reordenamos usando la misma vista para minimizar copias
        return values[order]

    arrays = {
        "ts": ts[order],
        "o": _concat_key("o"),
        "h": _concat_key("h"),
        "l": _concat_key("l"),
        "c": _concat_key("c"),
        "v": _concat_key("v"),
    }

    # 4) Guardar NPZ
    npz_dir = (NPZ_DIR / symbol).resolve()
    npz_dir.mkdir(parents=True, exist_ok=True)

    npz_path = npz_dir / f"{symbol}_all_1m.npz"
    print(f"[csv_1m_to_npz] Guardando NPZ: {npz_path}")
    np.savez_compressed(npz_path, **arrays)

    return npz_path


if __name__ == "__main__":
    # python -m src.data.csv_1m_to_npz
    csv_1m_to_npz()
