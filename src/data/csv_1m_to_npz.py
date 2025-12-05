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
) -> Path:
    """
    Convierte un CSV de barras 1m en un .npz numba-friendly (ts, o, h, l, c, v).

    - Si existe columna 'ts', la usa directamente como tiempo.
    - Si no, intenta usar alguna de las columnas en datetime_col_candidates.
    - El nombre final será: <symbol>_all_1m.npz en data/npz/<symbol>/
    """
    ensure_directories_exist()

    # 1) Localizar CSV
    if csv_path is None:
        csv_path = DATA_DIR / "other" / f"barras_1min_{symbol}_todos_anios.csv"

    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")

    print(f"[csv_1m_to_npz] Leyendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # 2) Asegurar índice datetime
    # Caso especial: si ya existe columna 'ts', la usamos primero.
    if "ts" in df.columns:
        if tz_aware:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        else:
            df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    else:
        # Fallback: buscar otra columna candidata
        time_col = None
        for col in datetime_col_candidates:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError(
                f"No se encontró columna de tiempo en {datetime_col_candidates}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        if tz_aware:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
        else:
            df[time_col] = pd.to_datetime(df[time_col])

        df = df.set_index(time_col)

    df = df.sort_index()

    # 3) Convertir a arrays OHLCV
    arrays = bars_df_to_npz_arrays(df)

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
