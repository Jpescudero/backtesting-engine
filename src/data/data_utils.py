from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

# =========================
# CARGA DE TICKS DESDE PARQUETS
# =========================


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def list_tick_files(
    parquet_root: str | Path,
    symbol: str | None = None,
    year: int | None = None,
    pattern: str = "*.parquet",
) -> list[Path]:
    """
    Devuelve la lista de ficheros parquet que encajan con (symbol, year).

    Asume la estructura generada en data_to_parquet.py:

        <parquet_root>/<symbol>/<year>/<symbol>_<YYYY-MM-DD>_<HH>.parquet
    """
    root = _as_path(parquet_root)

    files: list[Path] = []

    if symbol is not None and year is not None:
        year_dir = root / symbol / str(year)
        if year_dir.is_dir():
            files.extend(sorted(year_dir.glob(pattern)))
        return files

    if year is not None and symbol is None:
        # Buscar en todos los símbolos
        for sym_dir in root.iterdir():
            if not sym_dir.is_dir():
                continue
            year_dir = sym_dir / str(year)
            if year_dir.is_dir():
                files.extend(sorted(year_dir.glob(pattern)))
        return files

    if symbol is not None and year is None:
        for year_dir in (root / symbol).iterdir():
            if year_dir.is_dir():
                files.extend(sorted(year_dir.glob(pattern)))
        return sorted(files)

    # Sin filtros: todos los parquet
    for sym_dir in root.iterdir():
        if not sym_dir.is_dir():
            continue
        for year_dir in sym_dir.iterdir():
            if not year_dir.is_dir():
                continue
            files.extend(sorted(year_dir.glob(pattern)))

    return sorted(files)


def load_all_ticks(
    parquet_root: str | Path,
    symbol: str | None = None,
    year: int | None = None,
    pattern: str = "*.parquet",
) -> pd.DataFrame:
    """
    Carga todos los ticks que encajen con symbol/year en un único DataFrame.

    Úsalo sólo si sabes que el volumen de datos entra en memoria.
    """
    files = list_tick_files(parquet_root, symbol=symbol, year=year, pattern=pattern)
    if not files:
        raise FileNotFoundError(
            f"No se han encontrado parquet en {parquet_root} " f"para symbol={symbol!r}, year={year!r}"
        )

    frames = [pd.read_parquet(p) for p in files]
    df = pd.concat(frames).sort_index()
    return df


def iter_ticks_by_year(
    parquet_root: str | Path,
    symbol: str | None = None,
    pattern: str = "*.parquet",
) -> Iterator[tuple[int, pd.DataFrame]]:
    """
    Generador que va devolviendo (year, df_ticks) año a año.

    Esto te permite hacer backtesting de forma incremental sin
    cargar todos los años en memoria a la vez.
    """
    root = _as_path(parquet_root)

    years: set[int] = set()

    if symbol is not None:
        sym_dir = root / symbol
        if not sym_dir.is_dir():
            raise FileNotFoundError(f"No existe el directorio {sym_dir}")
        for year_dir in sym_dir.iterdir():
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.add(int(year_dir.name))
    else:
        for sym_dir in root.iterdir():
            if not sym_dir.is_dir():
                continue
            for year_dir in sym_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    years.add(int(year_dir.name))

    for y in sorted(years):
        df = load_all_ticks(root, symbol=symbol, year=y, pattern=pattern)
        yield y, df


# =========================
# AGREGACIÓN A OHLCV
# =========================


def make_ohlcv(
    df_ticks: pd.DataFrame,
    timeframe: str = "1min",
    price_col: str = "mid",
    volume_col: str | None = None,
    group_by: str | Sequence[str] | None = None,
    include_n_ticks: bool = False,
) -> pd.DataFrame:
    """
    Crea barras OHLCV a partir de un DataFrame de ticks.

    df_ticks: DataFrame con índice datetime (idealmente UTC).
    timeframe: string de resample ("1min", "5min", "1H"...).
    price_col: columna de precio (por defecto 'mid').
    volume_col: columna de volumen (si None, no agrega volumen).
    group_by: columna(s) para agrupar por símbolo, etc.
    include_n_ticks: añade 'n_ticks' con el nº de ticks por barra.
    """
    if not isinstance(df_ticks.index, pd.DatetimeIndex):
        raise TypeError("df_ticks debe tener un DatetimeIndex")

    df = df_ticks.sort_index()

    if group_by is None:
        resampled = df.resample(timeframe)
        return _make_ohlcv_from_resampler(resampled, price_col, volume_col, include_n_ticks)

    # Con agrupación (por ejemplo, distintos símbolos)
    if isinstance(group_by, str):
        group_by_cols = [group_by]
    else:
        group_by_cols = list(group_by)

    results = []
    for key, sub in df.groupby(group_by_cols):
        resampled = sub.resample(timeframe)
        ohlcv = _make_ohlcv_from_resampler(resampled, price_col, volume_col, include_n_ticks)
        # Añadimos las columnas de grupo como columnas normales
        if not isinstance(key, tuple):
            key = (key,)
        for col, val in zip(group_by_cols, key):
            ohlcv[col] = val
        ohlcv = ohlcv.set_index(group_by_cols, append=True)
        results.append(ohlcv)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results).sort_index()
    # Reordenamos niveles de índice para que queden (group_by..., ts)
    idx_names = list(out.index.names)
    ts_name = idx_names[0]
    other_names = idx_names[1:]
    out = out.reorder_levels(other_names + [ts_name]).sort_index()
    return out


def _make_ohlcv_from_resampler(
    resampled: pd.core.resample.DatetimeIndexResampler,
    price_col: str,
    volume_col: str | None,
    include_n_ticks: bool,
) -> pd.DataFrame:
    agg_dict: dict[str, tuple[str, str]] = {
        "open": (price_col, "first"),
        "high": (price_col, "max"),
        "low": (price_col, "min"),
        "close": (price_col, "last"),
    }

    if volume_col is not None and volume_col in resampled.obj.columns:
        agg_dict["volume"] = (volume_col, "sum")

    if include_n_ticks:
        agg_dict["n_ticks"] = (price_col, "size")

    ohlcv = resampled.agg(**agg_dict)

    if isinstance(ohlcv.columns, pd.MultiIndex):
        ohlcv.columns = [c[-1] for c in ohlcv.columns]

    return ohlcv
