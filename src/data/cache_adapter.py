from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from src.config.paths import DATA_DIR, PARQUET_TICKS_DIR
from src.data.data_utils import list_tick_files


def _ensure_datetime(value: pd.Timestamp | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


class LRUDataFrameCache:
    """Pequeña caché LRU en memoria para DataFrames.

    Se usa para acelerar recargas recientes sin aumentar el uso de RAM
    de forma descontrolada.
    """

    def __init__(self, max_size: int = 4) -> None:
        self.max_size = max_size
        self._store: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, key: str) -> pd.DataFrame | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: pd.DataFrame) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)


@dataclass
class MarketDataAdapter:
    """Adaptador de datos que carga tramos normalizados con caché.

    La API principal es ``load(symbol, start, end, fields, granularity)``
    y devuelve un ``pd.DataFrame`` con tipos estrictos.
    """

    parquet_root: Path = PARQUET_TICKS_DIR
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")
    memory_cache_size: int = 4
    enable_disk_cache: bool = True

    def __post_init__(self) -> None:
        self.parquet_root = self.parquet_root.resolve()
        self.cache_dir = self.cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = LRUDataFrameCache(self.memory_cache_size)

    def load(
        self,
        symbol: str,
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        fields: Iterable[str],
        granularity: str,
    ) -> pd.DataFrame:
        fields_tuple = tuple(sorted(fields))
        start_ts = _ensure_datetime(start)
        end_ts = _ensure_datetime(end)

        cache_key = self._cache_key(symbol, start_ts, end_ts, fields_tuple, granularity)

        cached = self._memory_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        if self.enable_disk_cache:
            disk_path = self._disk_path(cache_key)
            if disk_path.exists():
                df = pd.read_parquet(disk_path)
                normalized = self._normalize(df, fields_tuple)
                self._memory_cache.put(cache_key, normalized)
                return normalized.copy()

        df = self._load_from_source(symbol, start_ts, end_ts, fields_tuple)
        normalized = self._normalize(df, fields_tuple)

        if self.enable_disk_cache:
            disk_path = self._disk_path(cache_key)
            normalized.to_parquet(disk_path, index=True)

        self._memory_cache.put(cache_key, normalized)
        return normalized.copy()

    def _cache_key(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fields: Tuple[str, ...],
        granularity: str,
    ) -> str:
        key_str = f"{symbol}|{start.isoformat()}|{end.isoformat()}|{','.join(fields)}|{granularity}"
        digest = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
        return digest

    def _disk_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.parquet"

    def _load_from_source(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fields: Tuple[str, ...],
    ) -> pd.DataFrame:
        years = range(start.year, end.year + 1)
        files = []
        for year in years:
            files.extend(list_tick_files(self.parquet_root, symbol=symbol, year=year))

        if not files:
            raise FileNotFoundError(
                f"No se encontraron parquet para {symbol} en {self.parquet_root}"
            )

        frames = []
        for fpath in files:
            df = pd.read_parquet(fpath)
            if not isinstance(df.index, pd.DatetimeIndex):
                # Algunos parquet guardan la columna timestamp en 'ts'.
                if "ts" in df.columns:
                    df = df.set_index("ts")
                else:
                    raise TypeError("El parquet necesita un índice datetime o columna 'ts'")
            frames.append(df)

        df_all = pd.concat(frames).sort_index()
        mask = (df_all.index >= start) & (df_all.index <= end)
        filtered = df_all.loc[mask]

        missing_cols = [c for c in fields if c not in filtered.columns]
        if missing_cols:
            raise KeyError(f"Faltan columnas {missing_cols} en los parquet para {symbol}")

        return filtered[fields]

    def _normalize(self, df: pd.DataFrame, fields: Tuple[str, ...]) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("El DataFrame debe tener un DatetimeIndex para normalizar")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        for field in fields:
            if field not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[field]):
                df[field] = df[field].astype("float64")
            elif pd.api.types.is_datetime64_any_dtype(df[field]):
                df[field] = pd.to_datetime(df[field], utc=True)
            else:
                df[field] = df[field].astype("string")

        df = df.loc[:, fields]
        df.index.name = "ts"
        return df
