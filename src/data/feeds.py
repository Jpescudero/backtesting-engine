# src/data/feeds.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config.paths import NPZ_DIR, ensure_directories_exist
from src.engine.registries import feed_registry


@dataclass
class OHLCVArrays:
    """
    Contenedor sencillo para arrays OHLCV + timestamps.

    Estos arrays son 1D y NumPy contiguos, ideales para Numba.
    """
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    low: np.ndarray
    c: np.ndarray
    v: np.ndarray


def _load_npz_file(npz_path: Path) -> OHLCVArrays:
    """
    Carga un archivo .npz con claves: ts, o, h, l, c, v
    y devuelve un OHLCVArrays.
    """
    npz_path = npz_path.resolve()
    print(f"[feeds] Cargando NPZ: {npz_path}")
    data = np.load(npz_path)

    ts = data["ts"]
    o = data["o"]
    h = data["h"]
    low = data["l"]
    c = data["c"]
    v = data["v"]

    return OHLCVArrays(ts=ts, o=o, h=h, low=low, c=c, v=v)


def _concat_ohlcv_arrays(arrays_list: List[OHLCVArrays]) -> OHLCVArrays:
    """
    Concatena listas de OHLCVArrays en un solo OHLCVArrays.
    """
    ts = np.concatenate([a.ts for a in arrays_list])
    o = np.concatenate([a.o for a in arrays_list])
    h = np.concatenate([a.h for a in arrays_list])
    low = np.concatenate([a.low for a in arrays_list])
    c = np.concatenate([a.c for a in arrays_list])
    v = np.concatenate([a.v for a in arrays_list])

    return OHLCVArrays(ts=ts, o=o, h=h, low=low, c=c, v=v)


def _extract_years_from_timestamps(ts: np.ndarray) -> np.ndarray:
    """Devuelve un array de años a partir de timestamps enteros o datetime64."""

    dt64 = np.asarray(ts)

    if not np.issubdtype(dt64.dtype, np.datetime64):
        dt64 = dt64.astype("datetime64[ns]")

    return dt64.astype("datetime64[Y]").astype(np.int64) + 1970


def filter_ohlcv_by_years(data: OHLCVArrays, years: Sequence[int]) -> OHLCVArrays:
    """
    Filtra un OHLCVArrays quedándose sólo con las barras de los años indicados.
    """
    if not years:
        raise ValueError("Debe proporcionarse al menos un año para filtrar los datos")

    ts_years = _extract_years_from_timestamps(data.ts)
    target_years = np.array(list(years), dtype=np.int64)
    mask = np.isin(ts_years, target_years)

    if not mask.any():
        available_years = np.unique(ts_years)
        raise ValueError(
            "No hay barras para los años especificados: "
            f"{list(years)}. "
            "Años disponibles en el feed: "
            f"{available_years.tolist()}"
        )

    return OHLCVArrays(
        ts=data.ts[mask],
        o=data.o[mask],
        h=data.h[mask],
        low=data.low[mask],
        c=data.c[mask],
        v=data.v[mask],
    )


def split_ohlcv_train_test(
    data: OHLCVArrays, train_years: Sequence[int], test_years: Sequence[int]
) -> Tuple[OHLCVArrays, OHLCVArrays]:
    """
    Genera un split train/test por años sin duplicar la carga de datos.
    """
    train = filter_ohlcv_by_years(data, train_years)
    test = filter_ohlcv_by_years(data, test_years)
    return train, test


class NPZOHLCVFeed:
    """
    DataFeed basado en ficheros .npz de OHLCV.

    Uso típico:
        feed = NPZOHLCVFeed(symbol="NDXm", timeframe="1m")
        data = feed.load_all()  # devuelve OHLCVArrays
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1m",
        base_dir: Optional[Path] = None,
        use_mmap: bool = False,
    ) -> None:
        """
        - symbol: p.ej. 'NDXm'
        - timeframe: sufijo que se usó en el nombre del npz (p.ej. '_1m').
        - base_dir: carpeta base de NPZ; por defecto NPZ_DIR / symbol
        - use_mmap: si True, usa mmap_mode='r' para reducir RAM (no concatenado).
                    De momento lo dejamos sin implementar completamente (versión simple)
        """
        ensure_directories_exist()

        self.symbol = symbol
        self.timeframe = timeframe
        self.use_mmap = use_mmap

        if base_dir is None:
            base_dir = NPZ_DIR / symbol

        self.base_dir = base_dir.resolve()

    def list_npz_files(self) -> list[Path]:
        """
        Lista todos los archivos .npz del símbolo y timeframe.
        Por simplicidad, filtramos por '*<timeframe>.npz', p.ej. '*_1m.npz'.
        """
        pattern = f"*_{self.timeframe}.npz" if not self.timeframe.startswith("_") else f"*{self.timeframe}.npz"
        files = sorted(self.base_dir.glob(pattern))
        return files

    def load_all(self) -> OHLCVArrays:
        """
        Carga todos los .npz del símbolo/timeframe y los concatena
        en un único OHLCVArrays.
        """
        files = self.list_npz_files()
        if not files:
            raise FileNotFoundError(
                f"No se encontraron archivos .npz para {self.symbol}, timeframe={self.timeframe} en {self.base_dir}"
            )

        arrays_list: list[OHLCVArrays] = []
        for f in files:
            arrays_list.append(_load_npz_file(f))

        print(f"[feeds] Concatenando {len(files)} archivos NPZ para {self.symbol} ({self.timeframe})")
        return _concat_ohlcv_arrays(arrays_list)

    def load_years(self, years: Sequence[int]) -> OHLCVArrays:
        """
        Carga únicamente las barras pertenecientes a los años indicados.
        """
        all_data = self.load_all()
        return filter_ohlcv_by_years(all_data, years)

    def load_train_test(
        self, train_years: Sequence[int], test_years: Sequence[int]
    ) -> Tuple[OHLCVArrays, OHLCVArrays]:
        """
        Devuelve una tupla (train, test) filtrada por años.
        """
        all_data = self.load_all()
        return split_ohlcv_train_test(all_data, train_years, test_years)


# Registro por defecto
feed_registry.register("npz")(NPZOHLCVFeed)


if __name__ == "__main__":
    # Ejemplo rápido:
    # python -m src.data.feeds
    feed = NPZOHLCVFeed(symbol="NDXm", timeframe="1m")
    data = feed.load_all()
    print("[feeds] Número de barras cargadas:", data.c.shape[0])
