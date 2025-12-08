# src/data/data_to_parquet.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.config.paths import (
    DARWINEX_RAW_DIR,
    ensure_directories_exist,
    PARQUET_TICKS_DIR,
)

# Símbolo por defecto (puedes cambiarlo o parametrizarlo desde main)
DEFAULT_SYMBOL = "NDXm"

FILENAME_RE = re.compile(r"(?P<symbol>.+)_(?P<side>BID|ASK)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.log\.gz$")


def parse_filename(path: Path) -> dict:
    """
    Extrae symbol, side (BID/ASK), date (YYYY-MM-DD) y hour (HH)
    a partir del nombre del fichero Darwinex.
    """
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Nombre de fichero no reconocido: {path.name}")
    return m.groupdict()


def read_side_file(path: Path, side: str) -> pd.DataFrame:
    """
    Lee un fichero BID o ASK de Darwinex.

    Formato esperado:
        timestamp_ms, price, volume

    Devuelve un DataFrame con índice ts (datetime UTC) y columnas:
        - 'bid' / 'ask'
        - 'bid_volume' / 'ask_volume'
    """
    side = side.upper()
    assert side in {"BID", "ASK"}

    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp_ms", "price", "volume"],
    )

    # Epoch ms -> datetime (UTC)
    df["ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])

    price_col = "bid" if side == "BID" else "ask"
    vol_col = "bid_volume" if side == "BID" else "ask_volume"

    df = df.rename(columns={"price": price_col, "volume": vol_col})
    df = df.set_index("ts").sort_index()

    return df


def merge_bid_ask_to_parquet(
    bid_path: Path | None,
    ask_path: Path | None,
    symbol: str,
    date: str,
    hour: str,
    out_root: Path,
) -> Path:
    """
    Combina (si existen) los ficheros BID y ASK de una misma hora en
    un único DataFrame y lo guarda como parquet.

    Estructura que se genera:

        out_root / <symbol> / <year> / "<symbol>_<YYYY-MM-DD>_<HH>.parquet"

    Retorna la ruta del parquet generado.
    """
    dfs: list[pd.DataFrame] = []
    if bid_path is not None:
        dfs.append(read_side_file(bid_path, "BID"))
    if ask_path is not None:
        dfs.append(read_side_file(ask_path, "ASK"))

    if not dfs:
        raise ValueError(f"No hay datos BID/ASK para {symbol} {date} {hour}")

    df = pd.concat(dfs, axis=1).sort_index()

    # Forward-fill del bid/ask para tener siempre el mejor lado conocido
    if "bid" in df.columns:
        df["bid"] = df["bid"].ffill()
    if "ask" in df.columns:
        df["ask"] = df["ask"].ffill()

    # Precio medio (mid)
    if "bid" in df.columns and "ask" in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
    elif "bid" in df.columns:
        df["mid"] = df["bid"]
    elif "ask" in df.columns:
        df["mid"] = df["ask"]

    # Eliminamos filas sin mid
    df = df.dropna(subset=["mid"])

    # Añadimos símbolo
    df["symbol"] = symbol

    # Estructura de carpetas por año
    year = date[:4]
    out_subdir = out_root / symbol / year
    out_subdir.mkdir(parents=True, exist_ok=True)

    out_path = out_subdir / f"{symbol}_{date}_{hour}.parquet"

    # Necesitas 'pyarrow' o 'fastparquet' instalado en tu entorno.
    df.to_parquet(out_path, compression="snappy")

    return out_path


def data_to_parquet(
    symbol: str = DEFAULT_SYMBOL,
    raw_dir: Path | None = None,
    out_root: Path | None = None,
) -> None:
    """
    Recorre todos los .log.gz de Darwinex para un símbolo y genera parquets limpios.

    - Detecta ficheros BID y ASK de la misma hora y los une.
    - Es robusto a que falte BID o ASK en alguna hora (usa lo que haya).
    - Genera la estructura:
        PARQUET_TICKS_DIR / <symbol> / <year> / "<symbol>_<YYYY-MM-DD>_<HH>.parquet"
    """
    ensure_directories_exist()

    # Por defecto: data/raw/darwinex/<symbol>/
    base_raw_dir = Path(raw_dir) if raw_dir is not None else (DARWINEX_RAW_DIR / symbol)
    base_raw_dir = base_raw_dir.resolve()

    if not base_raw_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de datos crudos: {base_raw_dir}")

    # Carpeta de salida: data/parquet/ticks/
    out_root = Path(out_root) if out_root is not None else PARQUET_TICKS_DIR
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Buscamos todos los .log.gz de forma recursiva (por si tienes subcarpetas BID/ASK)
    files = sorted(base_raw_dir.rglob("*.log.gz"))
    if not files:
        print(f"No he encontrado ficheros .log.gz en {base_raw_dir}")
        return

    # Agrupamos por (symbol, date, hour)
    groups: Dict[Tuple[str, str, str], dict] = {}
    for f in files:
        info = parse_filename(f)
        key = (info["symbol"], info["date"], info["hour"])
        grp = groups.setdefault(key, {})
        grp[info["side"].lower()] = f  # 'bid' o 'ask'

    total = len(groups)
    for i, ((sym, date, hour), d) in enumerate(sorted(groups.items()), 1):
        bid_path = d.get("bid")
        ask_path = d.get("ask")
        try:
            out_path = merge_bid_ask_to_parquet(
                bid_path=bid_path,
                ask_path=ask_path,
                symbol=sym,
                date=date,
                hour=hour,
                out_root=out_root,
            )
            print(f"[{i}/{total}] {sym} {date} {hour} -> {out_path}")
        except Exception as exc:
            print(f"[{i}/{total}] ERROR procesando {sym} {date} {hour}: {exc}")


if __name__ == "__main__":
    # Uso autónomo (ejecución como módulo):
    # python -m src.data.data_to_parquet
    data_to_parquet()
