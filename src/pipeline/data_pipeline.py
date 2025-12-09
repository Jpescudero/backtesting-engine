from __future__ import annotations

from pathlib import Path

from src.config.paths import (
    NPZ_DIR,
    PARQUET_TICKS_DIR,
    ensure_directories_exist,
    resolve_data_dir_with_pattern,
)
from src.data.bars1m_to_excel import generate_1m_bars_csv, get_default_output_csv
from src.data.csv_1m_to_npz import csv_1m_to_npz
from src.data.data_to_parquet import DEFAULT_SYMBOL, data_to_parquet
from src.data.data_utils import list_tick_files


def ensure_ticks_and_csv(symbol: str = DEFAULT_SYMBOL) -> Path:
    """
    Garantiza que existan parquets de ticks y el CSV de barras de 1 minuto para un símbolo.
    """
    ensure_directories_exist()

    parquet_root = resolve_data_dir_with_pattern(PARQUET_TICKS_DIR, "*.parquet")

    try:
        tick_files = list_tick_files(parquet_root, symbol=symbol)
    except FileNotFoundError:
        tick_files = []

    if not tick_files:
        data_to_parquet(symbol=symbol)
        tick_files = list_tick_files(PARQUET_TICKS_DIR, symbol=symbol)
        if not tick_files:
            raise RuntimeError(f"No se han podido generar parquets de ticks para {symbol}")

    bars_csv = get_default_output_csv(symbol=symbol)
    if not bars_csv.exists():
        bars_csv = generate_1m_bars_csv(symbol=symbol)

    if not bars_csv.exists():
        raise RuntimeError(f"No se ha podido generar el CSV de barras 1m para {symbol}")

    return bars_csv


def ensure_npz_from_csv(symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Path:
    """
    Garantiza que exista al menos un fichero NPZ para el símbolo/timeframe indicado.
    """
    ensure_directories_exist()

    symbol_npz_dir = (NPZ_DIR / symbol).resolve()
    symbol_npz_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*_{timeframe}.npz" if not timeframe.startswith("_") else f"*{timeframe}.npz"
    existing_npz = list(symbol_npz_dir.glob(pattern))

    if existing_npz:
        return existing_npz[0]

    bars_csv = ensure_ticks_and_csv(symbol=symbol)
    npz_path = csv_1m_to_npz(symbol=symbol, csv_path=bars_csv)

    return npz_path


def prepare_npz_dataset(symbol: str, timeframe: str = "1m") -> tuple[Path, Path]:
    """
    Devuelve los paths al CSV de 1m y al NPZ (creándolos si faltan).
    """
    bars_csv_path = ensure_ticks_and_csv(symbol=symbol)
    npz_path = ensure_npz_from_csv(symbol=symbol, timeframe=timeframe)
    return bars_csv_path, npz_path
