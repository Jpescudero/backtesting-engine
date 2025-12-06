# src/config/paths.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _get_main_dir() -> Path:
    """
    Devuelve el directorio donde está el script principal (__main__).
    Si por lo que sea no se encuentra, hace un fallback al directorio
    padre de este propio archivo.
    """
    main_module = sys.modules.get("__main__")
    main_file: Optional[str] = getattr(main_module, "__file__", None)

    if main_file is not None:
        # Ej: C:/.../10. Backtesting/  si ejecutas `python main.py`
        return Path(main_file).resolve().parent

    # Fallback: si se ejecuta desde un intérprete interactivo, notebooks, etc.
    # asumimos que este archivo está dentro de src/config/
    # y el root es dos niveles por encima.
    return Path(__file__).resolve().parents[2]


# ================================
# Directorios base del proyecto
# ================================

PROJECT_ROOT   = _get_main_dir()          # carpeta donde vive main.py
SRC_DIR        = PROJECT_ROOT / "src"
DATA_DIR       = PROJECT_ROOT / "data"
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"
REPORTS_DIR    = PROJECT_ROOT / "reports"

# Subcarpetas de código dentro de src/
CONFIG_DIR       = SRC_DIR / "config"
DATA_MODULE_DIR  = SRC_DIR / "data"
ENGINE_DIR       = SRC_DIR / "engine"
STRATEGIES_DIR   = SRC_DIR / "strategies"
VISUALIZATION_DIR = SRC_DIR / "visualization"
ANALYTICS_DIR    = SRC_DIR / "analytics"


# ================================
# Subcarpetas de datos
# ================================

RAW_DATA_DIR       = DATA_DIR / "raw"
DARWINEX_RAW_DIR   = RAW_DATA_DIR / "darwinex"

PARQUET_DIR         = DATA_DIR / "parquet"
PARQUET_TICKS_DIR   = PARQUET_DIR / "ticks"
PARQUET_BARS_1M_DIR = PARQUET_DIR / "bars_1m"

NPZ_DIR        = DATA_DIR / "npz"
# Carpeta específica para NDXm (tu símbolo actual principal)
NPZ_NDXM_DIR   = NPZ_DIR / "NDXm"

OTHER_DATA_DIR = DATA_DIR / "other"


def ensure_directories_exist() -> None:
    """
    Crea (si no existen) todas las carpetas principales de datos.
    Llamar una vez al inicio del programa.
    """
    dirs_to_create = [
        DATA_DIR,
        RAW_DATA_DIR,
        DARWINEX_RAW_DIR,
        PARQUET_DIR,
        PARQUET_TICKS_DIR,
        PARQUET_BARS_1M_DIR,
        NPZ_DIR,
        NPZ_NDXM_DIR,
        OTHER_DATA_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
