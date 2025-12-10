# src/config/paths.py

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


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

PROJECT_ROOT = _get_main_dir()  # carpeta donde vive main.py
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

# Carpeta para scripts y experimentos de research
RESEARCH_DIR = PROJECT_ROOT / "research"

# Subcarpetas de código dentro de src/
CONFIG_DIR = SRC_DIR / "config"
DATA_MODULE_DIR = SRC_DIR / "data"
ENGINE_DIR = SRC_DIR / "engine"
STRATEGIES_DIR = SRC_DIR / "strategies"
VISUALIZATION_DIR = SRC_DIR / "visualization"
ANALYTICS_DIR = SRC_DIR / "analytics"


# ================================
# Configuración de hubs de datos
# ================================

DATA_ROOTS_FILE = CONFIG_DIR / "data_roots.json"
DATA_ROOTS_TEMPLATE = CONFIG_DIR / "data_roots.example.json"

DEFAULT_LOCAL_HUBS = [
    Path(r"C:/Users/JorgeP/Market Data"),
    Path(r"C:/Users/Jorge/Market Data"),
    Path(r"C:/Users/jorge/Market Data"),
]

DEFAULT_CLOUD_HUBS = [
    Path(r"C:/Users/JorgeP/OneDrive/Bolsa/Scripts/10. Backtesting/data"),
]


def _is_windows_style(path: Path) -> bool:
    return ":" in str(path)


def _platform_supports(path: Path) -> bool:
    """Evita crear rutas de Windows cuando el runtime es POSIX."""

    if os.name != "nt" and _is_windows_style(path):
        return False
    return True


def _load_data_hubs_from_file(path: Path) -> tuple[list[Path], list[Path]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    local_raw = payload.get("local_hubs", []) or []
    cloud_raw = payload.get("cloud_hubs", []) or []

    local_hubs = [Path(p) for p in local_raw]
    cloud_hubs = [Path(p) for p in cloud_raw]
    return local_hubs, cloud_hubs


def _load_data_hubs() -> tuple[list[Path], list[Path]]:
    """Lee los hubs configurados o aplica defaults seguros."""

    config_path: Optional[Path] = None
    if DATA_ROOTS_FILE.exists():
        config_path = DATA_ROOTS_FILE
    elif DATA_ROOTS_TEMPLATE.exists():
        config_path = DATA_ROOTS_TEMPLATE

    if config_path is not None:
        try:
            return _load_data_hubs_from_file(config_path)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(
                "No se pudo leer %s: %s. Se usarán rutas por defecto.",
                config_path,
                exc,
            )

    return DEFAULT_LOCAL_HUBS, DEFAULT_CLOUD_HUBS


def _filter_supported(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if _platform_supports(p)]


def _resolve_primary_root(local_hubs: Sequence[Path], fallback: Path) -> Path:
    for hub in local_hubs:
        if not _platform_supports(hub):
            continue
        if hub.exists():
            return hub
        try:
            hub.mkdir(parents=True, exist_ok=True)
            return hub
        except OSError:
            continue

    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _available_mirrors(candidates: Iterable[Path], primary: Path) -> list[Path]:
    mirrors: list[Path] = []
    for path in candidates:
        if not _platform_supports(path) or path.resolve() == primary.resolve():
            continue
        if path.exists():
            mirrors.append(path)
    return mirrors


LOCAL_DATA_HUBS, CLOUD_DATA_HUBS = _load_data_hubs()
LOCAL_DATA_HUBS = _filter_supported(LOCAL_DATA_HUBS)
CLOUD_DATA_HUBS = _filter_supported(CLOUD_DATA_HUBS)

PROJECT_DATA_FALLBACK = PROJECT_ROOT / "data"

DATA_DIR = _resolve_primary_root(LOCAL_DATA_HUBS, fallback=PROJECT_DATA_FALLBACK)

DATA_MIRRORS: list[Path] = _available_mirrors(
    [*LOCAL_DATA_HUBS, *CLOUD_DATA_HUBS, PROJECT_DATA_FALLBACK], primary=DATA_DIR
)


# ================================
# Subcarpetas de datos
# ================================

RAW_DATA_DIR = DATA_DIR / "raw"
DARWINEX_RAW_DIR = RAW_DATA_DIR / "darwinex"

PARQUET_DIR = DATA_DIR / "parquet"
PARQUET_TICKS_DIR = PARQUET_DIR / "ticks"
PARQUET_BARS_1M_DIR = PARQUET_DIR / "bars_1m"

NPZ_DIR = DATA_DIR / "npz"
# Carpeta específica para NDXm (tu símbolo actual principal)
NPZ_NDXM_DIR = NPZ_DIR / "NDXm"

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
        MODELS_DIR,
        RESEARCH_DIR,
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)


def resolve_data_path(relative: str | Path) -> Path:
    """Devuelve la ruta al archivo/directorio, prefiriendo el hub activo.

    Si el path relativo no existe en el hub principal, busca en los mirrors
    disponibles (por ejemplo, la copia en OneDrive).
    """

    rel_raw = Path(relative)

    if rel_raw.is_absolute():
        try:
            rel = rel_raw.relative_to(DATA_DIR)
        except ValueError:
            return rel_raw
    else:
        rel = rel_raw

    primary = DATA_DIR / rel
    if primary.exists():
        return primary

    for mirror in DATA_MIRRORS:
        candidate = mirror / rel
        if candidate.exists():
            return candidate

    return primary


def resolve_data_dir_with_pattern(base_dir: Path, pattern: str) -> Path:
    """Selecciona el directorio que realmente contiene ficheros coincidentes.

    Prioriza el hub activo; si no hay coincidencias, intenta localizar los
    ficheros en los mirrors disponibles antes de devolver el path original.
    """

    try:
        relative = base_dir.relative_to(DATA_DIR)
    except ValueError:
        return base_dir

    primary = DATA_DIR / relative
    if list(primary.glob(pattern)):
        return primary

    for mirror in DATA_MIRRORS:
        candidate = mirror / relative
        if list(candidate.glob(pattern)):
            return candidate

    return primary


def describe_data_hubs() -> str:
    """Resumen textual de las ubicaciones de datos configuradas."""

    mirrors = ", ".join(str(p) for p in DATA_MIRRORS) if DATA_MIRRORS else "(ninguno)"
    return f"principal={DATA_DIR}; mirrors={mirrors}"


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _sync_tree(source: Path, target: Path) -> list[Path]:
    copied: list[Path] = []

    for file_path in _iter_files(source):
        rel = file_path.relative_to(source)
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            if not destination.exists():
                shutil.copy2(file_path, destination)
                copied.append(destination)
                continue

            src_stat = file_path.stat()
            dst_stat = destination.stat()
            if src_stat.st_mtime > dst_stat.st_mtime or src_stat.st_size != dst_stat.st_size:
                shutil.copy2(file_path, destination)
                copied.append(destination)
        except OSError as exc:
            logging.warning("No se pudo sincronizar %s -> %s: %s", file_path, destination, exc)

    return copied


def sync_primary_to_cloud(mirrors: Optional[Sequence[Path]] = None) -> list[Path]:
    """Sincroniza el hub activo hacia mirrors en la nube.

    Devuelve las rutas de mirrors que se intentaron actualizar.
    """

    targets = list(mirrors) if mirrors is not None else list(CLOUD_DATA_HUBS)
    updated: list[Path] = []

    for target in targets:
        if not _platform_supports(target):
            continue

        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("No se pudo crear el mirror %s: %s", target, exc)
            continue

        copied = _sync_tree(DATA_DIR, target)
        if copied:
            logging.info("Copiados %d elementos a %s", len(copied), target)
        updated.append(target)

    return updated


def bootstrap_data_roots(sync_to_cloud: bool = False) -> str:
    """Prepara los hubs de datos y opcionalmente sincroniza con la nube."""

    ensure_directories_exist()

    if sync_to_cloud:
        sync_primary_to_cloud()

    summary = describe_data_hubs()
    logging.info("Hubs de datos: %s", summary)
    return summary
