"""Data loading utilities for intraday mean reversion research."""

from __future__ import annotations

import logging
from pathlib import Path, PurePath
from typing import Any

import pandas as pd

from src.config.paths import DATA_DIR, DATA_MIRRORS, PROJECT_ROOT, resolve_data_path

logger = logging.getLogger(__name__)


_EXPECTED_COLUMNS = {"open", "high", "low", "close", "volume"}


def _relative_to_casefold(path: Path, base: Path) -> Path | None:
    """Return the relative path if ``path`` is under ``base`` ignoring case.

    This helper mirrors ``Path.relative_to`` but performs a case-insensitive
    comparison so Windows-like absolute paths with different letter casing can
    still be mapped onto configured hubs.

    Parameters
    ----------
    path : pathlib.Path
        Absolute path to make relative.
    base : pathlib.Path
        Base directory to relativize against.

    Returns
    -------
    pathlib.Path | None
        Relative path components if ``path`` is a descendant of ``base`` when
        compared case-insensitively; otherwise ``None``.
    """

    path_parts = [part.lower() for part in PurePath(path).parts]
    base_parts = [part.lower() for part in PurePath(base).parts]

    if len(base_parts) > len(path_parts):
        return None

    if path_parts[: len(base_parts)] != base_parts:
        return None

    remainder = path.parts[len(base_parts) :]
    return Path(*remainder)


def _resolve_data_path(symbol: str, params: dict[str, Any]) -> Path:
    """Resolve the data file path using data hubs and project fallbacks.

    The function honors the active data hub, its mirrors, and project-relative
    paths. Absolute paths that point into a hub are remapped across available
    mirrors before failing over to the provided location.
    """

    base_path = Path(params["DATA_PATH"])
    pattern = str(params["DATA_FILE_PATTERN"])
    resolved_pattern = pattern.format(symbol=symbol)

    candidates: list[Path] = []
    resolved_filename = base_path / resolved_pattern

    def _add_candidate(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    if base_path.is_absolute():
        hubs = [DATA_DIR, *DATA_MIRRORS]
        relative_path: Path | None = None

        for hub in hubs:
            try:
                relative_path = resolved_filename.relative_to(hub)
                break
            except ValueError:
                relative_path = _relative_to_casefold(resolved_filename, hub)
                if relative_path is not None:
                    break

        if relative_path is not None:
            for hub in hubs:
                remapped = hub / relative_path
                _add_candidate(remapped)
        else:
            _add_candidate(resolved_filename)
    else:
        primary = resolve_data_path(resolved_filename)
        _add_candidate(primary)

        project_scoped = PROJECT_ROOT / resolved_filename
        _add_candidate(project_scoped)

    for existing in list(candidates):
        nested_with_symbol = existing.parent / symbol / existing.name
        _add_candidate(nested_with_symbol)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_roots: list[Path] = []
    if base_path.exists():
        search_roots.append(base_path)
    elif base_path.parent.exists():
        search_roots.append(base_path.parent)

    for root in search_roots:
        matches: list[Path] = sorted(root.rglob(resolved_filename.name))
        if resolved_pattern != resolved_filename.name:
            matches.extend(sorted(root.rglob(resolved_pattern)))
        for match in matches:
            if match.is_file():
                return match

    return candidates[0]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "timestamp" in df.columns:
        df = df.set_index(pd.to_datetime(df["timestamp"]))
        df = df.drop(columns=["timestamp"])
        return df
    raise ValueError("Dataframe must contain a datetime index or a 'timestamp' column")


def _deduplicate_index(df: pd.DataFrame) -> pd.DataFrame:
    if not df.index.has_duplicates:
        return df
    logger.warning("Duplicate timestamps detected; aggregating duplicates")
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    aggregated = (
        df.groupby(level=0)
        .agg({col: aggregations.get(col, "last") for col in df.columns})
        .sort_index()
    )
    return aggregated


def load_intraday_data(symbol: str, start_year: int, end_year: int, params: dict[str, Any]) -> pd.DataFrame:
    """Load minute-level intraday data for the given symbol and years.

    Parameters
    ----------
    symbol : str
        Instrument symbol to load.
    start_year : int
        Inclusive start year filter.
    end_year : int
        Inclusive end year filter.
    params : dict[str, Any]
        Configuration parameters containing data path and file pattern.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by datetime with OHLCV columns.

    Raises
    ------
    FileNotFoundError
        If the expected data file is missing.
    ValueError
        If the data lacks required columns or datetime information.
    """

    data_path = _resolve_data_path(symbol, params)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file for symbol '{symbol}' not found at {data_path}. "
            "Ensure the path and pattern are correct."
        )

    suffix = data_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(data_path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data file format: {suffix}")

    df = _ensure_datetime_index(df)
    missing_columns = _EXPECTED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Data file missing required columns: {sorted(missing_columns)}")

    df = df.sort_index()
    df = _deduplicate_index(df)

    year_mask = (df.index.year >= start_year) & (df.index.year <= end_year)
    filtered = df.loc[year_mask]
    if filtered.empty:
        raise ValueError(
            f"No data available for symbol '{symbol}' between years {start_year} and {end_year}."
        )

    return filtered
