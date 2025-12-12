"""Data loading utilities for intraday mean reversion research."""

from __future__ import annotations

import logging
from pathlib import Path, PurePath
from typing import Any

import numpy as np
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


def _resolve_data_path(symbol: str, base_path: Path, resolved_pattern: str) -> Path:
    """Resolve the data file path using data hubs and project fallbacks.

    The function honors the active data hub, its mirrors, and project-relative
    paths. Absolute paths that point into a hub are remapped across available
    mirrors before failing over to the provided location.
    """

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


def _find_symbol_directory(base_path: Path, symbol: str) -> Path | None:
    """Locate a directory named after the symbol within candidate roots.

    The search considers the configured ``DATA_PATH``, all of its ancestors,
    and configured data hubs. This is helpful when ``DATA_PATH`` mistakenly
    points to a specific file or sibling directory while the actual symbol
    data lives elsewhere under the same tree (for example,
    ``.../data/NDXm_1m.parquet`` while data resides in
    ``.../parquet/ticks/NDXm``).
    """

    search_roots: list[Path] = []
    seen: set[Path] = set()

    for candidate in (base_path, *base_path.parents, DATA_DIR, *DATA_MIRRORS):
        if candidate is None:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            search_roots.append(candidate)

    lower_symbol = symbol.lower()
    for root in search_roots:
        if root.name.lower() == lower_symbol and root.is_dir():
            return root
        for candidate in root.rglob("*"):
            if candidate.is_dir() and candidate.name.lower() == lower_symbol:
                return candidate
    return None


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
        DataFrame indexed by datetime with OHLCV columns. If ``DATA_PATH``
        points to a directory, all matching files under that directory (and
        a symbol-named subdirectory, if present) are loaded and concatenated.

    Raises
    ------
    FileNotFoundError
        If the expected data file is missing.
    ValueError
        If the data lacks required columns or datetime information.
    """

    base_path = Path(params["DATA_PATH"])
    pattern = str(params["DATA_FILE_PATTERN"])
    resolved_pattern = pattern.format(symbol=symbol)

    data_path = _resolve_data_path(symbol, base_path, resolved_pattern)
    data_exists = data_path.exists()

    def _read_npz(path: Path) -> pd.DataFrame:
        required_keys = {"ts", "o", "h", "l", "c", "v"}
        data = np.load(path)
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise ValueError(
                f"NPZ file {path} missing required keys: {sorted(missing_keys)}"
            )

        idx = pd.to_datetime(data["ts"], unit="ns", utc=True)
        return pd.DataFrame(
            {
                "open": data["o"],
                "high": data["h"],
                "low": data["l"],
                "close": data["c"],
                "volume": data["v"],
            },
            index=idx,
        )

    def _read_file(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
        if suffix == ".npz":
            return _read_npz(path)
        raise ValueError(f"Unsupported data file format: {suffix}")

    def _load_directory(path: Path) -> pd.DataFrame:
        target_suffix = Path(resolved_pattern).suffix.lower()
        expected_stem = Path(resolved_pattern).stem
        allowed_suffixes = {".parquet", ".pq", ".csv", ".txt", ".npz"}
        explicit_suffix = bool(target_suffix)
        if explicit_suffix:
            allowed_suffixes = {target_suffix}
        fallback_suffixes = {".parquet", ".pq", ".csv", ".txt", ".npz"}

        def _stem_matches(candidate_stem: str, strict: bool) -> bool:
            if not expected_stem:
                return True

            expected_stem_lower = expected_stem.lower()
            candidate_stem_lower = candidate_stem.lower()

            if candidate_stem == expected_stem or candidate_stem_lower == expected_stem_lower:
                return True

            if strict:
                return False

            simplified_expected = expected_stem_lower.replace("-", "").replace("_", "")
            simplified_candidate = candidate_stem_lower.replace("-", "").replace("_", "")

            if simplified_candidate == simplified_expected:
                return True

            symbol_lower = symbol.lower()
            return simplified_candidate.startswith(symbol_lower)

        def _collect_files(extensions: set[str], strict: bool) -> list[Path]:
            return sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix.lower() in extensions
                and _stem_matches(candidate.stem, strict=strict)
            )

        files = _collect_files(allowed_suffixes, strict=True)

        if not files and explicit_suffix:
            files = _collect_files(fallback_suffixes, strict=True)
            if files:
                logger.warning(
                    "No files with expected suffix '%s' found under %s; using fallback formats.",
                    target_suffix,
                    path,
                )

        if not files:
            relaxed_extensions = fallback_suffixes if explicit_suffix else allowed_suffixes
            files = _collect_files(relaxed_extensions, strict=False)
            if files:
                logger.warning(
                    "No files matching expected pattern '%s' found under %s; "
                    "loading files that start with symbol '%s'.",
                    resolved_pattern,
                    path,
                    symbol,
                )

        if not files:
            raise FileNotFoundError(
                f"No data files found for symbol '{symbol}' under directory {path}. "
                "Ensure the path and pattern are correct."
            )

        frames = [_read_file(file) for file in files]
        return pd.concat(frames, ignore_index=False)

    if data_exists and data_path.is_dir():
        df = _load_directory(data_path)
    elif data_exists:
        df = _read_file(data_path)
    else:
        symbol_dir = _find_symbol_directory(base_path, symbol)
        if symbol_dir is not None:
            df = _load_directory(symbol_dir)
        else:
            raise FileNotFoundError(
                f"Data file for symbol '{symbol}' not found at {data_path}. "
                "Ensure the path and pattern are correct."
            )

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
