"""Configuration loader for intraday mean reversion research."""

from __future__ import annotations

from pathlib import Path
from typing import Any


_REQUIRED_KEYS = [
    "SYMBOL",
    "START_YEAR",
    "END_YEAR",
    "DATA_PATH",
    "DATA_FILE_PATTERN",
    "LOOKBACK_MINUTES",
    "ZSCORE_ENTRY",
    "HOLD_TIME_BARS",
    "SESSION_START_TIME",
    "SESSION_END_TIME",
]


def _convert_value(raw_value: str) -> Any:
    """Infer type from a configuration string value.

    Parameters
    ----------
    raw_value : str
        Raw value read from the configuration file.

    Returns
    -------
    Any
        Value converted to int, float, bool, list, or left as string.
    """

    value = raw_value.strip()
    if "," in value:
        items = [item.strip() for item in value.split(",") if item.strip()]
        return [_convert_value(item) for item in items]

    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_params(path: str) -> dict[str, Any]:
    """Load research parameters from a key=value configuration file.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict[str, Any]
        Dictionary of parameters with inferred types.

    Raises
    ------
    FileNotFoundError
        If the configuration file is missing.
    ValueError
        If required keys are missing or any line is malformed.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    params: dict[str, Any] = {}
    for idx, line in enumerate(config_path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid line {idx}: '{line}' (expected key=value)")
        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        params[key] = _convert_value(value)

    missing = [key for key in _REQUIRED_KEYS if key not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    return params
