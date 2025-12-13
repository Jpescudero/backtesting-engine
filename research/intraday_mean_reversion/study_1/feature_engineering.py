"""Feature engineering for Study 1 (NO mean reversion regimes)."""
from __future__ import annotations

import hashlib
import json
from datetime import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


def parse_allowed_time_windows(value: str | Iterable[str] | Iterable[tuple[str, str]] | None) -> list[tuple[time, time]]:
    """Parse allowed time windows into a list of ``datetime.time`` tuples."""

    if value is None:
        return []

    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    else:
        tokens = list(value)

    windows: list[tuple[time, time]] = []
    for token in tokens:
        if isinstance(token, tuple) and len(token) == 2:
            start_raw, end_raw = token
        elif isinstance(token, str):
            if "-" not in token:
                raise ValueError(f"Invalid time window '{token}' (expected HH:MM-HH:MM)")
            start_raw, end_raw = token.split("-", maxsplit=1)
        else:
            raise ValueError(f"Unsupported time window token: {token!r}")

        start_parsed = _parse_hhmm(str(start_raw).strip())
        end_parsed = _parse_hhmm(str(end_raw).strip())
        windows.append((start_parsed, end_parsed))

    return windows


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling slope using simple linear regression over the window."""

    if window <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)

    def _calc(values: np.ndarray) -> float:
        x = np.arange(len(values), dtype=float)
        y = values.astype(float)
        if np.any(~np.isfinite(y)):
            return np.nan
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    return series.rolling(window=window, min_periods=window).apply(_calc, raw=True)


def compute_intraday_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Compute intraday features used by Study 1 regime filters."""

    close = df["close"].astype(float)
    logret_1m = np.log(close / close.shift(1))

    vol_window = int(params.get("vol_window_min", 30))
    trend_window = int(params.get("trend_window_min", 60))
    trend_method = str(params.get("trend_strength_method", "slope")).lower()
    shock_window = int(params.get("shock_window_min", 5))
    shock_sigma = float(params.get("shock_sigma_threshold", 2.5))
    shock_cooldown = int(params.get("shock_cooldown_min", shock_window))

    realized_vol = (
        logret_1m.rolling(window=vol_window, min_periods=vol_window).std().mul(np.sqrt(vol_window))
    )

    trend_strength = _compute_trend_strength(close, logret_1m, trend_window, trend_method)

    shock_regime = compute_shock_regime(logret_1m, shock_window, shock_sigma, shock_cooldown)

    features = pd.DataFrame(
        {
            "realized_vol": realized_vol,
            "trend_strength": trend_strength,
            "shock_active": shock_regime,
            "logret_1m": logret_1m,
        },
        index=df.index,
    )
    return features


def compute_shock_regime(
    log_returns: pd.Series, window: int, sigma_threshold: float, cooldown_min: int
) -> pd.Series:
    """Flag recent shocks and extend the avoidance period with a cooldown."""

    rolling_sigma = log_returns.rolling(window=window, min_periods=window).std()
    rolling_return = log_returns.rolling(window=window, min_periods=window).sum()
    shock_detected = (rolling_return.abs() > (sigma_threshold * rolling_sigma)).fillna(False)

    active = shock_detected.copy()
    for offset in range(1, max(cooldown_min, 1)):
        active |= shock_detected.shift(-offset)

    return active.fillna(False)


def _compute_trend_strength(
    close: pd.Series, log_returns: pd.Series, window: int, method: str
) -> pd.Series:
    method = method.lower()
    if method == "slope":
        return _rolling_slope(np.log(close), window)
    if method == "return_from_open":
        return log_returns.rolling(window=window, min_periods=window).sum()
    if method == "ema_slope":
        ema = close.ewm(span=window, adjust=False).mean()
        return ema.diff().rolling(window=window, min_periods=window).mean() / ema.shift(window)
    raise ValueError(f"Unsupported trend_strength_method='{method}'")


def build_run_id(params: dict[str, Any]) -> str:
    """Create a deterministic run identifier from params and timestamp hash."""

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    return f"{timestamp}_{digest}"
