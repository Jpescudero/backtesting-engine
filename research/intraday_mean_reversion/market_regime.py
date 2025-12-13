"""Market regime detection for intraday mean reversion research."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_REGIME_KEYS = [
    "REGIME_VOL_WINDOW_MIN",
    "REGIME_VOL_THRESHOLD_TYPE",
    "REGIME_VOL_THRESHOLD_VALUE",
    "REGIME_VOL_MODE",
    "REGIME_TREND_WINDOW_MIN",
    "REGIME_TREND_METHOD",
    "REGIME_TREND_THRESHOLD_TYPE",
    "REGIME_TREND_THRESHOLD_VALUE",
    "REGIME_TREND_MODE",
    "REGIME_SHOCK_WINDOW_MIN",
    "REGIME_SHOCK_SIGMA_THRESHOLD",
    "REGIME_SHOCK_COOLDOWN_MIN",
]


def detect_mean_reversion_regime(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """Identify when market conditions are supportive of mean reversion.

    Parameters
    ----------
    df : pd.DataFrame
        Intraday OHLCV data indexed by timestamp and containing a ``close``
        column.
    params : dict[str, Any]
        Parameter mapping containing the consolidated regime keys. If
        ``REGIME_USE_FILTERS`` is falsy the function returns a series of
        ``True`` values without applying any filters.

    Returns
    -------
    pd.Series
        Boolean series where ``True`` denotes that the strategy is allowed to
        operate under the mean-reversion regime.

    Raises
    ------
    KeyError
        If any required regime parameter is missing when filters are enabled.
    ValueError
        If an unsupported threshold type or mode is provided.
    """

    use_filters = bool(params.get("REGIME_USE_FILTERS", False))
    if not use_filters:
        return pd.Series(True, index=df.index, name="is_mr_regime")

    _validate_regime_params(params)

    close = df["close"].astype(float)
    logret_1m = np.log(close / close.shift(1))

    vol_window = int(params["REGIME_VOL_WINDOW_MIN"])
    vol_series = (
        logret_1m.rolling(window=vol_window, min_periods=vol_window).std().mul(np.sqrt(vol_window))
    )
    vol_ok = _evaluate_threshold(
        vol_series,
        str(params["REGIME_VOL_THRESHOLD_TYPE"]),
        float(params["REGIME_VOL_THRESHOLD_VALUE"]),
        str(params["REGIME_VOL_MODE"]),
        avoid_mode="avoid_high_vol",
        favor_mode="avoid_low_vol",
    )

    trend_window = int(params["REGIME_TREND_WINDOW_MIN"])
    trend_method = str(params["REGIME_TREND_METHOD"])
    trend_series = _compute_trend_strength(close, logret_1m, trend_window, trend_method)
    trend_ok = _evaluate_threshold(
        trend_series.abs(),
        str(params["REGIME_TREND_THRESHOLD_TYPE"]),
        float(params["REGIME_TREND_THRESHOLD_VALUE"]),
        str(params["REGIME_TREND_MODE"]),
        avoid_mode="avoid_high_trend",
        favor_mode="avoid_low_trend",
    )

    shock_window = int(params["REGIME_SHOCK_WINDOW_MIN"])
    shock_sigma = float(params["REGIME_SHOCK_SIGMA_THRESHOLD"])
    shock_cooldown = int(params["REGIME_SHOCK_COOLDOWN_MIN"])
    shock_active = _compute_shock_activity(logret_1m, shock_window, shock_sigma, shock_cooldown)
    shock_ok = ~shock_active

    regime = (vol_ok & trend_ok & shock_ok).fillna(False)
    return regime.astype(bool).rename("is_mr_regime")


def _validate_regime_params(params: dict[str, Any]) -> None:
    missing = [key for key in _REQUIRED_REGIME_KEYS if key not in params]
    if missing:
        missing_keys = ", ".join(missing)
        raise KeyError(f"Missing required regime parameters: {missing_keys}")


def _evaluate_threshold(
    series: pd.Series,
    threshold_type: str,
    threshold_value: float,
    mode: str,
    *,
    avoid_mode: str,
    favor_mode: str,
) -> pd.Series:
    threshold_type = threshold_type.lower()
    mode = mode.lower()

    if threshold_type == "percentile":
        cutoff = float(series.dropna().quantile(threshold_value))
    elif threshold_type == "absolute":
        cutoff = float(threshold_value)
    else:
        raise ValueError(f"Invalid threshold type '{threshold_type}'")

    if np.isnan(cutoff):
        return pd.Series(False, index=series.index)

    if mode == avoid_mode:
        return (series <= cutoff).fillna(False)
    if mode == favor_mode:
        return (series >= cutoff).fillna(False)
    raise ValueError(f"Invalid regime mode '{mode}'")


def _compute_trend_strength(
    close: pd.Series, log_returns: pd.Series, window: int, method: str
) -> pd.Series:
    method = method.lower()
    if window <= 1:
        return pd.Series(0.0, index=close.index)

    if method == "slope":
        return _rolling_slope(np.log(close), window)
    if method == "return_from_open":
        return log_returns.rolling(window=window, min_periods=window).sum()
    if method == "ema_slope":
        ema = close.ewm(span=window, adjust=False).mean()
        return ema.diff().rolling(window=window, min_periods=window).mean() / ema.shift(window)
    raise ValueError(f"Unsupported trend method '{method}'")


def _compute_shock_activity(
    log_returns: pd.Series, window: int, sigma_threshold: float, cooldown_min: int
) -> pd.Series:
    rolling_sigma = log_returns.rolling(window=window, min_periods=window).std()
    rolling_return = log_returns.rolling(window=window, min_periods=window).sum()
    valid_sigma = rolling_sigma > 0
    threshold = sigma_threshold * rolling_sigma
    shock_detected = (valid_sigma & (rolling_return.abs() > threshold)).fillna(False)

    active = shock_detected.copy()
    for offset in range(1, max(cooldown_min, 1)):
        active |= shock_detected.shift(-offset)

    return active.fillna(False)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    def _calc(values: np.ndarray) -> float:
        x = np.arange(len(values), dtype=float)
        y = values.astype(float)
        if np.any(~np.isfinite(y)):
            return np.nan
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    if window <= 1:
        return pd.Series(0.0, index=series.index)

    return series.rolling(window=window, min_periods=window).apply(_calc, raw=True)

