"""Feature construction for ML meta-labeling without look-ahead bias."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _encode_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create cyclical intraday time features and day-of-week encoding."""
    seconds_in_day = 24 * 60 * 60
    seconds = index.hour * 3600 + index.minute * 60 + index.second
    angles = 2 * np.pi * seconds / seconds_in_day
    features = {
        "time_of_day_sin": np.sin(angles),
        "time_of_day_cos": np.cos(angles),
        "day_of_week": index.dayofweek,
    }
    return pd.DataFrame(features, index=index)


def _trend_features(close: pd.Series, fast: int = 20, slow: int = 200) -> pd.Series:
    """Compute normalized EMA spread as a slow-moving trend filter."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    spread = (ema_fast - ema_slow) / close
    return spread


def _distance_to_high_of_day(close: pd.Series) -> pd.Series:
    """Distance to the intraday high of day, reset daily to avoid leakage."""
    daily_max = close.groupby(close.index.date).cummax()
    distance = (close - daily_max) / close
    return distance


def _recent_momentum(logret: pd.Series, windows: tuple[int, ...]) -> pd.DataFrame:
    """Rolling sums of log returns over the provided windows."""
    data = {f"momentum_{w}": logret.rolling(window=w, min_periods=w).sum() for w in windows}
    return pd.DataFrame(data, index=logret.index)


def _recent_realized_vol(logret: pd.Series, windows: tuple[int, ...]) -> pd.DataFrame:
    """Rolling realized volatility using log-return standard deviation."""
    data = {
        f"realized_vol_{w}": logret.rolling(window=w, min_periods=w).std().mul(np.sqrt(w))
        for w in windows
    }
    return pd.DataFrame(data, index=logret.index)


def build_feature_matrix(
    df_bars: pd.DataFrame, events: pd.DataFrame, params: dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct ML feature matrix aligned with event timestamps.

    Features rely only on information available up to the event timestamp to
    avoid leakage. Event indices are preserved for downstream joins.

    Args:
        df_bars: OHLCV DataFrame indexed by datetime with at least ``close``.
        events: Labeled events containing pre-computed z-score and label.
        params: Parameter dictionary controlling feature toggles and label name.

    Returns:
        Tuple of (X, y) where X is the feature matrix indexed like ``events``
        and y is the binary label series.
    """

    if events.empty:
        logger.info("No events provided; returning empty feature set")
        label_col = str(params.get("ML_LABEL", "is_r_H_net_positive"))
        return pd.DataFrame(index=events.index), events.get(label_col, pd.Series(dtype=int))

    close = df_bars["close"].astype(float)
    logret_1m = np.log(close / close.shift(1))

    feature_frames: list[pd.DataFrame] = []

    if params.get("FEAT_USE_Z", True):
        feature_frames.append(events[["z_score"]])
    if params.get("FEAT_USE_VOL", True):
        feature_frames.append(events[["vol_lookback", "ret_lookback"]])
    if params.get("FEAT_USE_TIME", True):
        time_feats = _encode_time_features(events.index)
        feature_frames.append(time_feats)
    if params.get("FEAT_USE_TREND", True):
        trend = _trend_features(close)
        feature_frames.append(trend.to_frame("trend_filter").loc[events.index])
    if params.get("FEAT_USE_DISTANCE_TO_HOD", True):
        distance = _distance_to_high_of_day(close)
        feature_frames.append(distance.to_frame("distance_to_HOD").loc[events.index])
    if params.get("FEAT_USE_RECENT_MOM", True):
        mom_frames = _recent_momentum(logret_1m, (5, 10, 20))
        feature_frames.append(mom_frames.loc[events.index])
        vol_frames = _recent_realized_vol(logret_1m, (5, 10, 20))
        feature_frames.append(vol_frames.loc[events.index])

    feature_matrix = pd.concat(feature_frames, axis=1)
    feature_matrix = feature_matrix.dropna()

    label_col = str(params.get("ML_LABEL", "is_r_H_net_positive"))
    y = events.loc[feature_matrix.index, label_col].astype(int)

    logger.info("Built feature matrix with shape %s", feature_matrix.shape)
    return feature_matrix, y
