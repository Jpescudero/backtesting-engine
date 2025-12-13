from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_mean_reversion.market_regime import detect_mean_reversion_regime


def test_detect_mean_reversion_regime_not_all_true_or_false() -> None:
    index = pd.date_range("2022-01-03 09:30", periods=160, freq="1min", tz="UTC")

    calm_returns = np.concatenate([np.full(40, 0.00005), np.full(40, -0.00005)])
    trending_returns = np.full(80, 0.002)
    log_returns = np.concatenate([calm_returns, trending_returns])
    price = 100 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame({"close": price}, index=index)

    params = {
        "REGIME_USE_FILTERS": True,
        "REGIME_VOL_WINDOW_MIN": 10,
        "REGIME_VOL_THRESHOLD_TYPE": "absolute",
        "REGIME_VOL_THRESHOLD_VALUE": 0.001,
        "REGIME_VOL_MODE": "avoid_high_vol",
        "REGIME_TREND_WINDOW_MIN": 10,
        "REGIME_TREND_METHOD": "slope",
        "REGIME_TREND_THRESHOLD_TYPE": "absolute",
        "REGIME_TREND_THRESHOLD_VALUE": 0.001,
        "REGIME_TREND_MODE": "avoid_high_trend",
        "REGIME_SHOCK_WINDOW_MIN": 5,
        "REGIME_SHOCK_SIGMA_THRESHOLD": 3.0,
        "REGIME_SHOCK_COOLDOWN_MIN": 3,
    }

    regime_flags = detect_mean_reversion_regime(df, params)

    assert regime_flags.any()
    assert (~regime_flags).any()
