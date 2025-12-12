import pandas as pd
import pytest

from research.intraday_mean_reversion.utils.events import _apply_cooldown_and_reset
from research.intraday_mean_reversion.utils.metrics import proportion_ci_wilson


def test_proportion_ci_wilson_matches_reference() -> None:
    low, high = proportion_ci_wilson(k=50, n=100, alpha=0.05)
    assert pytest.approx(low, rel=1e-3) == 0.4038
    assert pytest.approx(high, rel=1e-3) == 0.5962


def test_apply_cooldown_and_reset_filters_events() -> None:
    index = pd.date_range("2020-01-01 09:00", periods=10, freq="min")
    events = pd.DataFrame({"side": [1, 1, 1]}, index=index[[1, 2, 4]])
    z_score = pd.Series([0.0, 2.0, 2.0, 0.2, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0], index=index)

    filtered = _apply_cooldown_and_reset(
        events=events, z_score=z_score, cooldown_bars=2, z_reset=0.5, full_index=index
    )

    assert list(filtered.index) == [index[1], index[4]]
