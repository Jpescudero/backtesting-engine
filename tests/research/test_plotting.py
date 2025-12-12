"""Tests for plotting utilities in intraday mean reversion research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from research.intraday_mean_reversion.utils import plotting


@pytest.fixture()
def bin_stats_negative_error() -> pd.DataFrame:
    """Build bin statistics where confidence bounds invert around p_hat.

    The Wilson interval should normally bound ``p_hat``. This fixture crafts a
    DataFrame with deliberately inverted bounds to ensure the plotting function
    clips negative error bars before passing them to Matplotlib.
    """

    return pd.DataFrame(
        {
            "z_bin_left": [-1.0],
            "z_bin_right": [1.0],
            "p_hat": [0.2],
            "ci_low": [0.3],  # higher than p_hat -> negative lower error
            "ci_high": [0.1],  # lower than p_hat -> negative upper error
            "n": [10],
        }
    )


def test_plot_zscore_vs_success_clips_negative_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bin_stats_negative_error: pd.DataFrame
) -> None:
    """Ensure negative confidence interval deltas are clipped to zero for Matplotlib."""

    recorded: dict[str, Any] = {}

    def _mock_errorbar(x: Any, y: Any, yerr: Any, fmt: str, ecolor: str, capsize: int) -> None:
        recorded["yerr"] = yerr

    monkeypatch.setattr(plotting.plt, "errorbar", _mock_errorbar)

    output_path = tmp_path / "figure.png"

    plotting.plot_zscore_vs_success(
        labeled_events=pd.DataFrame(),
        output_path=output_path,
        bin_stats=bin_stats_negative_error,
    )

    assert output_path.exists(), "Plotting should produce an output file."
    assert "yerr" in recorded, "Matplotlib error bars should be invoked."
    lower_err, upper_err = recorded["yerr"]
    assert np.all(np.asarray(lower_err) >= 0), "Lower errors must be non-negative."
    assert np.all(np.asarray(upper_err) >= 0), "Upper errors must be non-negative."
