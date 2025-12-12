"""Threshold recommendation utilities for z-score bin analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Container for recommended z-score thresholds.

    Attributes
    ----------
    mode : str
        Operating mode considered when computing the recommendation.
    recommended_z_min_short : float | None
        Minimum positive z-score threshold for fading upward moves (short side).
    recommended_z_min_long : float | None
        Minimum absolute z-score threshold on the negative side for fading downward moves (long side).
    accepted_bins : pd.DataFrame
        Bins that satisfy the configured statistical constraints.
    """

    mode: str
    recommended_z_min_short: float | None
    recommended_z_min_long: float | None
    accepted_bins: pd.DataFrame


def _bin_filter(
    bin_stats: pd.DataFrame,
    criteria: dict[str, float],
    predicate: Callable[[pd.Series], bool],
) -> pd.DataFrame:
    """Return bins that satisfy all statistical constraints for a side."""

    if bin_stats.empty:
        return bin_stats

    masks = []
    for _, row in bin_stats.iterrows():
        if not predicate(row):
            masks.append(False)
            continue
        if row["n"] < criteria["MIN_EVENTS_PER_BIN"]:
            masks.append(False)
            continue
        if row["ci_low"] < criteria["MIN_CI_LOW"]:
            masks.append(False)
            continue
        if row["E_r_H_net"] < criteria["MIN_EXPECTANCY_NET"]:
            masks.append(False)
            continue
        if row.get("p_loss_below_x", np.nan) > criteria["MAX_TAIL_LOSS"]:
            masks.append(False)
            continue
        masks.append(True)

    mask_series = pd.Series(masks, index=bin_stats.index)
    return bin_stats.loc[mask_series]


def _compute_recommended_threshold(bin_stats: pd.DataFrame, side: str) -> float | None:
    """Compute the recommended threshold from acceptable bins for a given side."""

    if bin_stats.empty:
        return None

    if side == "short":
        return float(bin_stats["z_bin_center"].min())

    negative_bins = bin_stats.copy()
    negative_bins["abs_center"] = negative_bins["z_bin_center"].abs()
    return float(negative_bins.sort_values("abs_center")["abs_center"].iloc[0])


def recommend_thresholds_from_bins(bin_stats: pd.DataFrame, params: dict[str, Any]) -> ThresholdRecommendation:
    """Derive recommended z-score thresholds based on statistical constraints.

    Parameters
    ----------
    bin_stats : pd.DataFrame
        DataFrame produced by ``compute_zscore_bin_stats`` containing z-score bin metrics.
    params : dict[str, Any]
        Parameter dictionary containing mode selection and constraint thresholds.

    Returns
    -------
    ThresholdRecommendation
        Recommended thresholds for short and long sides plus the bins that satisfy the criteria.
    """

    mode = str(params.get("MODE", "both")).lower()
    criteria = {
        "MIN_EVENTS_PER_BIN": float(params.get("MIN_EVENTS_PER_BIN", 50)),
        "MIN_CI_LOW": float(params.get("MIN_CI_LOW", 0.40)),
        "MIN_EXPECTANCY_NET": float(params.get("MIN_EXPECTANCY_NET", 0.0)),
        "MAX_TAIL_LOSS": float(params.get("MAX_TAIL_LOSS", 1.0)),
    }

    if bin_stats.empty:
        logger.warning("Bin statistics are empty; cannot compute recommended thresholds")
        return ThresholdRecommendation(mode=mode, recommended_z_min_short=None, recommended_z_min_long=None, accepted_bins=bin_stats)

    bin_stats = bin_stats.copy()
    if "z_bin_center" not in bin_stats.columns:
        bin_stats["z_bin_center"] = (bin_stats["z_bin_left"] + bin_stats["z_bin_right"]) / 2

    positive_bins = bin_stats[bin_stats["z_bin_center"] > 0]
    negative_bins = bin_stats[bin_stats["z_bin_center"] < 0]

    accepted_positive = _bin_filter(positive_bins, criteria, predicate=lambda row: True)
    accepted_negative = _bin_filter(negative_bins, criteria, predicate=lambda row: True)

    recommended_z_min_short = _compute_recommended_threshold(accepted_positive, side="short")
    recommended_z_min_long = _compute_recommended_threshold(accepted_negative, side="long")

    frontier_indices: set[int] = set()
    if not accepted_positive.empty and recommended_z_min_short is not None:
        frontier_idx = accepted_positive["z_bin_center"].idxmin()
        frontier_indices.add(frontier_idx)
        frontier_row = accepted_positive.loc[frontier_idx]
        logger.info(
            "Frontier fade_up bin z>=%.2f: n=%d p_hat=%.3f ci=[%.3f, %.3f] E_net=%.6f",
            recommended_z_min_short,
            int(frontier_row["n"]),
            float(frontier_row["p_hat"]),
            float(frontier_row["ci_low"]),
            float(frontier_row["ci_high"]),
            float(frontier_row["E_r_H_net"]),
        )

    if not accepted_negative.empty and recommended_z_min_long is not None:
        negative_with_abs = accepted_negative.assign(abs_center=accepted_negative["z_bin_center"].abs())
        frontier_idx = negative_with_abs.sort_values("abs_center").index[0]
        frontier_indices.add(frontier_idx)
        frontier_row = accepted_negative.loc[frontier_idx]
        logger.info(
            "Frontier fade_down bin |z|>=%.2f: n=%d p_hat=%.3f ci=[%.3f, %.3f] E_net=%.6f",
            recommended_z_min_long,
            int(frontier_row["n"]),
            float(frontier_row["p_hat"]),
            float(frontier_row["ci_low"]),
            float(frontier_row["ci_high"]),
            float(frontier_row["E_r_H_net"]),
        )

    if mode == "fade_up_only":
        accepted_negative = accepted_negative.iloc[0:0]
    elif mode == "fade_down_only":
        accepted_positive = accepted_positive.iloc[0:0]

    accepted_bins = pd.concat(
        [
            accepted_positive.assign(direction="fade_up"),
            accepted_negative.assign(direction="fade_down"),
        ]
    ).sort_values("z_bin_center")
    accepted_bins["is_frontier"] = accepted_bins.index.isin(frontier_indices)
    accepted_bins["recommended_z_min_short"] = recommended_z_min_short
    accepted_bins["recommended_z_min_long"] = recommended_z_min_long

    if not accepted_bins.empty:
        centers_repr = ", ".join(
            f"{row.direction}@{row.z_bin_center:.2f} (n={int(row.n)})" for row in accepted_bins.itertuples()
        )
        logger.info("Accepted bins under constraints: %s", centers_repr)

    if recommended_z_min_short is None and mode == "fade_up_only":
        logger.warning("No acceptable bins found for fade_up_only constraints")
    if recommended_z_min_long is None and mode == "fade_down_only":
        logger.warning("No acceptable bins found for fade_down_only constraints")

    frontier_info = []
    if recommended_z_min_short is not None:
        frontier_info.append(f"fade_up>= {recommended_z_min_short:.2f}")
    if recommended_z_min_long is not None:
        frontier_info.append(f"fade_down>= {recommended_z_min_long:.2f}")
    if frontier_info:
        logger.info("Recommended thresholds derived: %s", ", ".join(frontier_info))

    return ThresholdRecommendation(
        mode=mode,
        recommended_z_min_short=recommended_z_min_short,
        recommended_z_min_long=recommended_z_min_long,
        accepted_bins=accepted_bins,
    )
