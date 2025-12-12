"""Plotting utilities for intraday mean reversion research."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_zscore_bin_stats


_OUTPUT_DIRNAME = "output"


def _prepare_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_return_distribution(labeled_events: pd.DataFrame, output_path: Path, by_side: bool = False) -> None:
    """Plot histogram of net returns with optional side separation.

    Parameters
    ----------
    labeled_events : pandas.DataFrame
        Labeled events containing ``r_H_net`` and ``side`` columns.
    output_path : Path
        Destination path for the figure.
    by_side : bool, optional
        Whether to plot separate distributions for longs and shorts.
    """

    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4))

    if by_side:
        for side, color in [(1, "green"), (-1, "red")]:
            subset = labeled_events.loc[labeled_events["side"] == side, "r_H_net"]
            plt.hist(subset, bins=50, alpha=0.6, label=f"side={side}", color=color)
    else:
        plt.hist(labeled_events["r_H_net"], bins=50, alpha=0.7, color="steelblue")

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Distribution of net returns")
    plt.xlabel("r_H_net")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_zscore_vs_success(
    labeled_events: pd.DataFrame,
    output_path: Path,
    bins: int = 20,
    loss_tail_x: float = 0.001,
    bin_stats: pd.DataFrame | None = None,
    recommended_threshold: float | None = None,
) -> pd.DataFrame:
    """Plot probability of success by z-score bin with confidence intervals."""

    bin_stats = (
        bin_stats
        if bin_stats is not None
        else compute_zscore_bin_stats(labeled_events, bins=bins, loss_tail_x=loss_tail_x)
    )

    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4))

    centers = (bin_stats["z_bin_left"] + bin_stats["z_bin_right"]) / 2
    widths = bin_stats["z_bin_right"] - bin_stats["z_bin_left"]

    plt.bar(centers, bin_stats["p_hat"], width=widths, align="center", color="steelblue", alpha=0.7)
    lower_error = bin_stats["p_hat"] - bin_stats["ci_low"]
    upper_error = bin_stats["ci_high"] - bin_stats["p_hat"]
    plt.errorbar(
        centers,
        bin_stats["p_hat"],
        yerr=[lower_error, upper_error],
        fmt="none",
        ecolor="black",
        capsize=3,
    )

    for x, y, n in zip(centers, bin_stats["p_hat"], bin_stats["n"]):
        plt.text(x, y + 0.02, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)

    if recommended_threshold is not None:
        plt.axvline(recommended_threshold, color="red", linestyle="--", linewidth=1.5, label="recommended")
        plt.legend()

    plt.title("P(r_H_net > 0) by z-score bin (95% CI)")
    plt.xlabel("z-score")
    plt.ylabel("Probability of success")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return bin_stats


def plot_zscore_vs_expected_return(
    labeled_events: pd.DataFrame,
    output_path: Path,
    bins: int = 20,
    loss_tail_x: float = 0.001,
    bin_stats: pd.DataFrame | None = None,
    recommended_threshold: float | None = None,
) -> pd.DataFrame:
    """Plot expected net return by z-score bin with trade counts."""

    bin_stats = (
        bin_stats
        if bin_stats is not None
        else compute_zscore_bin_stats(labeled_events, bins=bins, loss_tail_x=loss_tail_x)
    )

    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4))

    centers = (bin_stats["z_bin_left"] + bin_stats["z_bin_right"]) / 2
    widths = bin_stats["z_bin_right"] - bin_stats["z_bin_left"]

    plt.bar(centers, bin_stats["E_r_H_net"], width=widths, align="center", color="darkorange", alpha=0.7)

    for x, y, n in zip(centers, bin_stats["E_r_H_net"], bin_stats["n"]):
        plt.text(x, y, f"n={n}", ha="center", va="bottom", fontsize=7, rotation=90)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    if recommended_threshold is not None:
        plt.axvline(recommended_threshold, color="red", linestyle="--", linewidth=1.5, label="recommended")
        plt.legend()

    plt.title("E[r_H_net] by z-score bin")
    plt.xlabel("z-score")
    plt.ylabel("E[r_H_net]")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return bin_stats


def plot_heatmap_param_space(results_df: pd.DataFrame, output_path: Path, value_col: str) -> None:
    """Plot heatmap of parameter space performance."""

    required_cols = {"LOOKBACK_MINUTES", "ZSCORE_ENTRY", value_col}
    if not required_cols.issubset(results_df.columns):
        raise ValueError(f"results_df must contain columns: {required_cols}")

    pivot = results_df.pivot_table(
        index="LOOKBACK_MINUTES", columns="ZSCORE_ENTRY", values=value_col, aggfunc="mean"
    )

    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 6))
    heatmap = plt.imshow(pivot, cmap="coolwarm", aspect="auto", origin="lower")
    plt.colorbar(heatmap, label=value_col)
    plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
    plt.xlabel("ZSCORE_ENTRY")
    plt.ylabel("LOOKBACK_MINUTES")
    plt.title(f"Heatmap of {value_col}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
