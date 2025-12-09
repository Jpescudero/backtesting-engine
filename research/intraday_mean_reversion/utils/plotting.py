"""Plotting utilities for intraday mean reversion research."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_zscore_vs_success(labeled_events: pd.DataFrame, output_path: Path, bins: int = 20) -> None:
    """Plot probability of success by z-score bin."""

    _prepare_output_path(output_path)
    plt.figure(figsize=(8, 4))

    z_scores = labeled_events["z_score"]
    success = labeled_events["is_r_H_net_positive"].astype(float)
    bin_edges = np.linspace(z_scores.min(), z_scores.max(), bins + 1)

    binned = pd.cut(z_scores, bins=bin_edges, include_lowest=True)
    success_by_bin = success.groupby(binned, observed=False).mean().fillna(0)
    centers = success_by_bin.index.map(lambda interval: interval.mid)
    widths = bin_edges[1:] - bin_edges[:-1]

    plt.bar(centers, success_by_bin.values, width=widths, align="center")

    plt.title("P(r_H_net > 0) by z-score bin")
    plt.xlabel("z-score")
    plt.ylabel("Probability of success")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
