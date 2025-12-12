"""Reporting utilities to compare baseline vs ML-filtered events."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_sharpe(daily_pnl: pd.Series) -> float:
    if daily_pnl.empty or daily_pnl.std() == 0:
        return 0.0
    return float(np.sqrt(252) * daily_pnl.mean() / daily_pnl.std())


def _aggregate_pnl(labeled_events: pd.DataFrame) -> pd.Series:
    pnl = labeled_events.set_index("entry_timestamp")["r_H_net"]
    return pnl.groupby(pnl.index.normalize()).sum()


def summarize_uplift(
    labeled_events: pd.DataFrame, probs: pd.Series, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute uplift summary comparing baseline vs ML filtered events."""

    baseline = labeled_events.copy()
    ml_filtered = labeled_events.loc[probs.index[probs >= threshold]]

    baseline_daily = _aggregate_pnl(baseline)
    ml_daily = _aggregate_pnl(ml_filtered)

    summary = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "n_trades": len(baseline),
                "E_r_H_net": baseline["r_H_net"].mean(),
                "p_win": baseline["is_r_H_net_positive"].mean(),
                "q05": baseline["r_H_net"].quantile(0.05),
                "q95": baseline["r_H_net"].quantile(0.95),
                "p_loss_below_1bp": (baseline["r_H_net"] < -0.0001).mean(),
                "sharpe_daily": _safe_sharpe(baseline_daily),
            },
            {
                "variant": "ml_filtered",
                "n_trades": len(ml_filtered),
                "E_r_H_net": ml_filtered["r_H_net"].mean(),
                "p_win": ml_filtered["is_r_H_net_positive"].mean(),
                "q05": ml_filtered["r_H_net"].quantile(0.05),
                "q95": ml_filtered["r_H_net"].quantile(0.95),
                "p_loss_below_1bp": (ml_filtered["r_H_net"] < -0.0001).mean(),
                "sharpe_daily": _safe_sharpe(ml_daily),
            },
        ]
    )

    return summary, baseline_daily.to_frame("baseline"), ml_daily.to_frame("ml_filtered")


def save_predictions(
    output_dir: Path, labeled_events: pd.DataFrame, probs: pd.Series, threshold: float
) -> None:
    """Persist prediction-level outputs and uplift summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    predictions = labeled_events.copy()
    predictions["proba"] = probs
    predictions.to_csv(output_dir / "ml_predictions.csv")

    summary, baseline_daily, ml_daily = summarize_uplift(labeled_events, probs, threshold)
    summary.to_csv(output_dir / "ml_uplift_summary.csv", index=False)
    ml_daily.to_csv(output_dir / "daily_pnl_ml.csv")

    _plot_probability_histogram(predictions, plot_dir)
    _plot_roc_curve(predictions, plot_dir)
    _plot_uplift_thresholds(predictions, plot_dir)


def _plot_probability_histogram(predictions: pd.DataFrame, plot_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    predictions["proba"].hist(bins=20, alpha=0.6, label="all")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "proba_hist.png")
    plt.close()


def _plot_roc_curve(predictions: pd.DataFrame, plot_dir: Path) -> None:
    try:
        from sklearn.metrics import RocCurveDisplay
    except ImportError:  # pragma: no cover - optional dependency
        logger.warning("scikit-learn not installed; skipping ROC curve plot")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(predictions["is_r_H_net_positive"], predictions["proba"], ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(plot_dir / "roc_curve.png")
    plt.close(fig)


def _plot_uplift_thresholds(predictions: pd.DataFrame, plot_dir: Path) -> None:
    thresholds = np.linspace(0.0, 1.0, 21)
    expected_returns = []
    sharpe_values = []
    trade_counts = []

    for thr in thresholds:
        filtered = predictions.loc[predictions["proba"] >= thr]
        daily = _aggregate_pnl(filtered)
        trade_counts.append(len(filtered))
        expected_returns.append(filtered["r_H_net"].mean())
        sharpe_values.append(_safe_sharpe(daily))

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(thresholds, expected_returns, marker="o")
    axes[0].set_ylabel("E[r_H_net]")
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)

    axes[1].plot(thresholds, trade_counts, marker="o")
    axes[1].set_ylabel("N trades")

    axes[2].plot(thresholds, sharpe_values, marker="o")
    axes[2].set_ylabel("Sharpe (daily)")
    axes[2].set_xlabel("Probability threshold")

    fig.tight_layout()
    fig.savefig(plot_dir / "uplift_thresholds.png")
    plt.close(fig)
