"""CLI entrypoint for intraday mean reversion research."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.intraday_mean_reversion.optimizers.grid_search import GridSearchOptimizer
from research.intraday_mean_reversion.optimizers.ml_meta_labeling import run_meta_labeling
from research.intraday_mean_reversion.utils.config_loader import load_params
from research.intraday_mean_reversion.utils.data_loader import load_intraday_data
from research.intraday_mean_reversion.utils.events import detect_mean_reversion_events
from research.intraday_mean_reversion.utils.costs import load_cost_model
from research.intraday_mean_reversion.utils.labeling import label_events
from research.intraday_mean_reversion.utils.metrics import compute_daily_pnl, compute_metrics, compute_zscore_bin_stats
from research.intraday_mean_reversion.utils.plotting import (
    plot_heatmap_param_space,
    plot_return_distribution,
    plot_zscore_vs_expected_return,
    plot_zscore_vs_success,
)
from research.intraday_mean_reversion.utils.thresholding import recommend_thresholds_from_bins

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intraday mean reversion research runner")
    parser.add_argument(
        "--params-file",
        default="intraday_mean_reversion_research_params.txt",
        help="Path to parameter file",
    )
    parser.add_argument("--run-grid-search", action="store_true", help="Run grid search instead of single run")
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).parent / "output",
        type=Path,
        help="Output directory for artifacts",
    )
    parser.add_argument("--run-ml", action="store_true", help="Execute ML meta-labeling pipeline")
    parser.add_argument("--ml-only", action="store_true", help="Run only ML pipeline after labeling")
    parser.add_argument(
        "--ml-proba-threshold",
        type=float,
        default=None,
        help="Override ML probability threshold for filtering",
    )
    return parser.parse_args()


def _run_single(
    df: pd.DataFrame, base_params: dict[str, Any], output_dir: Path, run_ml: bool = False
) -> None:
    logger.info("Detecting events...")
    events = detect_mean_reversion_events(df, base_params)
    cost_model = load_cost_model(base_params)
    logger.info("Labeling events...")
    labeled = label_events(df, events, base_params, cost_model)
    logger.info("Computing metrics...")
    metrics = compute_metrics(labeled)
    daily_pnl = compute_daily_pnl(labeled)
    loss_tail_x = float(base_params.get("LOSS_TAIL_X", 0.001))
    z_bins = int(base_params.get("Z_BINNING_NBINS", 20))

    output_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_dir / "labeled_events.csv")
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    daily_pnl.to_csv(output_dir / "daily_pnl.csv", index=False)

    plot_return_distribution(labeled, output_dir / "return_distribution_gross.png", by_side=True, return_col="r_H_raw")
    plot_return_distribution(labeled, output_dir / "return_distribution_net.png", by_side=True, return_col="r_H_net")
    bin_stats = compute_zscore_bin_stats(labeled, bins=z_bins, loss_tail_x=loss_tail_x)
    recommendation = recommend_thresholds_from_bins(bin_stats, base_params)
    logger.info(
        "Recommended thresholds (mode=%s): Z_MIN_SHORT=%s | Z_MIN_LONG=%s",
        recommendation.mode,
        recommendation.recommended_z_min_short,
        recommendation.recommended_z_min_long,
    )

    selected_threshold = None
    if recommendation.mode == "fade_up_only":
        selected_threshold = recommendation.recommended_z_min_short
    elif recommendation.mode == "fade_down_only":
        selected_threshold = -recommendation.recommended_z_min_long if recommendation.recommended_z_min_long else None

    bin_stats.to_csv(output_dir / "zscore_bins.csv", index=False)
    recommendation.accepted_bins.to_csv(output_dir / "recommended_thresholds.csv", index=False)

    plot_zscore_vs_success(
        labeled,
        output_dir / "zscore_vs_success_net.png",
        bins=z_bins,
        loss_tail_x=loss_tail_x,
        bin_stats=bin_stats,
        recommended_threshold=selected_threshold,
        return_col="r_H_net",
    )
    plot_zscore_vs_success(
        labeled,
        output_dir / "zscore_vs_success_gross.png",
        bins=z_bins,
        loss_tail_x=loss_tail_x,
        bin_stats=bin_stats,
        recommended_threshold=selected_threshold,
        return_col="r_H_raw",
    )
    plot_zscore_vs_expected_return(
        labeled,
        output_dir / "zscore_expected_return_net.png",
        bins=z_bins,
        loss_tail_x=loss_tail_x,
        bin_stats=bin_stats,
        recommended_threshold=selected_threshold,
        return_col="r_H_net",
    )
    plot_zscore_vs_expected_return(
        labeled,
        output_dir / "zscore_expected_return_gross.png",
        bins=z_bins,
        loss_tail_x=loss_tail_x,
        bin_stats=bin_stats,
        recommended_threshold=selected_threshold,
        return_col="r_H_raw",
    )

    if run_ml:
        ml_output_dir = output_dir / "ml"
        if base_params.get("ML_PROBA_THRESHOLD_OVERRIDE") is not None:
            base_params["ML_PROBA_THRESHOLD"] = base_params["ML_PROBA_THRESHOLD_OVERRIDE"]
        run_meta_labeling(df, labeled, base_params, ml_output_dir)

    logger.info("Summary metrics: %s", metrics)


def _run_grid_search(df: pd.DataFrame, base_params: dict[str, Any], output_dir: Path) -> None:
    grid_params = {
        "LOOKBACK_MINUTES": [int(x) for x in base_params.get("GRID_LOOKBACK_MINUTES", [])],
        "ZSCORE_ENTRY": [float(x) for x in base_params.get("GRID_ZSCORE_ENTRY", [])],
        "HOLD_TIME_BARS": [int(x) for x in base_params.get("GRID_HOLD_TIME_BARS", [])],
    }

    optimizer = GridSearchOptimizer(grid_params, lambda override: _objective(df, base_params, override))
    results_df = optimizer.run()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "grid_search_results.csv", index=False)

    if results_df.empty:
        logger.warning("Grid search produced no results; check parameter grid configuration")
        return

    plot_heatmap_param_space(results_df, output_dir / "heatmap_E_r_H_net.png", value_col="E_r_H_net")

    best_idx = results_df["E_r_H_net"].idxmax()
    logger.info("Best parameters: %s", results_df.loc[best_idx].to_dict())


def _objective(df: pd.DataFrame, base_params: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    params = {**base_params, **override}
    cost_model = load_cost_model(params)
    events = detect_mean_reversion_events(df, params)
    labeled = label_events(df, events, params, cost_model)
    metrics = compute_metrics(labeled)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = _parse_args()
    logger.info("Loading parameters from %s", args.params_file)
    params = load_params(args.params_file)

    logger.info(
        "Loading data for %s from %s to %s", params["SYMBOL"], params["START_YEAR"], params["END_YEAR"]
    )
    data = load_intraday_data(params["SYMBOL"], int(params["START_YEAR"]), int(params["END_YEAR"]), params)

    output_dir = Path(args.output_dir)
    run_ml = bool(params.get("RUN_ML", False)) or args.run_ml or args.ml_only
    if args.ml_proba_threshold is not None:
        params["ML_PROBA_THRESHOLD_OVERRIDE"] = args.ml_proba_threshold

    if args.run_grid_search and not args.ml_only:
        logger.info("Running grid search...")
        _run_grid_search(data, params, output_dir)
    else:
        logger.info("Running single evaluation...")
        _run_single(data, params, output_dir, run_ml=run_ml)


if __name__ == "__main__":
    main()
