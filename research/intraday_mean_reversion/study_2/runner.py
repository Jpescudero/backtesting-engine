"""Runner for Study 2: intraday mean reversion half-life characterization."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research.intraday_mean_reversion.study_1.feature_engineering import build_run_id, compute_intraday_features
from research.intraday_mean_reversion.study_1.regime_filters import (
    _apply_shock_filter,
    _apply_trend_filter,
    _apply_volatility_filter,
    evaluate_filters,
)
from research.intraday_mean_reversion.study_2.half_life import (
    TimeBucket,
    compute_half_life_log,
    compute_reference_mean,
    parse_time_buckets,
    reversion_probability_by_horizon,
    summarize_by_bucket,
)
from research.intraday_mean_reversion.utils.config_loader import load_params
from research.intraday_mean_reversion.utils.costs import load_cost_model
from research.intraday_mean_reversion.utils.data_loader import load_intraday_data
from research.intraday_mean_reversion.utils.events import detect_mean_reversion_events
from research.intraday_mean_reversion.utils.labeling import label_events

logger = logging.getLogger(__name__)


_HORIZONS_MIN = (5, 10, 15, 30)

_REQUIRED_STUDY2_KEYS = {"HALF_LIFE_THRESHOLD", "MAX_LOOKAHEAD_MIN", "TIME_BUCKETS"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Study 2 half-life analysis")
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "study_2_params.txt",
        type=Path,
        help="Path to parameter config file",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "output" / "study_2",
        type=Path,
        help="Base output directory",
    )
    parser.add_argument("--run-id", default=None, help="Optional run identifier override")
    return parser.parse_args()


def resolve_params(config_path: Path | str) -> dict[str, Any]:
    """Load study parameters ensuring Study 2 requirements are present."""

    base = load_params(str(config_path))
    missing = _REQUIRED_STUDY2_KEYS - set(base)
    if missing:
        raise ValueError(f"Missing Study 2 parameters: {', '.join(sorted(missing))}")
    return base


def _regime_time_fraction(features: pd.DataFrame, params: dict[str, Any]) -> float:
    vol_filter = _apply_volatility_filter(features, params)
    trend_filter = _apply_trend_filter(features, params)
    shock_filter = _apply_shock_filter(features, params)

    time_filter = pd.Series(False, index=features.index, dtype=bool)
    if bool(params.get("use_time_filter", False)):
        from research.intraday_mean_reversion.study_1.feature_engineering import parse_allowed_time_windows

        windows = parse_allowed_time_windows(params.get("allowed_time_windows"))
        times = pd.Series(features.index.time, index=features.index)
        for start, end in windows:
            time_filter |= (times >= start) & (times <= end)
        time_filter = ~time_filter

    combined_filter = vol_filter | trend_filter | shock_filter | time_filter
    return float((~combined_filter).mean()) if len(features) else float("nan")


def _prepare_events(
    df: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    features = compute_intraday_features(df, params)
    events = detect_mean_reversion_events(df, params)
    cost_model = load_cost_model(params)
    labeled = label_events(df, events, params, cost_model)
    filters = evaluate_filters(features, labeled.index, params)
    keep_mask = ~filters.any(axis=1)
    return features, labeled.loc[keep_mask], len(labeled)


def _persist_outputs(run_dir: Path, artifacts: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, obj in artifacts.items():
        if name.endswith("json"):
            Path(run_dir, name).write_text(json.dumps(obj, indent=2, default=str))
        elif isinstance(obj, pd.DataFrame):
            obj.to_csv(Path(run_dir, name), index=False)
        else:
            raise ValueError(f"Unsupported artifact type for {name}")


def run_study(
    params: dict[str, Any], output_dir: Path, df: pd.DataFrame | None = None, run_id: str | None = None
) -> Path:
    """Execute Study 2 end-to-end and persist artifacts."""

    resolved_params = params.copy()
    run_identifier = run_id or build_run_id(resolved_params)
    run_dir = output_dir / run_identifier

    if df is None:
        df = load_intraday_data(
            resolved_params["SYMBOL"],
            int(resolved_params["START_YEAR"]),
            int(resolved_params["END_YEAR"]),
            resolved_params,
        )

    features, events, total_events = _prepare_events(df, resolved_params)
    if events.empty:
        logger.warning("No events available after regime filters; outputs will be empty")

    time_buckets: list[TimeBucket] = parse_time_buckets(resolved_params["TIME_BUCKETS"])
    reference_mean = compute_reference_mean(df["close"], int(resolved_params["LOOKBACK_MINUTES"]))

    event_log = compute_half_life_log(events, df["close"], reference_mean, resolved_params, time_buckets)
    bucket_summary = summarize_by_bucket(event_log)
    reversion_probs = reversion_probability_by_horizon(event_log, _HORIZONS_MIN)

    regime_context = {
        "percent_time_in_mr_regime": _regime_time_fraction(features, resolved_params),
        "total_events_detected": int(total_events),
        "events_analyzed": int(len(event_log)),
        "half_life_definition": {
            "threshold_fraction": resolved_params["HALF_LIFE_THRESHOLD"],
            "max_lookahead_min": resolved_params["MAX_LOOKAHEAD_MIN"],
            "reference_mean": "rolling_mean(close, LOOKBACK_MINUTES) frozen at t0",
        },
        "time_buckets": [bucket.__dict__ for bucket in time_buckets],
    }

    artifacts = {
        "params.json": resolved_params,
        "regime_context.json": regime_context,
        "event_half_life_log.csv": event_log,
        "half_life_by_time_bucket.csv": bucket_summary,
        "reversion_probability_by_horizon.csv": reversion_probs,
    }

    _persist_outputs(run_dir, artifacts)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = _parse_args()
    params = resolve_params(args.config)
    run_dir = run_study(params, Path(args.output_dir), run_id=args.run_id)
    logger.info("Study 2 completed. Outputs at %s", run_dir)


if __name__ == "__main__":
    main()
