"""Runner for Study 1: detecting regimes where mean reversion should be avoided."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.intraday_mean_reversion.study_1.feature_engineering import (
    build_run_id,
    compute_intraday_features,
)
from research.intraday_mean_reversion.study_1.regime_filters import evaluate_filters
from research.intraday_mean_reversion.utils.config_loader import load_params
from research.intraday_mean_reversion.utils.costs import load_cost_model
from research.intraday_mean_reversion.utils.data_loader import load_intraday_data
from research.intraday_mean_reversion.utils.events import detect_mean_reversion_events
from research.intraday_mean_reversion.utils.labeling import label_events

logger = logging.getLogger(__name__)

_DEFAULT_FILTER_PARAMS = {
    "vol_window_min": 30,
    "vol_threshold_type": "percentile",
    "vol_threshold_value": 0.7,
    "vol_regime_mode": "avoid_high_vol",
    "trend_window_min": 60,
    "trend_strength_method": "slope",
    "trend_threshold": 0.0,
    "use_time_filter": False,
    "allowed_time_windows": "09:45-11:30,13:30-15:30",
    "shock_window_min": 5,
    "shock_sigma_threshold": 2.5,
    "shock_cooldown_min": 5,
}


_REQUIRED_OUTPUT_COLUMNS = {
    "timestamp",
    "side",
    "entry_price",
    "exit_price",
    "holding_time",
    "base_signal",
    "vol_filter",
    "trend_filter",
    "time_filter",
    "shock_filter",
    "is_filtered",
    "executed",
    "pnl_gross",
    "pnl_net",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Study 1 NO-MR regime filters")
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "study_1_params.txt",
        type=Path,
        help="Path to parameter config file",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parents[1] / "output" / "study_1",
        type=Path,
        help="Base output directory",
    )
    parser.add_argument("--run-id", default=None, help="Optional run identifier override")
    return parser.parse_args()


def resolve_params(config_path: Path | str) -> dict[str, Any]:
    """Load study parameters merging defaults and config overrides."""

    base = load_params(str(config_path))
    resolved = {**_DEFAULT_FILTER_PARAMS, **base}
    return resolved


def _annotate_returns(df: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    returns = labeled.copy()
    close = df["close"].astype(float)
    entry_prices = returns["entry_price"]
    for horizon in (1, 5, 10):
        exit_prices = close.shift(-(horizon + 1)).reindex(returns.index)
        returns[f"r_{horizon}m"] = (exit_prices / entry_prices - 1.0) * returns["side"]
    returns["pnl_gross"] = returns["r_H_raw"] * entry_prices
    returns["pnl_net"] = returns["r_H_net"] * entry_prices
    return returns


def _build_trade_log(labeled: pd.DataFrame, filters: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    combined = labeled.join(filters)
    combined["is_filtered"] = filters.any(axis=1)
    combined["base_signal"] = True
    combined["holding_time"] = int(params.get("HOLD_TIME_BARS", 1))
    combined["executed"] = ~combined["is_filtered"]
    combined = combined.rename(columns={"exit_price_raw": "exit_price"})
    log_columns = [
        "side",
        "entry_price",
        "exit_price",
        "holding_time",
        "base_signal",
        "vol_filter",
        "trend_filter",
        "time_filter",
        "shock_filter",
        "is_filtered",
        "executed",
        "pnl_gross",
        "pnl_net",
        "r_H_net",
        "r_H_raw",
        "r_1m",
        "r_5m",
        "r_10m",
    ]
    trade_log = combined[log_columns]
    trade_log.insert(0, "timestamp", combined.index)
    return trade_log.reset_index(drop=True)


def _compute_equity_curve(trade_log: pd.DataFrame) -> pd.DataFrame:
    executed = trade_log[trade_log["executed"]]
    returns = executed["r_H_net"].fillna(0.0)
    cumulative = (1 + returns).cumprod()
    equity_curve = pd.DataFrame({"timestamp": executed["timestamp"], "equity": cumulative})
    equity_curve["returns"] = returns.values
    return equity_curve


def _daily_returns(trade_log: pd.DataFrame) -> pd.Series:
    executed = trade_log[trade_log["executed"]].copy()
    if executed.empty:
        return pd.Series(dtype=float)
    executed["date"] = pd.to_datetime(executed["timestamp"]).dt.normalize()
    return executed.groupby("date")["r_H_net"].sum()


def _compute_metrics(trade_log: pd.DataFrame) -> dict[str, Any]:
    daily = _daily_returns(trade_log)
    returns = trade_log.loc[trade_log["executed"], "r_H_net"]

    sharpe = _annualized_sharpe(daily)
    equity_curve = (1 + returns.fillna(0.0)).cumprod()
    drawdown = _max_drawdown(equity_curve)
    calmar = _calmar_ratio(daily, drawdown)
    profit_factor = _profit_factor(returns)
    pct_losing_days = float((daily < 0).mean()) if not daily.empty else np.nan
    tail_metric = float(daily.quantile(0.05)) if not daily.empty else np.nan

    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "profit_factor": profit_factor,
        "%_losing_days": pct_losing_days,
        "tail_metric_p5": tail_metric,
    }


def _annualized_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return float("nan")
    std = daily_returns.std(ddof=0)
    if std in {0.0, float("nan")}:
        return float("nan")
    return float(daily_returns.mean() / std * np.sqrt(252))


def _calmar_ratio(daily_returns: pd.Series, max_drawdown: float) -> float:
    if not np.isfinite(max_drawdown) or max_drawdown >= 0:
        return float("nan")
    annual_return = daily_returns.mean() * 252 if not daily_returns.empty else float("nan")
    if not np.isfinite(annual_return):
        return float("nan")
    return float(annual_return / abs(max_drawdown))


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdowns = equity_curve / running_max - 1.0
    return float(drawdowns.min())


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(abs(gains / losses))


def _compute_regime_summary(trade_log: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    labeled = labeled.copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=[
                "bucket_id",
                "bucket_type",
                "regime",
                "n_trades",
                "avg_return_1m",
                "avg_return_5m",
                "avg_return_10m",
                "hit_rate_5m",
                "hit_rate_10m",
                "mfe_mean",
                "mae_mean",
            ]
        )
    filters_by_ts = trade_log.set_index("timestamp")["is_filtered"].reindex(labeled.index).fillna(False)
    labeled["regime"] = np.where(filters_by_ts, "no_mr", "mr")
    labeled["vol_bucket"] = _safe_qcut(labeled["realized_vol"], 5)
    labeled["trend_bucket"] = _safe_qcut(labeled["trend_strength"].abs(), 5)
    labeled["hour_bucket"] = labeled.index.hour

    buckets = []
    for bucket_col, prefix in [("hour_bucket", "hour"), ("vol_bucket", "vol_q"), ("trend_bucket", "trend_q")]:
        bucketed = _aggregate_bucket(labeled, bucket_col, prefix)
        buckets.append(bucketed)

    summary = pd.concat(buckets, ignore_index=True)
    return summary


def _aggregate_bucket(labeled: pd.DataFrame, bucket_col: str, prefix: str) -> pd.DataFrame:
    grouped = labeled.groupby([bucket_col, "regime"], dropna=False)
    records = []
    for (bucket_id, regime), group in grouped:
        record = {
            "bucket_id": f"{prefix}_{bucket_id}",
            "bucket_type": prefix,
            "regime": regime,
            "n_trades": len(group),
            "avg_return_1m": float(group.get("r_1m", pd.Series(dtype=float)).mean()),
            "avg_return_5m": float(group.get("r_5m", pd.Series(dtype=float)).mean()),
            "avg_return_10m": float(group.get("r_10m", pd.Series(dtype=float)).mean()),
            "hit_rate_5m": float((group.get("r_5m", pd.Series(dtype=float)) > 0).mean()),
            "hit_rate_10m": float((group.get("r_10m", pd.Series(dtype=float)) > 0).mean()),
            "mfe_mean": float("nan"),
            "mae_mean": float("nan"),
        }
        records.append(record)
    return pd.DataFrame(records)


def _safe_qcut(series: pd.Series, q: int) -> pd.Series:
    if series.notna().nunique() < 2:
        return pd.Series(np.nan, index=series.index)
    return pd.qcut(series, q, labels=False, duplicates="drop")


def run_study(params: dict[str, Any], output_dir: Path, df: pd.DataFrame | None = None, run_id: str | None = None) -> Path:
    """Execute Study 1 end-to-end and persist artifacts."""

    resolved_params = {**_DEFAULT_FILTER_PARAMS, **params}
    run_identifier = run_id or build_run_id(resolved_params)
    run_dir = output_dir / run_identifier
    run_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_intraday_data(
            resolved_params["SYMBOL"],
            int(resolved_params["START_YEAR"]),
            int(resolved_params["END_YEAR"]),
            resolved_params,
        )

    features = compute_intraday_features(df, resolved_params)
    events = detect_mean_reversion_events(df, resolved_params)
    cost_model = load_cost_model(resolved_params)
    labeled = label_events(df, events, resolved_params, cost_model)
    labeled = labeled.join(features[["realized_vol", "trend_strength", "shock_active"]], how="left")
    labeled = _annotate_returns(df, labeled)

    filters = evaluate_filters(features, labeled.index, resolved_params)
    trade_log = _build_trade_log(labeled, filters, resolved_params)

    regime_summary = _compute_regime_summary(trade_log, labeled)

    equity_curve = _compute_equity_curve(trade_log)
    metrics = _compute_metrics(trade_log)

    params_path = run_dir / "params.json"
    metrics_path = run_dir / "metrics.json"
    regime_path = run_dir / "regime_summary.csv"
    trade_log_path = run_dir / "trade_log.csv"
    equity_path = run_dir / "equity_curve.csv"

    params_path.write_text(json.dumps(resolved_params, indent=2, default=str))
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    regime_summary.to_csv(regime_path, index=False)
    trade_log.to_csv(trade_log_path, index=False)
    equity_curve.to_csv(equity_path, index=False)

    _validate_outputs(trade_log, regime_summary)

    return run_dir


def _validate_outputs(trade_log: pd.DataFrame, regime_summary: pd.DataFrame) -> None:
    missing_cols = _REQUIRED_OUTPUT_COLUMNS - set(trade_log.columns)
    if missing_cols:
        raise ValueError(f"Trade log missing required columns: {sorted(missing_cols)}")
    if regime_summary.empty:
        logger.warning("Regime summary is empty; check input data or filters")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = _parse_args()
    params = resolve_params(args.config)
    run_dir = run_study(params, Path(args.output_dir), run_id=args.run_id)
    logger.info("Study 1 completed. Outputs at %s", run_dir)


if __name__ == "__main__":
    main()
