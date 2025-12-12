"""Meta-labeling pipeline to filter intraday mean reversion events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from research.intraday_mean_reversion.utils.ml_cv import (
    evaluate_classifier,
    generate_walk_forward_splits,
    time_series_train_test_split,
)
from research.intraday_mean_reversion.utils.ml_features import build_feature_matrix
from research.intraday_mean_reversion.utils.ml_reporting import save_predictions

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Container for CV fold metrics and predictions."""

    label: str
    metrics: Dict[str, float]
    predictions: pd.DataFrame


def _build_model(model_name: str) -> Any:
    """Create a sklearn pipeline for the requested estimator."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - guarded by tests
        raise ImportError("scikit-learn is required for ML meta-labeling") from exc

    model = model_name.lower()
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    elif model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported ML_MODEL '{model_name}'")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict[str, Any],
    proba_threshold: float,
) -> tuple[list[FoldResult], pd.DataFrame]:
    """Perform walk-forward CV and collect fold metrics/predictions."""
    folds = generate_walk_forward_splits(
        events=X,
        train_start_year=int(params.get("ML_TRAIN_START_YEAR", params.get("START_YEAR", 2018))),
        train_end_year=int(params.get("ML_TRAIN_END_YEAR", params.get("END_YEAR", 2020))),
        test_start_year=int(params.get("ML_TEST_START_YEAR", params.get("END_YEAR", 2021))),
        test_end_year=int(params.get("ML_TEST_END_YEAR", params.get("END_YEAR", 2022))),
        fold_years=int(params.get("ML_FOLD_YEARS", 1)),
        min_train_days=int(params.get("ML_MIN_TRAIN_DAYS", 200)),
        embargo_days=int(params.get("ML_EMBARGO_DAYS", 0)),
    )

    if not folds:
        logger.warning("No CV folds generated; returning empty results")
        return [], pd.DataFrame()

    fold_results: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []
    model_name = str(params.get("ML_MODEL", "logreg"))

    for split in folds:
        X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, split)
        model = _build_model(model_name)
        metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test)
        proba = model.predict_proba(X_test)[:, 1]
        pred_df = pd.DataFrame(
            {"proba": proba, "y_true": y_test, "fold": split.label}, index=X_test.index
        )
        pred_df["y_pred"] = (pred_df["proba"] >= proba_threshold).astype(int)
        fold_results.append(FoldResult(label=split.label, metrics=metrics, predictions=pred_df))
        predictions.append(pred_df)

    predictions_df = pd.concat(predictions).sort_index()
    return fold_results, predictions_df


def run_meta_labeling(
    df_bars: pd.DataFrame,
    labeled_events: pd.DataFrame,
    params: dict[str, Any],
    output_dir: Path,
) -> Tuple[list[FoldResult], pd.DataFrame]:
    """Execute meta-labeling CV pipeline and persist outputs."""

    proba_threshold = float(params.get("ML_PROBA_THRESHOLD", 0.55))
    X, y = build_feature_matrix(df_bars, labeled_events, params)
    if X.empty:
        logger.warning("Feature matrix is empty; skipping ML pipeline")
        return [], pd.DataFrame()

    fold_results, predictions = _run_cv(X, y, params, proba_threshold)
    if predictions.empty:
        logger.warning("No predictions generated from CV")
        return fold_results, predictions

    # Align predictions with event payload
    enriched_predictions = labeled_events.loc[predictions.index].copy()
    enriched_predictions["proba"] = predictions["proba"]
    save_predictions(output_dir, enriched_predictions, predictions["proba"], proba_threshold)

    return fold_results, enriched_predictions
