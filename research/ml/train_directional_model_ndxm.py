"""Train a baseline directional classifier for NDXm bars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.paths import DATA_DIR, MODELS_DIR, REPORTS_DIR, ensure_directories_exist


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline directional model for NDXm")
    parser.add_argument("--dataset-file", type=str, required=True, help="Path to parquet dataset")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion for test split",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    ensure_directories_exist()

    dataset_path = Path(args.dataset_file)
    if not dataset_path.is_absolute():
        dataset_path = DATA_DIR / dataset_path
    dataset_path = dataset_path.resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_parquet(dataset_path).sort_index()

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")

    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        shuffle=False,  # preserve temporal ordering
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        subsample=0.8,
        random_state=args.random_state,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(json.dumps(report, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "ndxm_directional_v1.joblib"
    scaler_path = MODELS_DIR / "ndxm_directional_v1_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metrics_dir = REPORTS_DIR / "research" / "ml"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics_ndxm_directional_v1.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
