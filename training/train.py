"""Offline training: replays the synthetic CSV chronologically per-user to build
the same rolling features the online scorer computes from Redis, then fits an
XGBoost classifier + IsolationForest and saves the ensemble bundle.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from common.config import (
    FRAUD_SCORE_THRESHOLD,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    MODEL_REGISTRY_NAME,
    MODEL_VERSION,
    XGB_WEIGHT,
)
from common.features import FEATURE_NAMES, compute_features, features_to_vector


def build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    history_amounts: dict[str, list[float]] = defaultdict(list)
    history_devices: dict[str, set[str]] = defaultdict(set)

    X, y = [], []
    for row in rows:  # rows must already be in chronological order
        uid = row["user_id"]
        feats = compute_features(
            amount=float(row["amount"]),
            hour=int(row["hour"]),
            user_age_days=int(row["user_age_days"]),
            merchant_category=row["merchant_category"],
            device_id=row["device_id"],
            prior_amounts=history_amounts[uid],
            prior_devices=history_devices[uid],
        )
        X.append(features_to_vector(feats))
        y.append(int(row["is_fraud"]))

        history_amounts[uid].append(float(row["amount"]))
        history_amounts[uid] = history_amounts[uid][-30:]
        history_devices[uid].add(row["device_id"])

    return np.array(X, dtype=float), np.array(y, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--xgb-weight", type=float, default=XGB_WEIGHT)
    parser.add_argument("--threshold", type=float, default=FRAUD_SCORE_THRESHOLD)
    parser.add_argument(
        "--register", action="store_true",
        help="Log the run to MLflow and register the model. Requires MLFLOW_TRACKING_URI. "
             "Registers with no stage — promotion is a separate deliberate step, "
             "see scripts/promote_model.py.",
    )
    args = parser.parse_args()

    with open(args.data, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["timestamp"])

    X, y = build_feature_matrix(rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc", random_state=42,
    )
    xgb.fit(X_train, y_train)

    iso = IsolationForest(n_estimators=args.n_estimators, contamination=args.contamination, random_state=42)
    iso.fit(X_train)
    iso_raw = -iso.score_samples(X_train).reshape(-1, 1)  # higher = more anomalous
    iso_scaler = MinMaxScaler(clip=True).fit(iso_raw)

    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    iso_score = iso_scaler.transform(-iso.score_samples(X_test).reshape(-1, 1)).ravel()
    ensemble = args.xgb_weight * xgb_proba + (1 - args.xgb_weight) * iso_score

    metrics = {
        "xgb_auc": float(roc_auc_score(y_test, xgb_proba)),
        "ensemble_auc": float(roc_auc_score(y_test, ensemble)),
        "ensemble_avg_precision": float(average_precision_score(y_test, ensemble)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_fraud_rate": float(y_test.mean()),
    }
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(classification_report(y_test, ensemble > args.threshold))

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    bundle = {
        "xgb": xgb,
        "iso": iso,
        "iso_scaler": iso_scaler,
        "feature_names": FEATURE_NAMES,
        "version": MODEL_VERSION,
        # Baked in rather than read from the environment at serve time: the
        # drift reference below is computed with these exact values, so a
        # deploy-time override would silently invalidate every drift
        # comparison against it.
        "xgb_weight": args.xgb_weight,
        "threshold": args.threshold,
    }
    joblib.dump(bundle, args.out)
    print(f"saved model bundle to {args.out}")

    # Reference distribution for the monitor service's drift comparison
    # (same two columns it reads back from production: amount, fraud_score).
    # Written next to the model file, not MODEL_DIR — args.out may point
    # somewhere else (e.g. CI uses a relative path).
    reference_path = os.path.join(out_dir, "reference.csv")
    pd.DataFrame({
        "amount": X_test[:, FEATURE_NAMES.index("amount")],
        "fraud_score": ensemble,
        "is_fraud": y_test,
    }).to_csv(reference_path, index=False)
    print(f"saved drift reference sample to {reference_path}")

    if args.register:
        _log_to_mlflow(args, metrics, bundle_path=args.out, reference_path=reference_path)


def _log_to_mlflow(args, metrics: dict, bundle_path: str, reference_path: str) -> None:
    """Log params, metrics and artifacts, then register the model *unstaged*.

    Registering is not promoting. A new version lands with no stage and is
    inert; moving it to Production is a separate, auditable transition
    (scripts/promote_model.py). Training that auto-promotes is how a model
    nobody evaluated ends up serving traffic.
    """
    import mlflow

    if not MLFLOW_TRACKING_URI:
        raise SystemExit("--register needs MLFLOW_TRACKING_URI set")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MODEL_REGISTRY_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "contamination": args.contamination,
            "xgb_weight": args.xgb_weight,
            "threshold": args.threshold,
            "data": args.data,
            "feature_count": len(FEATURE_NAMES),
        })
        mlflow.log_metrics(metrics)
        # The bundle is logged as a plain artifact rather than via mlflow.sklearn:
        # it holds two fitted estimators plus a scaler, and the scorer wants the
        # whole thing loaded as one object. See common/registry.ARTIFACT_NAME.
        mlflow.log_artifact(bundle_path, artifact_path="model")
        mlflow.log_artifact(reference_path, artifact_path="model")
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_REGISTRY_NAME,
        )
        print(f"registered {MODEL_REGISTRY_NAME} version {result.version} (no stage)")
        print(f"promote with: python -m scripts.promote_model --version {result.version} --stage Staging")


if __name__ == "__main__":
    main()
