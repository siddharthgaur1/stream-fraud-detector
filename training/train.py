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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

from common.config import MODEL_DIR, MODEL_PATH, MODEL_VERSION
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
    args = parser.parse_args()

    with open(args.data, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["timestamp"])

    X, y = build_feature_matrix(rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="auc", random_state=42,
    )
    xgb.fit(X_train, y_train)

    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
    iso.fit(X_train)
    iso_raw = -iso.score_samples(X_train).reshape(-1, 1)  # higher = more anomalous
    iso_scaler = MinMaxScaler(clip=True).fit(iso_raw)

    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    iso_score = iso_scaler.transform(-iso.score_samples(X_test).reshape(-1, 1)).ravel()
    ensemble = 0.7 * xgb_proba + 0.3 * iso_score

    print("XGBoost AUC:", roc_auc_score(y_test, xgb_proba))
    print("Ensemble AUC:", roc_auc_score(y_test, ensemble))
    print(classification_report(y_test, ensemble > 0.7))

    joblib.dump({
        "xgb": xgb,
        "iso": iso,
        "iso_scaler": iso_scaler,
        "feature_names": FEATURE_NAMES,
        "version": MODEL_VERSION,
    }, args.out)
    print(f"saved model bundle to {args.out}")

    # Reference distribution for the monitor service's drift comparison
    # (same two columns it reads back from production: amount, fraud_score).
    reference_path = os.path.join(MODEL_DIR, "reference.csv")
    pd.DataFrame({
        "amount": X_test[:, FEATURE_NAMES.index("amount")],
        "fraud_score": ensemble,
        "is_fraud": y_test,
    }).to_csv(reference_path, index=False)
    print(f"saved drift reference sample to {reference_path}")


if __name__ == "__main__":
    main()
