"""Loads the trained XGBoost + IsolationForest bundle and scores a single
transaction: ensemble fraud probability + top SHAP feature contributions.
"""
from __future__ import annotations

import numpy as np
import shap
import structlog

from common import registry
from common.config import FRAUD_SCORE_THRESHOLD, XGB_WEIGHT
from common.features import FEATURE_NAMES, compute_features, features_to_vector
from scorer.redis_store import RedisFeatureStore

log = structlog.get_logger()

TOP_K_SHAP = 5


class FraudScorer:
    def __init__(self):
        # Source is the registry stage when MLflow is configured, else the
        # local joblib file. See common/registry.py.
        bundle, version, source = registry.load_bundle()
        self.version = version
        self.source = source
        self.xgb = bundle["xgb"]
        self.iso = bundle["iso"]
        self.iso_scaler = bundle["iso_scaler"]
        self.feature_names = bundle["feature_names"]
        self.xgb_weight = bundle.get("xgb_weight", XGB_WEIGHT)
        self.threshold = bundle.get("threshold", FRAUD_SCORE_THRESHOLD)
        self._shap_explainer = shap.TreeExplainer(self.xgb)
        self.store = RedisFeatureStore()
        log.info("model_loaded", version=self.version, source=source,
                 xgb_weight=self.xgb_weight, threshold=self.threshold)

    def score(self, tx: dict) -> dict:
        prior_amounts, prior_devices = self.store.get_history(tx["user_id"])
        features = compute_features(
            amount=tx["amount"],
            hour=tx["hour"],
            user_age_days=tx["user_age_days"],
            merchant_category=tx["merchant_category"],
            device_id=tx["device_id"],
            prior_amounts=prior_amounts,
            prior_devices=prior_devices,
        )
        vector = np.array([features_to_vector(features)], dtype=float)

        xgb_proba = float(self.xgb.predict_proba(vector)[0, 1])
        iso_raw = -self.iso.score_samples(vector)
        iso_score = float(self.iso_scaler.transform(iso_raw.reshape(-1, 1))[0, 0])
        # Weight and threshold come from the bundle, not the environment. They
        # were baked in at training time and the drift reference distribution
        # was computed with them; letting a deploy-time env var override them
        # silently invalidates every drift comparison against that reference.
        fraud_score = self.xgb_weight * xgb_proba + (1 - self.xgb_weight) * iso_score

        shap_values = self._shap_explainer.shap_values(vector)[0]
        ranked = sorted(zip(FEATURE_NAMES, shap_values), key=lambda kv: abs(kv[1]), reverse=True)
        shap_explanation = {name: round(float(val), 4) for name, val in ranked[:TOP_K_SHAP]}

        # record AFTER computing features, so this transaction only affects
        # rolling stats for transactions that come after it, never its own score
        self.store.update(tx["user_id"], tx["amount"], tx["device_id"])

        return {
            "fraud_score": round(fraud_score, 4),
            "is_fraud": fraud_score >= self.threshold,
            "shap_explanation": shap_explanation,
        }
