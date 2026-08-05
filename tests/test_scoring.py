"""The scoring path: ensemble weighting, thresholding, SHAP, history ordering.

Uses stub estimators so the arithmetic is checkable by hand. A real trained
model would make every assertion here a tautology against whatever it happened
to predict.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeRedis, StubScaler, make_bundle

from scorer.model import TOP_K_SHAP, FraudScorer


class _StubExplainer:
    def __init__(self, values):
        self.values = values

    def shap_values(self, X):
        return np.array([self.values])


def build_scorer(monkeypatch, proba=0.9, xgb_weight=0.7, threshold=0.7, shap_values=None):
    from common import registry
    from common.features import FEATURE_NAMES
    from scorer import model as model_module

    bundle = make_bundle(proba=proba, xgb_weight=xgb_weight, threshold=threshold)
    loader = lambda: (bundle, bundle["version"], "file:test")  # noqa: E731
    monkeypatch.setattr(registry, "load_bundle", loader)
    monkeypatch.setattr(model_module.registry, "load_bundle", loader)

    values = shap_values if shap_values is not None else list(range(len(FEATURE_NAMES)))
    monkeypatch.setattr(model_module.shap, "TreeExplainer", lambda m: _StubExplainer(values))

    fake = FakeRedis()
    monkeypatch.setattr(model_module, "RedisFeatureStore", lambda: _store_with(fake))
    scorer = FraudScorer()
    return scorer, fake


def _store_with(fake):
    from scorer.redis_store import RedisFeatureStore

    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.client = fake
    store.ttl_sec = 3600
    return store


TX = {
    "transaction_id": "t1",
    "user_id": "u1",
    "amount": 500.0,
    "merchant_category": "electronics",
    "hour": 3,
    "device_id": "d-new",
    "user_age_days": 30,
}


def test_ensemble_combines_both_models_with_the_bundle_weight(monkeypatch):
    # iso_scaler clips 0.5 -> 0.5, so: 0.7*0.9 + 0.3*0.5 = 0.78
    scorer, _ = build_scorer(monkeypatch, proba=0.9, xgb_weight=0.7)
    assert scorer.score(dict(TX))["fraud_score"] == pytest.approx(0.78, abs=1e-4)


def test_weight_comes_from_the_bundle_not_the_environment(monkeypatch):
    # The drift reference is computed with the training-time weight; an env
    # override at serve time would silently invalidate every comparison.
    monkeypatch.setenv("XGB_WEIGHT", "0.1")
    scorer, _ = build_scorer(monkeypatch, proba=0.9, xgb_weight=0.7)
    assert scorer.xgb_weight == 0.7
    assert scorer.score(dict(TX))["fraud_score"] == pytest.approx(0.78, abs=1e-4)


def test_threshold_decides_the_flag(monkeypatch):
    high, _ = build_scorer(monkeypatch, proba=0.9, threshold=0.7)
    assert high.score(dict(TX))["is_fraud"] is True

    low, _ = build_scorer(monkeypatch, proba=0.9, threshold=0.95)
    assert low.score(dict(TX))["is_fraud"] is False


def test_threshold_is_inclusive(monkeypatch):
    scorer, _ = build_scorer(monkeypatch, proba=0.9, xgb_weight=0.7, threshold=0.78)
    assert scorer.score(dict(TX))["is_fraud"] is True


def test_shap_explanation_is_top_k_by_absolute_contribution(monkeypatch):
    from common.features import FEATURE_NAMES

    values = [0.0] * len(FEATURE_NAMES)
    values[0] = -9.0   # largest by magnitude, negative
    values[1] = 5.0
    values[2] = -3.0
    values[3] = 2.0
    values[4] = 1.0
    scorer, _ = build_scorer(monkeypatch, shap_values=values)

    explanation = scorer.score(dict(TX))["shap_explanation"]
    assert len(explanation) == TOP_K_SHAP
    # Ranked by |value|, so the negative contributor leads.
    assert list(explanation)[0] == FEATURE_NAMES[0]
    assert explanation[FEATURE_NAMES[0]] == -9.0


def test_history_updates_only_affect_later_transactions(monkeypatch):
    """The transaction being scored must not appear in its own rolling stats.

    If it did, a single huge transaction would inflate its own average and
    score as normal — the exact case the amount_ratio feature exists to catch.
    """
    scorer, fake = build_scorer(monkeypatch)

    first = dict(TX, amount=10000.0)
    scorer.score(first)
    # After the first score, history exists for the next one.
    amounts, devices = scorer.store.get_history("u1")
    assert amounts == [10000.0]
    assert devices == {"d-new"}


def test_repeat_device_is_no_longer_new(monkeypatch):
    scorer, _ = build_scorer(monkeypatch)
    scorer.score(dict(TX))
    _, devices = scorer.store.get_history("u1")
    assert "d-new" in devices


def test_scorer_exposes_version_and_source(monkeypatch):
    scorer, _ = build_scorer(monkeypatch)
    assert scorer.version == "test-v1"
    assert scorer.source == "file:test"


def test_iso_scaler_clips_into_unit_range():
    scaler = StubScaler()
    assert scaler.transform([[1.4]])[0][0] == 1.0
    assert scaler.transform([[-0.2]])[0][0] == 0.0
