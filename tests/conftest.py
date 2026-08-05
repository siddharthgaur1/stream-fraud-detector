"""Test fixtures.

Everything here runs without Kafka, Redis or Postgres. Those are integration
concerns; the logic worth unit-testing is the feature computation, the
ensemble scoring, the auth/rate-limit path and the drift trigger, and none of
those need a broker to exercise.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Must be set before anything imports api.security, which reads it at import
# time to decide whether to fail closed.
os.environ.setdefault("API_KEYS", "test-key-one,test-key-two")


class FakeRedis:
    """In-memory stand-in for the bits of redis-py that RedisFeatureStore uses."""

    def __init__(self):
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.expiries: dict[str, int] = {}
        self.counters: dict[str, int] = {}
        self.alive = True

    # -- feature store surface
    def lrange(self, key, start, end):
        return self.lists.get(key, [])[start : end + 1]

    def smembers(self, key):
        return set(self.sets.get(key, set()))

    def pipeline(self):
        return _FakePipeline(self)

    def ping(self):
        if not self.alive:
            import redis

            raise redis.RedisError("down")
        return True

    # -- rate limiter surface
    def incr(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def expire(self, key, ttl):
        self.expiries[key] = ttl
        return True


class _FakePipeline:
    def __init__(self, store):
        self.store = store
        self.ops = []

    def lpush(self, key, value):
        self.ops.append(lambda: self.store.lists.setdefault(key, []).insert(0, str(value)))
        return self

    def ltrim(self, key, start, end):
        def _trim():
            self.store.lists[key] = self.store.lists.get(key, [])[start : end + 1]

        self.ops.append(_trim)
        return self

    def sadd(self, key, value):
        self.ops.append(lambda: self.store.sets.setdefault(key, set()).add(value))
        return self

    def incr(self, key):
        self.ops.append(lambda: self.store.incr(key))
        return self

    def expire(self, key, ttl):
        self.ops.append(lambda: self.store.expire(key, ttl))
        return self

    def execute(self):
        return [op() for op in self.ops]


@pytest.fixture
def fake_redis():
    return FakeRedis()


class StubModel:
    """Deterministic stand-in for XGBClassifier / IsolationForest.

    The real models are trained artifacts; these tests are about the wiring
    around them — ensemble weighting, thresholding, SHAP ordering — which is
    exactly the part a trained model would obscure.
    """

    def __init__(self, proba: float = 0.9):
        self.proba = proba

    def predict_proba(self, X):
        return np.array([[1 - self.proba, self.proba]] * len(X))

    def score_samples(self, X):
        return np.array([-0.5] * len(X))


class StubScaler:
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), 0.0, 1.0)


def make_bundle(proba=0.9, xgb_weight=0.7, threshold=0.7, version="test-v1"):
    from common.features import FEATURE_NAMES

    return {
        "xgb": StubModel(proba),
        "iso": StubModel(),
        "iso_scaler": StubScaler(),
        "feature_names": FEATURE_NAMES,
        "version": version,
        "xgb_weight": xgb_weight,
        "threshold": threshold,
    }
