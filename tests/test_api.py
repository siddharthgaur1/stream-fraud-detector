"""HTTP surface, with Kafka/Redis/Postgres replaced by fakes.

These are the assertions an operator depends on: /health has to be able to say
"take me out of rotation", /score has to reject an unauthenticated caller, and
/metrics has to keep working when everything else is broken.
"""
from __future__ import annotations

import pytest
from conftest import FakeRedis
from fastapi.testclient import TestClient

pytest.importorskip("httpx", reason="TestClient needs httpx")

VALID_KEY = "test-key-one"

PAYLOAD = {
    "transaction_id": "t-1",
    "user_id": "u1",
    "amount": 480.0,
    "merchant_category": "electronics",
    "hour": 3,
    "device_id": "d-new",
    "location_hash": "560001",
    "user_age_days": 30,
}


class _FakeScorer:
    version = "test-v1"
    source = "file:test"

    def __init__(self, redis_ok=True):
        self.store = _FakeStore(redis_ok)
        self.calls = []

    def score(self, tx):
        self.calls.append(tx)
        return {"fraud_score": 0.91, "is_fraud": True, "shap_explanation": {"amount": 1.2}}


class _FakeStore:
    def __init__(self, ok):
        self.ok = ok

    def ping(self):
        return self.ok


@pytest.fixture
def client(monkeypatch):
    """Builds the app with every external dependency stubbed out."""
    import api.main as main
    import api.security as security

    inserted = []
    monkeypatch.setattr(main.db, "init_pool", lambda: None)
    monkeypatch.setattr(main.db, "close_pool", lambda: None)
    monkeypatch.setattr(main.db, "ping", lambda: True)
    monkeypatch.setattr(main.db, "insert_score", lambda row: inserted.append(row))
    monkeypatch.setattr(
        main.db, "stats_last",
        lambda minutes=60: {"transaction_count": 10, "fraud_count": 2,
                            "fraud_rate": 0.2, "avg_score": 0.35},
    )

    scorer = _FakeScorer()
    monkeypatch.setattr(main, "FraudScorer", lambda: scorer)

    class _Thread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    class _Stop:
        def set(self):
            return None

    monkeypatch.setattr(
        main, "start_stream_consumer_thread",
        lambda *a, **kw: (_Thread(), _Stop()),
    )

    security.init_rate_limiter()
    security._limiter.client = FakeRedis()
    security._limiter.limit = 1000

    with TestClient(main.app) as c:
        c.scorer = scorer
        c.inserted = inserted
        yield c


# -- auth --------------------------------------------------------------------


def test_score_requires_a_key(client):
    assert client.post("/score", json=PAYLOAD).status_code == 401


def test_score_rejects_a_wrong_key(client):
    r = client.post("/score", json=PAYLOAD, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_score_accepts_a_valid_key(client):
    r = client.post("/score", json=PAYLOAD, headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["fraud_score"] == 0.91
    assert body["is_fraud"] is True
    assert body["latency_ms"] >= 0


def test_stats_requires_a_key(client):
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"X-API-Key": VALID_KEY}).status_code == 200


def test_monitoring_requires_a_key(client):
    assert client.get("/monitoring").status_code == 401


# -- health ------------------------------------------------------------------


def test_health_is_unauthenticated(client):
    # Load balancers do not send custom headers.
    assert client.get("/health").status_code == 200


def test_healthy_stack_reports_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
    assert body["redis_connected"] is True
    assert body["model_version"] == "test-v1"


def test_health_returns_503_when_the_database_is_down(client, monkeypatch):
    """200-on-everything makes the endpoint useless: an orchestrator can never
    take a broken replica out of rotation."""
    import api.main as main

    monkeypatch.setattr(main.db, "ping", lambda: False)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"


def test_health_returns_503_when_redis_is_down(client):
    client.scorer.store.ok = False
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["redis_connected"] is False


def test_kafka_down_is_reported_but_not_fatal(client, monkeypatch):
    """POST /score works with the stream consumer dead, so Kafka being down
    must not pull the whole replica out of rotation."""
    import api.main as main

    class _Dead:
        def is_alive(self):
            return False

        def join(self, timeout=None):  # the app still joins it on shutdown
            return None

    monkeypatch.setitem(main.state, "consumer_thread", _Dead())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["kafka_connected"] is False


# -- failure handling --------------------------------------------------------


def test_scoring_failure_is_a_500_not_a_crash(client, monkeypatch):
    def boom(tx):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(client.scorer, "score", boom)
    r = client.post("/score", json=PAYLOAD, headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 500


def test_db_failure_still_returns_the_score(client, monkeypatch):
    """The caller needs the score and it is valid; losing the audit row is a
    separate problem, counted in fraud_db_insert_failures_total."""
    import api.main as main

    def boom(row):
        raise RuntimeError("db gone")

    monkeypatch.setattr(main.db, "insert_score", boom)
    r = client.post("/score", json=PAYLOAD, headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 200
    assert r.json()["fraud_score"] == 0.91


def test_invalid_payload_is_a_422(client):
    bad = dict(PAYLOAD, hour=99)  # schema constrains hour to 0-23
    r = client.post("/score", json=bad, headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 422


# -- metrics -----------------------------------------------------------------


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post("/score", json=PAYLOAD, headers={"X-API-Key": VALID_KEY})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    assert "fraud_score_requests_total" in body
    assert "fraud_score_latency_seconds" in body
    assert "fraud_model_info" in body


def test_metrics_record_a_flagged_transaction(client):
    client.post("/score", json=PAYLOAD, headers={"X-API-Key": VALID_KEY})
    assert 'fraud_flagged_total{source="http"}' in client.get("/metrics").text


def test_monitoring_404s_before_any_report_exists(client, monkeypatch, tmp_path):
    import api.main as main

    monkeypatch.setattr(main, "REPORTS_DIR", str(tmp_path))
    r = client.get("/monitoring", headers={"X-API-Key": VALID_KEY})
    assert r.status_code == 404
