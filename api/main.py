"""FastAPI app: scoring endpoint, stats, health, metrics, monitoring. Also owns
the background Kafka->Redis feature-engineering consumer (scorer/stream_consumer.py)
so the whole "scorer" service in docker-compose is just this one process."""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Response

from api import db, metrics
from api.security import init_rate_limiter, require_api_key
from common.config import MODEL_VERSION, REPORTS_DIR
from common.schemas import HealthResponse, ScoreRequest, ScoreResponse, StatsResponse
from scorer.model import FraudScorer
from scorer.stream_consumer import start_stream_consumer_thread

log = structlog.get_logger()

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_pool()
    init_rate_limiter()
    scorer = FraudScorer()
    state["scorer"] = scorer
    metrics.MODEL_INFO.labels(version=scorer.version, source=scorer.source).set(1)

    def _note_message() -> None:
        state["last_stream_message_at"] = time.time()

    state["last_stream_message_at"] = time.time()
    state["consumer_thread"], state["consumer_stop"] = start_stream_consumer_thread(
        scorer, db.insert_score, on_message=_note_message
    )
    log.info("api_started", model_version=scorer.version, model_source=scorer.source)
    yield
    state["consumer_stop"].set()
    state["consumer_thread"].join(timeout=5)
    db.close_pool()
    log.info("api_stopped")


app = FastAPI(title="stream-fraud-detector", lifespan=lifespan)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, key_id: str = Depends(require_api_key)) -> ScoreResponse:
    start = time.perf_counter()
    try:
        result = state["scorer"].score(req.model_dump())
    except Exception:
        metrics.SCORE_REQUESTS.labels(source="http", outcome="error").inc()
        log.exception("scoring_failed", transaction_id=req.transaction_id, key_id=key_id)
        raise HTTPException(status_code=500, detail="scoring failed") from None
    latency_s = time.perf_counter() - start

    metrics.SCORE_LATENCY.labels(source="http").observe(latency_s)
    metrics.SCORE_REQUESTS.labels(source="http", outcome="ok").inc()
    if result["is_fraud"]:
        metrics.FRAUD_FLAGGED.labels(source="http").inc()

    row = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "amount": req.amount,
        "merchant_category": req.merchant_category,
        **result,
    }
    try:
        db.insert_score(row)
    except Exception:
        # The score itself is valid and the caller needs it, so this stays a
        # 200 — but it is counted, because sustained failures mean /stats and
        # drift detection are blind while scoring still looks healthy.
        metrics.DB_INSERT_FAILURES.inc()
        log.exception("db_insert_failed", transaction_id=req.transaction_id)

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_score=result["fraud_score"],
        is_fraud=result["is_fraud"],
        shap_explanation=result["shap_explanation"],
        latency_ms=round(latency_s * 1000, 2),
    )


@app.get("/stats", response_model=StatsResponse)
def stats(key_id: str = Depends(require_api_key)) -> StatsResponse:
    s = db.stats_last(minutes=60)
    return StatsResponse(window="1h", **s)


def _health_payload() -> tuple[HealthResponse, bool]:
    scorer = state.get("scorer")
    thread = state.get("consumer_thread")
    payload = HealthResponse(
        status="ok",
        model_version=getattr(scorer, "version", MODEL_VERSION),
        kafka_connected=thread is not None and thread.is_alive(),
        redis_connected=scorer.store.ping() if scorer else False,
        db_connected=db.ping(),
    )
    # Redis and the DB are hard dependencies of scoring: without Redis the
    # rolling features silently degrade to "no history", which is worse than
    # refusing traffic. Kafka is not — POST /score still works with the stream
    # consumer down, so it is reported but does not mark the service unhealthy.
    healthy = payload.redis_connected and payload.db_connected
    if not healthy:
        payload.status = "degraded"
    return payload, healthy


@app.get("/health", response_model=HealthResponse)
def health(response: Response) -> HealthResponse:
    """Unauthenticated on purpose: load balancers and orchestrators probe it,
    and it exposes no transaction data."""
    payload, healthy = _health_payload()
    if not healthy:
        # 200-on-everything makes this endpoint useless to an orchestrator: it
        # can never take a broken replica out of rotation.
        response.status_code = 503
    return payload


@app.get("/metrics")
def prometheus_metrics() -> Response:
    """Unauthenticated: scrapers rarely carry custom headers, and this exposes
    only aggregate counters. Keep it off the public listener in a real deploy."""
    thread = state.get("consumer_thread")
    if thread is not None:
        metrics.STREAM_LAG_SECONDS.set(
            time.time() - state.get("last_stream_message_at", time.time())
        )
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/monitoring")
def monitoring(format: str = "json", key_id: str = Depends(require_api_key)):
    if format == "html":
        path = os.path.join(REPORTS_DIR, "latest.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="no report generated yet")
        with open(path, encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/html")

    path = os.path.join(REPORTS_DIR, "latest.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no report generated yet")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/monitoring/ml-monitor")
def monitoring_ml_monitor(key_id: str = Depends(require_api_key)):
    """Second monitoring backend (sibling ml-monitor package), same report
    cadence as /monitoring but its own per-feature KS/PSI + prediction-drift
    view. 404 if ml-monitor isn't installed or hasn't produced a report yet."""
    path = os.path.join(REPORTS_DIR, "ml_monitor_latest.json")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail="no ml-monitor report generated yet (is ml-monitor installed?)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
