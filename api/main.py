"""FastAPI app: scoring endpoint, stats, health, monitoring. Also owns the
background Kafka->Redis feature-engineering consumer (see scorer/feature_consumer.py)
so the whole "scorer" service in docker-compose is just this one process."""
from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Response

from api import db
from common.config import MODEL_VERSION, REPORTS_DIR
from common.schemas import HealthResponse, ScoreRequest, ScoreResponse, StatsResponse
from scorer.model import FraudScorer
from scorer.stream_consumer import start_stream_consumer_thread

log = structlog.get_logger()

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_pool()
    state["scorer"] = FraudScorer()
    state["consumer_thread"], state["consumer_stop"] = start_stream_consumer_thread(
        state["scorer"], db.insert_score
    )
    log.info("api_started")
    yield
    state["consumer_stop"].set()
    state["consumer_thread"].join(timeout=5)
    db.close_pool()
    log.info("api_stopped")


app = FastAPI(title="stream-fraud-detector", lifespan=lifespan)


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    try:
        result = state["scorer"].score(req.model_dump())
    except Exception:
        log.exception("scoring_failed", transaction_id=req.transaction_id)
        raise HTTPException(status_code=500, detail="scoring failed")
    latency_ms = (time.perf_counter() - start) * 1000

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
        log.exception("db_insert_failed", transaction_id=req.transaction_id)

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_score=result["fraud_score"],
        is_fraud=result["is_fraud"],
        shap_explanation=result["shap_explanation"],
        latency_ms=round(latency_ms, 2),
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    s = db.stats_last(minutes=60)
    return StatsResponse(window="1h", **s)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        kafka_connected=state.get("consumer_thread") is not None and state["consumer_thread"].is_alive(),
        redis_connected=state["scorer"].store.ping() if "scorer" in state else False,
        db_connected=db.ping(),
    )


@app.get("/monitoring")
def monitoring(format: str = "json"):
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
def monitoring_ml_monitor():
    """Second monitoring backend (sibling ml-monitor package), same report
    cadence as /monitoring but its own per-feature KS/PSI + prediction-drift
    view. 404 if ml-monitor isn't installed or hasn't produced a report yet."""
    path = os.path.join(REPORTS_DIR, "ml_monitor_latest.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="no ml-monitor report generated yet (is ml-monitor installed?)")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
