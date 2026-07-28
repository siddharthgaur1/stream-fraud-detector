"""Shared pydantic data contracts used across producer, scorer, api and monitor."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    hour: int = Field(ge=0, le=23)
    device_id: str
    location_hash: str
    user_age_days: int
    timestamp: datetime
    is_fraud_actual: bool | None = None  # ground truth, only present in synthetic stream


class ScoreRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    hour: int = Field(ge=0, le=23)
    device_id: str
    location_hash: str
    user_age_days: int


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    shap_explanation: dict[str, float]
    latency_ms: float


class StatsResponse(BaseModel):
    window: str
    transaction_count: int
    fraud_count: int
    fraud_rate: float
    avg_score: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    kafka_connected: bool
    redis_connected: bool
    db_connected: bool
