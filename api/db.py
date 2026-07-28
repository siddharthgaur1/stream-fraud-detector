"""Thin sync Postgres access via a connection pool. Simple queries only, no ORM."""
from __future__ import annotations

import json

import psycopg2
import psycopg2.pool

from common.config import POSTGRES_DSN

_pool: psycopg2.pool.SimpleConnectionPool | None = None


def init_pool(minconn: int = 1, maxconn: int = 10) -> None:
    global _pool
    _pool = psycopg2.pool.SimpleConnectionPool(minconn, maxconn, dsn=POSTGRES_DSN)


def close_pool() -> None:
    if _pool:
        _pool.closeall()


def ping() -> bool:
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        finally:
            _pool.putconn(conn)
    except psycopg2.Error:
        return False


def insert_score(row: dict) -> None:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scored_transactions
                    (transaction_id, user_id, amount, merchant_category,
                     fraud_score, is_fraud, shap_explanation)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (
                    row["transaction_id"], row["user_id"], row["amount"], row["merchant_category"],
                    row["fraud_score"], row["is_fraud"], json.dumps(row["shap_explanation"]),
                ),
            )
        conn.commit()
    finally:
        _pool.putconn(conn)


def stats_last(minutes: int = 60) -> dict:
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*), coalesce(sum(is_fraud::int), 0), coalesce(avg(fraud_score), 0)
                FROM scored_transactions
                WHERE scored_at >= now() - (%s || ' minutes')::interval
                """,
                (minutes,),
            )
            count, fraud_count, avg_score = cur.fetchone()
        return {
            "transaction_count": count,
            "fraud_count": fraud_count,
            "fraud_rate": (fraud_count / count) if count else 0.0,
            "avg_score": float(avg_score),
        }
    finally:
        _pool.putconn(conn)


def fetch_recent(limit: int = 5000) -> list[dict]:
    """Recent scored transactions for the monitor service's drift comparison."""
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT amount, fraud_score, is_fraud, scored_at
                FROM scored_transactions
                ORDER BY scored_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = ["amount", "fraud_score", "is_fraud", "scored_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        _pool.putconn(conn)
