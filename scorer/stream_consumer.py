"""Background Kafka consumer that makes the pipeline actually real-time: every
transaction the producer publishes gets scored (scorer/model.py, same code
path as POST /score) and persisted to Postgres as it arrives, instead of only
scoring transactions a client explicitly POSTs. Runs as a daemon thread inside
the api/scorer process (see api/main.py startup).
"""
from __future__ import annotations

import json
import threading
import time

import structlog
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

from common.config import KAFKA_BROKERS, KAFKA_TOPIC

log = structlog.get_logger()


def _connect(retries: int = 30, delay: float = 2.0) -> KafkaConsumer:
    for attempt in range(retries):
        try:
            return KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BROKERS,
                group_id="stream-scorer",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
            )
        except NoBrokersAvailable:
            log.warning("kafka_not_ready_stream_consumer", attempt=attempt)
            time.sleep(delay)
    raise RuntimeError(f"could not connect to Kafka at {KAFKA_BROKERS}")


def run_stream_consumer(scorer, insert_score, stop_event: threading.Event) -> None:
    consumer = _connect()
    log.info("stream_consumer_started", topic=KAFKA_TOPIC)
    try:
        while not stop_event.is_set():
            batch = consumer.poll(timeout_ms=1000)
            for records in batch.values():
                for record in records:
                    tx = record.value
                    try:
                        result = scorer.score(tx)
                        insert_score({
                            "transaction_id": tx["transaction_id"],
                            "user_id": tx["user_id"],
                            "amount": tx["amount"],
                            "merchant_category": tx["merchant_category"],
                            **result,
                        })
                    except Exception:
                        log.exception("stream_scoring_failed", transaction_id=tx.get("transaction_id"))
    finally:
        consumer.close()
        log.info("stream_consumer_stopped")


def start_stream_consumer_thread(scorer, insert_score) -> tuple[threading.Thread, threading.Event]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_stream_consumer, args=(scorer, insert_score, stop_event), daemon=True
    )
    thread.start()
    return thread, stop_event
