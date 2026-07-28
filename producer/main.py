"""Publishes simulated transactions to Kafka/Redpanda at a fixed rate until stopped."""
from __future__ import annotations

import json
import os
import signal
import time

import structlog
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from common.config import KAFKA_BROKERS, KAFKA_TOPIC
from common.schemas import Transaction
from producer.stream import SyntheticUserPool

log = structlog.get_logger()

N_USERS = int(os.environ.get("PRODUCER_N_USERS", "200"))
RATE_PER_SEC = float(os.environ.get("PRODUCER_RATE_PER_SEC", "5"))

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("shutdown_signal_received", signum=signum)
    _shutdown = True


def connect_with_retry(retries: int = 30, delay: float = 2.0) -> KafkaProducer:
    for attempt in range(retries):
        try:
            return KafkaProducer(
                bootstrap_servers=KAFKA_BROKERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
            )
        except NoBrokersAvailable:
            log.warning("kafka_not_ready", attempt=attempt)
            time.sleep(delay)
    raise RuntimeError(f"could not connect to Kafka at {KAFKA_BROKERS}")


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    producer = connect_with_retry()
    pool = SyntheticUserPool(n_users=N_USERS)
    log.info("producer_started", topic=KAFKA_TOPIC, rate_per_sec=RATE_PER_SEC, n_users=N_USERS)

    interval = 1.0 / RATE_PER_SEC
    sent = 0
    try:
        while not _shutdown:
            tx = Transaction(**pool.next_transaction())
            producer.send(KAFKA_TOPIC, key=tx.user_id, value=tx.model_dump(mode="json"))
            sent += 1
            if sent % 100 == 0:
                log.info("producer_progress", sent=sent)
            time.sleep(interval)
    finally:
        log.info("producer_flushing", sent=sent)
        producer.flush(timeout=10)
        producer.close()
        log.info("producer_stopped")


if __name__ == "__main__":
    main()
