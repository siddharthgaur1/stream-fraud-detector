"""Periodically compares production-scored transactions against the training
reference sample with Evidently and writes latest.html/.json to REPORTS_DIR,
which api/main.py's GET /monitoring endpoint just reads back."""
from __future__ import annotations

import json
import os
import signal
import time

import pandas as pd
import structlog
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from api import db
from common.config import MODEL_DIR, REPORTS_DIR
from monitor import alerting, ml_monitor_backend

log = structlog.get_logger()

INTERVAL_SEC = int(os.environ.get("MONITOR_INTERVAL_SEC", "60"))
MIN_ROWS = 30

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    log.info("monitor_shutdown_signal", signum=signum)
    _shutdown = True


def _atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)


def run_once(reference: pd.DataFrame, trigger: alerting.DriftTrigger | None = None) -> dict | None:
    recent = db.fetch_recent(limit=5000)
    if len(recent) < MIN_ROWS:
        # Deliberately does NOT touch the drift trigger. A quiet window is not
        # evidence that drift cleared, and resetting the counter here would let
        # a low-traffic period silently disarm an alert that was building.
        log.info("monitor_skip_insufficient_data", rows=len(recent))
        return None

    current = pd.DataFrame(recent)[["amount", "fraud_score", "is_fraud"]]
    current["is_fraud"] = current["is_fraud"].astype(int)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[["amount", "fraud_score", "is_fraud"]], current_data=current)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    _atomic_write(os.path.join(REPORTS_DIR, "latest.html"), report.get_html())

    result = report.as_dict()
    drift_metric = result["metrics"][0]["result"]
    summary = {
        "generated_at": time.time(),
        "rows_compared": len(current),
        "dataset_drift": drift_metric.get("dataset_drift"),
        "drift_share": drift_metric.get("drift_share"),
        "current_fraud_rate": float(current["is_fraud"].mean()),
        "current_avg_score": float(current["fraud_score"].mean()),
    }
    if trigger is not None:
        decision = trigger.observe(summary["drift_share"])
        summary["consecutive_drifted_windows"] = decision["consecutive"]
        summary["retrain_recommended"] = decision["should_retrain"]
        if decision["should_alert"]:
            log.warning("drift_alert", **decision)
            alerting.send_slack(alerting.format_alert(summary, decision, trigger.threshold))
        elif decision["recovered"]:
            log.info("drift_cleared", **decision)
            alerting.send_slack(alerting.format_recovery(summary))

    _atomic_write(os.path.join(REPORTS_DIR, "latest.json"), json.dumps(summary))
    log.info("monitor_report_written", **summary)

    if ml_monitor_backend.ML_MONITOR_AVAILABLE:
        try:
            ml_report = ml_monitor_backend.run_once(reference, current)
            _atomic_write(
                os.path.join(REPORTS_DIR, "ml_monitor_latest.json"), json.dumps(ml_report, default=str)
            )
        except Exception:
            log.exception("ml_monitor_report_failed")

    return summary


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    db.init_pool()
    reference = pd.read_csv(os.path.join(MODEL_DIR, "reference.csv"))
    trigger = alerting.DriftTrigger()
    log.info("monitor_started", interval_sec=INTERVAL_SEC,
             drift_threshold=trigger.threshold, required_windows=trigger.required_windows)

    while not _shutdown:
        try:
            run_once(reference, trigger)
        except Exception:
            log.exception("monitor_run_failed")
        for _ in range(INTERVAL_SEC):
            if _shutdown:
                break
            time.sleep(1)

    db.close_pool()
    log.info("monitor_stopped")


if __name__ == "__main__":
    main()
