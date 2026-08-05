"""Drift alerting and the retrain trigger.

The trigger deliberately requires N *consecutive* drifted windows rather than
firing on the first one. A single window crosses the threshold regularly for
reasons that are not model degradation — a batch of traffic from one merchant,
a deploy, a quiet hour where the sample is small. Paging on that trains people
to ignore the alert, and an ignored alert is worse than none.

Two consecutive windows at the default 60s cadence means ~2 minutes of
sustained drift before anyone is woken up. Raise DRIFT_CONSECUTIVE_WINDOWS if
your traffic is spikier; the cost of raising it is detection latency, and the
cost of lowering it is that the alert stops meaning anything.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import structlog

from common.config import (
    DRIFT_CONSECUTIVE_WINDOWS,
    DRIFT_SHARE_THRESHOLD,
    REPORTS_DIR,
    SLACK_WEBHOOK_URL,
)

log = structlog.get_logger()

STATE_FILE = "drift_state.json"


class DriftTrigger:
    """Counts consecutive drifted windows and decides when to alert.

    State is persisted so a monitor restart does not silently reset the
    counter back to zero — otherwise a crash-looping monitor could never
    accumulate enough windows to fire, which is exactly when you want it to.
    """

    def __init__(self, reports_dir: str = REPORTS_DIR,
                 threshold: float = DRIFT_SHARE_THRESHOLD,
                 required_windows: int = DRIFT_CONSECUTIVE_WINDOWS):
        self.path = os.path.join(reports_dir, STATE_FILE)
        self.threshold = threshold
        self.required_windows = required_windows
        self.consecutive = 0
        self.alerted = False
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            self.consecutive = int(data.get("consecutive", 0))
            self.alerted = bool(data.get("alerted", False))
        except (OSError, ValueError, TypeError):
            pass  # no prior state, or it is unreadable; start clean

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"consecutive": self.consecutive, "alerted": self.alerted}, f)
        os.replace(tmp, self.path)

    def observe(self, drift_share: float | None) -> dict:
        """Record one window. Returns a decision dict.

        `should_alert` is True only on the transition into the alerting state,
        so a drift condition that persists for an hour pages once rather than
        sixty times. It re-arms when drift clears.
        """
        drifted = drift_share is not None and drift_share >= self.threshold
        if drifted:
            self.consecutive += 1
        else:
            self.consecutive = 0

        should_alert = False
        recovered = False
        if self.consecutive >= self.required_windows and not self.alerted:
            should_alert = True
            self.alerted = True
        elif not drifted and self.alerted:
            recovered = True
            self.alerted = False

        self._save()
        return {
            "drifted": drifted,
            "consecutive": self.consecutive,
            "should_alert": should_alert,
            "recovered": recovered,
            "should_retrain": self.alerted,
        }


def send_slack(text: str, webhook_url: str = SLACK_WEBHOOK_URL) -> bool:
    """Post to Slack. Returns True if sent. No-ops (logged) without a webhook.

    urllib rather than requests/httpx: this is one POST of a small JSON body,
    and the service already carries enough dependencies.
    """
    if not webhook_url:
        log.warning("slack_webhook_unset", advice="set SLACK_WEBHOOK_URL to receive drift alerts")
        return False
    payload = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - fixed https webhook from config
            ok = 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as e:
        # Never let a failed alert kill the monitor loop: a broken webhook must
        # not also stop drift reports being written.
        log.warning("slack_alert_failed", error=str(e))
        return False
    if not ok:
        log.warning("slack_alert_rejected", status=resp.status)
    return ok


def format_alert(summary: dict, decision: dict, threshold: float) -> str:
    return (
        f":rotating_light: *stream-fraud-detector: drift detected*\n"
        f"`drift_share` {summary.get('drift_share')} >= {threshold} "
        f"for {decision['consecutive']} consecutive windows.\n"
        f"Rows compared: {summary.get('rows_compared')} · "
        f"current fraud rate: {summary.get('current_fraud_rate'):.4f} · "
        f"avg score: {summary.get('current_avg_score'):.4f}\n"
        f"Runbook: what to do -> RUNBOOK.md#when-drift-fires"
    )


def format_recovery(summary: dict) -> str:
    return (
        f":white_check_mark: *stream-fraud-detector: drift cleared*\n"
        f"`drift_share` back to {summary.get('drift_share')}. No action needed."
    )
