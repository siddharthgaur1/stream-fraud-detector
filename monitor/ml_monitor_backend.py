"""Second, optional monitoring backend using the sibling "ml-monitor" package,
run alongside the existing Evidently-based backend in monitor/main.py.

ml-monitor isn't published to PyPI (pip install -e ../ml-monitor to enable
it) and stream-fraud-detector has no hard dependency on it: if it isn't
installed, run_once() below just returns None and monitor/main.py skips
writing its report, leaving the Evidently pipeline as the sole monitor.
"""
from __future__ import annotations

import structlog

import pandas as pd

log = structlog.get_logger()

try:
    from ml_monitor import DriftConfig, Monitor

    ML_MONITOR_AVAILABLE = True
except ImportError:
    ML_MONITOR_AVAILABLE = False


def run_once(reference: pd.DataFrame, current: pd.DataFrame) -> dict | None:
    """Compare `current` (recent scored transactions) against `reference`
    (the training-time sample) using ml-monitor's KS/PSI detectors on
    `amount`/`is_fraud`, and prediction drift on `fraud_score`.

    Builds a fresh in-memory Monitor per call rather than keeping one alive
    across runs -- this backend is invoked on the same fixed interval as the
    Evidently one, each time against a fresh `current` snapshot, so there's
    no rolling state to preserve between calls.
    """
    if not ML_MONITOR_AVAILABLE:
        return None

    monitor = Monitor(
        reference_data=reference[["amount", "is_fraud"]],
        reference_predictions=reference["fraud_score"].to_numpy(),
        config=DriftConfig(window_size=max(len(current), 1)),
        db_path=":memory:",
    )
    for row in current.itertuples(index=False):
        monitor.log(features={"amount": row.amount, "is_fraud": row.is_fraud}, prediction=row.fraud_score)

    report = monitor.drift_report()
    log.info(
        "ml_monitor_report_written",
        n_samples=report.n_samples,
        drifted_features=report.drifted_features(),
    )
    return report.to_dict()
