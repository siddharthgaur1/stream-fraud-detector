"""The retrain trigger.

This is the piece that decides whether someone gets woken up, so its edge
cases are the ones that matter operationally: it must not fire on a single
noisy window, must not re-fire every minute for an hour, and must survive a
monitor restart without losing count.
"""
from __future__ import annotations

from monitor.alerting import DriftTrigger, format_alert, format_recovery, send_slack


def _trigger(tmp_path, threshold=0.5, windows=2):
    return DriftTrigger(reports_dir=str(tmp_path), threshold=threshold, required_windows=windows)


def test_single_drifted_window_does_not_alert(tmp_path):
    # One window over the line is routine. Paging on it trains people to
    # ignore the alert.
    d = _trigger(tmp_path).observe(0.9)
    assert d["drifted"] is True
    assert d["should_alert"] is False


def test_alert_fires_on_the_required_consecutive_window(tmp_path):
    t = _trigger(tmp_path, windows=2)
    assert t.observe(0.9)["should_alert"] is False
    d = t.observe(0.9)
    assert d["should_alert"] is True
    assert d["should_retrain"] is True


def test_alert_does_not_repeat_while_drift_persists(tmp_path):
    t = _trigger(tmp_path, windows=2)
    t.observe(0.9)
    assert t.observe(0.9)["should_alert"] is True
    for _ in range(10):
        d = t.observe(0.9)
        assert d["should_alert"] is False
        assert d["should_retrain"] is True   # still degraded, just not re-paging


def test_a_clean_window_resets_the_streak(tmp_path):
    t = _trigger(tmp_path, windows=3)
    t.observe(0.9)
    t.observe(0.9)
    assert t.observe(0.1)["consecutive"] == 0
    assert t.observe(0.9)["should_alert"] is False   # streak restarts, not resumes


def test_recovery_is_reported_once_and_rearms(tmp_path):
    t = _trigger(tmp_path, windows=1)
    assert t.observe(0.9)["should_alert"] is True
    recovery = t.observe(0.0)
    assert recovery["recovered"] is True
    assert recovery["should_retrain"] is False
    assert t.observe(0.0)["recovered"] is False      # only on the transition
    assert t.observe(0.9)["should_alert"] is True    # re-armed


def test_value_exactly_at_the_threshold_counts_as_drifted(tmp_path):
    assert _trigger(tmp_path, threshold=0.5).observe(0.5)["drifted"] is True


def test_missing_drift_share_is_not_treated_as_drift(tmp_path):
    # Evidently returns None when it cannot compute the share. Absence of a
    # measurement is not evidence of drift.
    assert _trigger(tmp_path).observe(None)["drifted"] is False


def test_state_survives_a_restart(tmp_path):
    """A crash-looping monitor must still be able to accumulate windows —
    otherwise the alert can never fire in exactly the situation you want it to."""
    first = _trigger(tmp_path, windows=2)
    first.observe(0.9)

    second = _trigger(tmp_path, windows=2)   # fresh object, same reports dir
    assert second.consecutive == 1
    assert second.observe(0.9)["should_alert"] is True


def test_alerted_flag_survives_a_restart_so_it_does_not_double_page(tmp_path):
    first = _trigger(tmp_path, windows=1)
    assert first.observe(0.9)["should_alert"] is True

    second = _trigger(tmp_path, windows=1)
    assert second.observe(0.9)["should_alert"] is False


def test_corrupt_state_file_starts_clean(tmp_path):
    (tmp_path / "drift_state.json").write_text("not json")
    t = _trigger(tmp_path)
    assert t.consecutive == 0


def test_send_slack_without_a_webhook_is_a_noop():
    assert send_slack("hello", webhook_url="") is False


def test_alert_message_names_the_runbook(tmp_path):
    summary = {"drift_share": 0.8, "rows_compared": 500,
               "current_fraud_rate": 0.031, "current_avg_score": 0.42}
    decision = {"consecutive": 2}
    text = format_alert(summary, decision, threshold=0.5)
    # An alert that does not say what to do is a notification, not an alert.
    assert "RUNBOOK.md#when-drift-fires" in text
    assert "0.8" in text
    assert "2 consecutive" in text


def test_recovery_message_says_no_action_needed():
    assert "No action needed" in format_recovery({"drift_share": 0.0})
