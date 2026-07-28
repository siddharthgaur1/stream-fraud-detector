"""Feature engineering shared by the offline trainer and the online scorer.

Both sides call `compute_features` on the same inputs (current transaction +
a short window of the user's recent amounts + known devices) so there is no
train/serve skew. Only the *source* of the history differs: Redis online,
an in-memory replay offline.
"""
from __future__ import annotations

import statistics

MERCHANT_CATEGORIES = [
    "groceries", "electronics", "fuel", "utilities", "dining",
    "travel", "entertainment", "pharmacy", "clothing", "online_services",
]
MERCHANT_OTHER_CODE = len(MERCHANT_CATEGORIES)

FEATURE_NAMES = [
    "amount", "hour", "user_age_days",
    "avg5", "avg10", "avg30", "std10",
    "amount_ratio_avg10", "tx_count_seen", "is_new_device", "merchant_cat_code",
]


def merchant_code(category: str) -> int:
    try:
        return MERCHANT_CATEGORIES.index(category)
    except ValueError:
        return MERCHANT_OTHER_CODE


def _window_stats(amounts: list[float], n: int) -> tuple[float, float]:
    window = amounts[-n:]
    if not window:
        return 0.0, 0.0
    mean = statistics.fmean(window)
    std = statistics.pstdev(window) if len(window) > 1 else 0.0
    return mean, std


def compute_features(
    *,
    amount: float,
    hour: int,
    user_age_days: int,
    merchant_category: str,
    device_id: str,
    prior_amounts: list[float],
    prior_devices: set[str],
) -> dict[str, float]:
    """`prior_amounts`/`prior_devices` reflect history BEFORE this transaction."""
    avg5, _ = _window_stats(prior_amounts, 5)
    avg10, std10 = _window_stats(prior_amounts, 10)
    avg30, _ = _window_stats(prior_amounts, 30)
    return {
        "amount": amount,
        "hour": hour,
        "user_age_days": user_age_days,
        "avg5": avg5,
        "avg10": avg10,
        "avg30": avg30,
        "std10": std10,
        "amount_ratio_avg10": amount / (avg10 + 1e-6) if avg10 else 1.0,
        "tx_count_seen": min(len(prior_amounts), 30),
        "is_new_device": 0.0 if device_id in prior_devices else 1.0,
        "merchant_cat_code": float(merchant_code(merchant_category)),
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
