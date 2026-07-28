"""Self-check for the feature engineering used by both training and scoring.
Run: python tests/test_features.py
"""
from common.features import FEATURE_NAMES, compute_features, features_to_vector, merchant_code


def test_new_device_and_amount_ratio():
    feats = compute_features(
        amount=1000.0, hour=2, user_age_days=30, merchant_category="fuel",
        device_id="new-device", prior_amounts=[100.0, 100.0, 100.0], prior_devices={"old-device"},
    )
    assert feats["is_new_device"] == 1.0
    assert feats["avg10"] == 100.0
    assert abs(feats["amount_ratio_avg10"] - 10.0) < 1e-4  # 10x the recent average -> anomalous


def test_known_device_no_history():
    feats = compute_features(
        amount=50.0, hour=12, user_age_days=1, merchant_category="groceries",
        device_id="d1", prior_amounts=[], prior_devices=set(),
    )
    assert feats["is_new_device"] == 1.0  # first-ever transaction: no known devices yet
    assert feats["avg10"] == 0.0
    assert feats["amount_ratio_avg10"] == 1.0  # no history -> neutral ratio, not div-by-zero


def test_vector_matches_declared_feature_names():
    feats = compute_features(
        amount=1.0, hour=0, user_age_days=1, merchant_category="unknown_category",
        device_id="d", prior_amounts=[1.0] * 40, prior_devices={"d"},
    )
    vector = features_to_vector(feats)
    assert len(vector) == len(FEATURE_NAMES)
    assert merchant_code("unknown_category") == merchant_code("also_unknown")  # both map to OTHER


if __name__ == "__main__":
    test_new_device_and_amount_ratio()
    test_known_device_no_history()
    test_vector_matches_declared_feature_names()
    print("all feature engineering checks passed")
