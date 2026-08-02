"""Load test for POST /score against the full docker-compose stack.

Run: docker compose up -d && docker compose run --rm trainer  (see README Setup)
Then:
    locust -f loadtest/locustfile.py --host http://localhost:8000

Headless, for a fixed run (e.g. CI or a quick number for the README):
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
        --headless -u 50 -r 10 -t 60s --csv loadtest/results
"""
import random
import uuid

from locust import HttpUser, between, task

from common.features import MERCHANT_CATEGORIES

# Mirrors producer/stream.py's synthetic distribution -- see common/features.py
# and training/generate_data.py for the ranges these are drawn from.
_DEVICE_IDS = [f"device-{i:04d}" for i in range(200)] + ["brand-new-device"]
_LOCATION_HASHES = [f"5600{i:02d}" for i in range(20)]


class ScorerUser(HttpUser):
    wait_time = between(0.05, 0.3)

    @task
    def score_transaction(self):
        payload = {
            "transaction_id": f"loadtest-{uuid.uuid4().hex[:12]}",
            "user_id": f"user_{random.randint(0, 999):05d}",
            "amount": round(random.uniform(10, 50000), 2),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "hour": random.randint(0, 23),
            "device_id": random.choice(_DEVICE_IDS),
            "location_hash": random.choice(_LOCATION_HASHES),
            "user_age_days": random.randint(1, 2000),
        }
        with self.client.post("/score", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}: {resp.text[:200]}")
            elif "fraud_score" not in resp.json():
                resp.failure("response missing fraud_score")
