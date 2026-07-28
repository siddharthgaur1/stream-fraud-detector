"""In-memory pool of simulated users that emits one transaction at a time,
reusing the same fraud-injection pattern as training/generate_data.py so the
live stream and the offline training set share statistical shape.
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime

from faker import Faker

from common.features import MERCHANT_CATEGORIES

FRAUD_RATE = 0.02


class SyntheticUserPool:
    def __init__(self, n_users: int, seed: int | None = None):
        fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

        self.users = []
        for u in range(n_users):
            self.users.append({
                "user_id": f"user_{u:05d}",
                "user_age_days": random.randint(1, 2000),
                "typical_amount": random.uniform(100, 3000),
                "devices": [str(uuid.uuid4())[:8] for _ in range(random.randint(1, 3))],
                "locations": [fake.postcode() for _ in range(random.randint(1, 2))],
            })

    def next_transaction(self) -> dict:
        user = random.choice(self.users)
        is_fraud = random.random() < FRAUD_RATE

        if is_fraud:
            hour = random.choice([1, 2, 3, 4])
            amount = user["typical_amount"] * random.uniform(5, 15)
            device_id = str(uuid.uuid4())[:8]
        else:
            now = datetime.utcnow()
            hour = now.hour
            amount = max(10.0, random.gauss(user["typical_amount"], user["typical_amount"] * 0.3))
            device_id = random.choice(user["devices"])

        return {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user["user_id"],
            "amount": round(amount, 2),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "hour": hour,
            "device_id": device_id,
            "location_hash": random.choice(user["locations"]),
            "user_age_days": user["user_age_days"],
            "timestamp": datetime.utcnow().replace(hour=hour).isoformat(),
            "is_fraud_actual": is_fraud,
        }
