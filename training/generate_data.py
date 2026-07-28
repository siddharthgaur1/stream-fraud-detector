"""Generate a synthetic UPI/fintech transaction stream for offline training.

Produces chronological per-user sequences (so rolling features replay
correctly) with ~2% injected fraud: unusual hour + abnormally high amount +
a device never seen before for that user.
"""
from __future__ import annotations

import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta

from faker import Faker

from common.features import MERCHANT_CATEGORIES

FIELDS = [
    "transaction_id", "user_id", "amount", "merchant_category", "hour",
    "device_id", "location_hash", "user_age_days", "timestamp", "is_fraud",
]


def generate(n_users: int, tx_per_user: int, fraud_rate: float, seed: int) -> list[dict]:
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    rows: list[dict] = []
    for u in range(n_users):
        user_id = f"user_{u:05d}"
        user_age_days = random.randint(1, 2000)
        typical_amount = random.uniform(100, 3000)  # INR-ish
        devices = [str(uuid.uuid4())[:8] for _ in range(random.randint(1, 3))]
        locations = [fake.postcode() for _ in range(random.randint(1, 2))]
        start = datetime.utcnow() - timedelta(days=random.randint(30, 365))

        t = start
        for _ in range(tx_per_user):
            t += timedelta(hours=random.uniform(1, 48))
            is_fraud = random.random() < fraud_rate

            if is_fraud:
                hour = random.choice([1, 2, 3, 4])
                amount = typical_amount * random.uniform(5, 15)
                device_id = str(uuid.uuid4())[:8]  # never-seen device
            else:
                hour = t.hour
                amount = max(10.0, random.gauss(typical_amount, typical_amount * 0.3))
                device_id = random.choice(devices)

            rows.append({
                "transaction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "amount": round(amount, 2),
                "merchant_category": random.choice(MERCHANT_CATEGORIES),
                "hour": hour,
                "device_id": device_id,
                "location_hash": random.choice(locations),
                "user_age_days": user_age_days,
                "timestamp": t.replace(hour=hour).isoformat(),
                "is_fraud": int(is_fraud),
            })
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=500)
    parser.add_argument("--tx-per-user", type=int, default=80)
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/transactions.csv")
    args = parser.parse_args()

    rows = generate(args.n_users, args.tx_per_user, args.fraud_rate, args.seed)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows ({sum(r['is_fraud'] for r in rows)} fraud) to {args.out}")


if __name__ == "__main__":
    main()
