"""Redis-backed per-user rolling history: last 30 amounts + known devices.

Same shape the offline trainer replays in-memory (common/features.py), so
online features match training features exactly.
"""
from __future__ import annotations

import redis

from common.config import REDIS_URL

HISTORY_LEN = 30


class RedisFeatureStore:
    def __init__(self, url: str = REDIS_URL):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def get_history(self, user_id: str) -> tuple[list[float], set[str]]:
        # LPUSH stores newest-first; reverse to oldest-first so amounts[-n:]
        # (used by common.features) picks the n MOST RECENT values, matching
        # how the offline trainer appends+truncates its history.
        raw = self.client.lrange(f"user:{user_id}:amounts", 0, HISTORY_LEN - 1)
        amounts = [float(a) for a in reversed(raw)]
        devices = self.client.smembers(f"user:{user_id}:devices")
        return amounts, devices

    def update(self, user_id: str, amount: float, device_id: str) -> None:
        pipe = self.client.pipeline()
        pipe.lpush(f"user:{user_id}:amounts", amount)
        pipe.ltrim(f"user:{user_id}:amounts", 0, HISTORY_LEN - 1)
        pipe.sadd(f"user:{user_id}:devices", device_id)
        pipe.execute()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False
