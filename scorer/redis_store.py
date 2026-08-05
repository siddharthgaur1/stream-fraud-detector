"""Redis-backed per-user rolling history: last 30 amounts + known devices.

Same shape the offline trainer replays in-memory (common/features.py), so
online features match training features exactly.
"""
from __future__ import annotations

import redis

from common.config import REDIS_HISTORY_TTL_SEC, REDIS_URL

HISTORY_LEN = 30


class RedisFeatureStore:
    def __init__(self, url: str = REDIS_URL, ttl_sec: int = REDIS_HISTORY_TTL_SEC):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_sec = ttl_sec

    def get_history(self, user_id: str) -> tuple[list[float], set[str]]:
        # LPUSH stores newest-first; reverse to oldest-first so amounts[-n:]
        # (used by common.features) picks the n MOST RECENT values, matching
        # how the offline trainer appends+truncates its history.
        raw = self.client.lrange(f"user:{user_id}:amounts", 0, HISTORY_LEN - 1)
        amounts = [float(a) for a in reversed(raw)]
        devices = self.client.smembers(f"user:{user_id}:devices")
        return amounts, devices

    def update(self, user_id: str, amount: float, device_id: str) -> None:
        amounts_key = f"user:{user_id}:amounts"
        devices_key = f"user:{user_id}:devices"
        pipe = self.client.pipeline()
        pipe.lpush(amounts_key, amount)
        pipe.ltrim(amounts_key, 0, HISTORY_LEN - 1)
        pipe.sadd(devices_key, device_id)
        # Sliding TTL, refreshed on every transaction. The amounts list is
        # capped at HISTORY_LEN, but the *number of keys* is not: without this,
        # Redis grows with every user_id ever seen and never shrinks. Invisible
        # with 200 synthetic users; an eventual OOM at real cardinality.
        # ponytail: the devices set is still uncapped per user — fine for
        # consumer cards, revisit if one user_id can accumulate thousands.
        pipe.expire(amounts_key, self.ttl_sec)
        pipe.expire(devices_key, self.ttl_sec)
        pipe.execute()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False
