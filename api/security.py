"""API-key auth and rate limiting for the public endpoints.

Fail-closed by design: if `API_KEYS` is not set the app refuses to start
unless `ALLOW_UNAUTHENTICATED=true` is *also* set. An unauthenticated
scoring endpoint that silently works is how a demo ends up on the public
internet with no auth on it — the escape hatch has to be deliberate and
visible in the process environment, not the default.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time

import redis
import structlog
from fastapi import Header, HTTPException, Request

from common.config import RATE_LIMIT_PER_MIN, REDIS_URL

log = structlog.get_logger()


def _load_keys() -> set[str]:
    raw = os.environ.get("API_KEYS", "").strip()
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys and os.environ.get("ALLOW_UNAUTHENTICATED", "").lower() != "true":
        raise RuntimeError(
            "API_KEYS is unset. Set it to a comma-separated list of keys, or set "
            "ALLOW_UNAUTHENTICATED=true to run without auth (local development only)."
        )
    return keys


API_KEYS = _load_keys()
AUTH_DISABLED = not API_KEYS

if AUTH_DISABLED:
    log.warning("auth_disabled", reason="ALLOW_UNAUTHENTICATED=true", advice="never do this off localhost")


def _key_id(api_key: str) -> str:
    """Short stable identifier for logs and rate-limit buckets.

    Never log or bucket on the raw key: logs get shipped to places the key
    should not reach, and a Redis key name is not a secret.
    """
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class RateLimiter:
    """Fixed-window counter in Redis, so the limit holds across uvicorn workers
    and across replicas — an in-process counter would give each worker its own
    full allowance.

    ponytail: fixed window, so a client can send 2x the limit across a window
    boundary. Move to a sliding window or token bucket if that burst actually
    matters; for abuse protection on a scoring endpoint it does not.
    """

    def __init__(self, url: str = REDIS_URL, limit_per_min: int = RATE_LIMIT_PER_MIN):
        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.limit = limit_per_min

    def check(self, key_id: str) -> tuple[bool, int]:
        """Returns (allowed, seconds_until_reset)."""
        if self.limit <= 0:
            return True, 0
        window = int(time.time() // 60)
        bucket = f"ratelimit:{key_id}:{window}"
        try:
            pipe = self.client.pipeline()
            pipe.incr(bucket)
            pipe.expire(bucket, 120)
            count, _ = pipe.execute()
        except redis.RedisError:
            # Fail open on a Redis outage. Rate limiting is abuse protection,
            # not authorisation — dropping every request because the limiter is
            # down turns a degraded dependency into a full outage.
            log.warning("rate_limit_backend_unavailable", key_id=key_id)
            return True, 0
        return int(count) <= self.limit, 60 - int(time.time() % 60)


_limiter: RateLimiter | None = None


def init_rate_limiter() -> None:
    global _limiter
    _limiter = RateLimiter()


def require_api_key(request: Request, x_api_key: str = Header(default="")) -> str:
    """FastAPI dependency: validates the key, then applies the per-key limit."""
    if AUTH_DISABLED:
        return "anonymous"

    # compare_digest against every configured key: a plain `in` check on a set
    # short-circuits on the first differing byte and leaks key content by timing.
    if not any(secrets.compare_digest(x_api_key, valid) for valid in API_KEYS):
        client = request.client.host if request.client else None
        log.warning("auth_rejected", path=request.url.path, client=client)
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    key_id = _key_id(x_api_key)
    if _limiter is not None:
        allowed, reset_in = _limiter.check(key_id)
        if not allowed:
            log.warning("rate_limited", key_id=key_id, limit_per_min=_limiter.limit)
            raise HTTPException(
                status_code=429,
                detail=f"rate limit of {_limiter.limit}/min exceeded",
                headers={"Retry-After": str(reset_in)},
            )
    return key_id
