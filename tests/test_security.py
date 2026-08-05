"""Auth and rate limiting.

The property worth protecting is fail-closed: an unauthenticated scoring
endpoint that silently works is how a demo ends up public with no auth.
"""
from __future__ import annotations

import importlib

import pytest
from conftest import FakeRedis
from fastapi import HTTPException


def _reload_security(monkeypatch, **env):
    for key in ("API_KEYS", "ALLOW_UNAUTHENTICATED"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import api.security

    return importlib.reload(api.security)


class _Req:
    class client:
        host = "1.2.3.4"

    class url:
        path = "/score"


def test_missing_api_keys_refuses_to_start(monkeypatch):
    with pytest.raises(RuntimeError, match="API_KEYS is unset"):
        _reload_security(monkeypatch)


def test_unauthenticated_mode_requires_an_explicit_opt_in(monkeypatch):
    sec = _reload_security(monkeypatch, ALLOW_UNAUTHENTICATED="true")
    assert sec.AUTH_DISABLED is True
    assert sec.require_api_key(_Req(), "") == "anonymous"


def test_valid_key_is_accepted(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a,key-b")
    sec.init_rate_limiter()
    monkeypatch.setattr(sec._limiter, "client", FakeRedis())
    assert sec.require_api_key(_Req(), "key-b")


def test_invalid_key_is_rejected(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    with pytest.raises(HTTPException) as exc:
        sec.require_api_key(_Req(), "wrong")
    assert exc.value.status_code == 401


def test_missing_header_is_rejected(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    with pytest.raises(HTTPException) as exc:
        sec.require_api_key(_Req(), "")
    assert exc.value.status_code == 401


def test_key_id_is_not_the_key(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="super-secret")
    key_id = sec._key_id("super-secret")
    # Logs and Redis key names are not secret stores.
    assert "super-secret" not in key_id
    assert key_id == sec._key_id("super-secret")  # stable


def test_rate_limit_blocks_past_the_ceiling(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    limiter = sec.RateLimiter.__new__(sec.RateLimiter)
    limiter.client = FakeRedis()
    limiter.limit = 3

    assert [limiter.check("id")[0] for _ in range(3)] == [True, True, True]
    allowed, reset_in = limiter.check("id")
    assert allowed is False
    assert 0 <= reset_in <= 60


def test_rate_limit_buckets_are_per_key(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    limiter = sec.RateLimiter.__new__(sec.RateLimiter)
    limiter.client = FakeRedis()
    limiter.limit = 1

    assert limiter.check("caller-one")[0] is True
    assert limiter.check("caller-two")[0] is True   # separate budget
    assert limiter.check("caller-one")[0] is False


def test_rate_limit_of_zero_disables_it(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    limiter = sec.RateLimiter.__new__(sec.RateLimiter)
    limiter.client = FakeRedis()
    limiter.limit = 0
    assert all(limiter.check("id")[0] for _ in range(100))


def test_rate_limiter_fails_open_when_redis_is_down(monkeypatch):
    """Rate limiting is abuse protection, not authorisation. Dropping every
    request because the limiter is unreachable turns a degraded dependency
    into a full outage."""
    import redis

    sec = _reload_security(monkeypatch, API_KEYS="key-a")

    class Broken:
        def pipeline(self):
            raise redis.RedisError("down")

    limiter = sec.RateLimiter.__new__(sec.RateLimiter)
    limiter.client = Broken()
    limiter.limit = 1
    assert limiter.check("id") == (True, 0)


def test_rate_limited_caller_gets_429_with_retry_after(monkeypatch):
    sec = _reload_security(monkeypatch, API_KEYS="key-a")
    sec.init_rate_limiter()
    sec._limiter.client = FakeRedis()
    sec._limiter.limit = 1

    assert sec.require_api_key(_Req(), "key-a")
    with pytest.raises(HTTPException) as exc:
        sec.require_api_key(_Req(), "key-a")
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers
