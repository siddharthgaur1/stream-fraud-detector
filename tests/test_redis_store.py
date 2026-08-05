"""RedisFeatureStore: ordering and expiry.

Ordering is the subtle one. LPUSH writes newest-first, but common.features
takes `amounts[-n:]` expecting oldest-first — get that backwards and the
online features quietly disagree with the offline ones, which is train/serve
skew that no test of either side alone would catch.
"""
from __future__ import annotations

from scorer.redis_store import HISTORY_LEN, RedisFeatureStore


def _store(fake_redis, ttl=100):
    store = RedisFeatureStore.__new__(RedisFeatureStore)
    store.client = fake_redis
    store.ttl_sec = ttl
    return store


def test_history_is_returned_oldest_first(fake_redis):
    store = _store(fake_redis)
    for amount in (10.0, 20.0, 30.0):
        store.update("u1", amount, "d1")

    amounts, _ = store.get_history("u1")
    assert amounts == [10.0, 20.0, 30.0]
    # This is the property common.features actually depends on.
    assert amounts[-1:] == [30.0], "amounts[-n:] must select the most recent"


def test_history_is_capped_at_history_len(fake_redis):
    store = _store(fake_redis)
    for i in range(HISTORY_LEN + 25):
        store.update("u1", float(i), "d1")

    amounts, _ = store.get_history("u1")
    assert len(amounts) == HISTORY_LEN
    assert amounts[-1] == float(HISTORY_LEN + 24)


def test_devices_accumulate(fake_redis):
    store = _store(fake_redis)
    store.update("u1", 1.0, "d1")
    store.update("u1", 1.0, "d2")
    _, devices = store.get_history("u1")
    assert devices == {"d1", "d2"}


def test_users_do_not_share_history(fake_redis):
    store = _store(fake_redis)
    store.update("u1", 100.0, "d1")
    store.update("u2", 999.0, "d2")
    assert store.get_history("u1")[0] == [100.0]
    assert store.get_history("u2")[0] == [999.0]


def test_unknown_user_has_empty_history(fake_redis):
    amounts, devices = _store(fake_redis).get_history("never-seen")
    assert amounts == []
    assert devices == set()


def test_both_keys_get_a_ttl(fake_redis):
    # Without this the key space grows with every user_id ever seen.
    store = _store(fake_redis, ttl=1234)
    store.update("u1", 1.0, "d1")
    assert fake_redis.expiries["user:u1:amounts"] == 1234
    assert fake_redis.expiries["user:u1:devices"] == 1234


def test_ttl_is_refreshed_on_every_update(fake_redis):
    store = _store(fake_redis, ttl=1234)
    store.update("u1", 1.0, "d1")
    fake_redis.expiries.clear()
    store.update("u1", 2.0, "d1")
    # Sliding, not set-once: an active user must not expire mid-stream.
    assert fake_redis.expiries["user:u1:amounts"] == 1234


def test_ping_reports_a_down_backend(fake_redis):
    store = _store(fake_redis)
    assert store.ping() is True
    fake_redis.alive = False
    assert store.ping() is False
