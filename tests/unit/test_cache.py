import time

from src.utils.cache import TTLCache


def test_set_get_roundtrip(tmp_path):
    cache = TTLCache(tmp_path, ttl_seconds=60)
    key = TTLCache.make_key("search", "q", 5)
    cache.set(key, {"a": 1})
    value, stored_at = cache.get(key)
    assert value == {"a": 1}
    assert stored_at <= time.time()
    assert cache.stats()["hits"] == 1


def test_expired_entry_is_miss_but_available_stale(tmp_path):
    cache = TTLCache(tmp_path, ttl_seconds=0)
    key = TTLCache.make_key("search", "q")
    cache.set(key, [1, 2])
    cache.ttl_seconds = -1
    assert cache.get_stale(key)[0] == [1, 2]
    assert cache.get(key) is None
    assert cache.stats()["misses"] == 1


def test_disabled_cache_is_noop(tmp_path):
    cache = TTLCache(tmp_path / "x", enabled=False)
    cache.set("k", 1)
    assert cache.get("k") is None
    assert cache.get_stale("k") is None


def test_keys_are_stable_and_distinct():
    assert TTLCache.make_key("a", 1) == TTLCache.make_key("a", 1)
    assert TTLCache.make_key("a", 1) != TTLCache.make_key("a", 2)


def test_clear_and_corrupt_file(tmp_path):
    cache = TTLCache(tmp_path)
    cache.set("k1", 1)
    (tmp_path / "bad.json").write_text("{not json")
    assert cache.get("bad") is None
    assert cache.get_stale("bad") is None
    assert cache.clear() == 2
