"""Small disk-backed TTL cache used to avoid repeat (paid / rate-limited) API calls.

Research results are cached for 30 minutes by default (``cache.ttl_seconds``),
which is what the project FAQ recommends for cost control. Entries are JSON files
named by a SHA-256 of the namespace and arguments, so the cache is safe to delete
at any time.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class TTLCache:
    def __init__(self, directory: str | Path, ttl_seconds: int = 1800, enabled: bool = True):
        self.directory = Path(directory)
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(namespace: str, *parts: Any) -> str:
        raw = json.dumps([namespace, *parts], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> tuple[Any, float] | None:
        """Return ``(value, stored_at_epoch)`` or ``None`` if missing/expired."""
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            entry = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self.misses += 1
            return None
        if time.time() - entry["stored_at"] > self.ttl_seconds:
            # Keep the file: expired entries are still useful via get_stale() during outages.
            self.misses += 1
            return None
        self.hits += 1
        return entry["value"], entry["stored_at"]

    def get_stale(self, key: str) -> tuple[Any, float] | None:
        """Return an entry even if expired — used as a last-resort fallback during outages."""
        if not self.enabled:
            return None
        try:
            entry = json.loads(self._path(key).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return entry["value"], entry["stored_at"]

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._path(key).write_text(json.dumps({"stored_at": time.time(), "value": value}, default=str))

    def clear(self) -> int:
        removed = 0
        if self.directory.exists():
            for path in self.directory.glob("*.json"):
                path.unlink()
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
