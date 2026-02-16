from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RateLimiter:
    min_interval_s: float
    _last: float = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last = time.time()


class SimpleJsonCache:
    """
    Very small, file-backed key-value cache.
    Good enough for MVP; swap to SQLite/redis later.
    """

    def __init__(self, path: str, enabled: bool = True, max_items: int = 20000):
        self.path = path
        self.enabled = enabled
        self.max_items = max_items
        self._data: Dict[str, Any] = {}
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self) -> None:
        if not self.enabled:
            return
        # crude cap
        if len(self._data) > self.max_items:
            # drop oldest-ish by key sort (MVP hack)
            for k in sorted(self._data.keys())[: len(self._data) - self.max_items]:
                self._data.pop(k, None)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f)

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._data[key] = value


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def now_iso() -> str:
    # keep it dependency-free
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())