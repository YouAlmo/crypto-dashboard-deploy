import time
import hashlib
import threading

from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────
# Ultra Fast Memory Cache
# ─────────────────────────────────────────────────────────────

class MemoryCache:

    def __init__(self):

        self.cache: Dict[str, Dict] = {}

        self.lock = threading.Lock()

    # ─────────────────────────────────────
    # Create Cache Key
    # ─────────────────────────────────────

    def _make_key(
        self,
        *args,
        **kwargs,
    ) -> str:

        raw = str(args) + str(kwargs)

        return hashlib.md5(
            raw.encode()
        ).hexdigest()

    # ─────────────────────────────────────
    # Get Cached Item
    # ─────────────────────────────────────

    def get(
        self,
        key: str,
    ) -> Optional[Any]:

        with self.lock:

            item = self.cache.get(key)

            if item is None:
                return None

            expiry = item.get("expiry")

            if expiry < time.time():

                del self.cache[key]

                return None

            return item.get("value")

    # ─────────────────────────────────────
    # Set Cache Item
    # ─────────────────────────────────────

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 30,
    ):

        with self.lock:

            self.cache[key] = {

                "value": value,

                "expiry": time.time() + ttl,

                "created": time.time(),
            }

    # ─────────────────────────────────────
    # Delete Item
    # ─────────────────────────────────────

    def delete(
        self,
        key: str,
    ):

        with self.lock:

            if key in self.cache:
                del self.cache[key]

    # ─────────────────────────────────────
    # Clear Cache
    # ─────────────────────────────────────

    def clear(self):

        with self.lock:

            self.cache.clear()

    # ─────────────────────────────────────
    # Cleanup Expired
    # ─────────────────────────────────────

    def cleanup(self):

        now = time.time()

        with self.lock:

            expired = [

                key

                for key, value
                in self.cache.items()

                if value["expiry"] < now
            ]

            for key in expired:
                del self.cache[key]

    # ─────────────────────────────────────
    # Stats
    # ─────────────────────────────────────

    def stats(self) -> Dict:

        with self.lock:

            return {

                "items": len(self.cache),

                "keys": list(
                    self.cache.keys()
                )[:10],
            }


# ─────────────────────────────────────────────────────────────
# Global Cache Instance
# ─────────────────────────────────────────────────────────────

_cache = MemoryCache()


# ─────────────────────────────────────────────────────────────
# Public Helpers
# ─────────────────────────────────────────────────────────────

def cache_get(
    key: str,
):

    return _cache.get(key)


def cache_set(
    key: str,
    value,
    ttl: int = 30,
):

    _cache.set(
        key,
        value,
        ttl,
    )


def cache_delete(
    key: str,
):

    _cache.delete(key)


def cache_clear():

    _cache.clear()


def cache_cleanup():

    _cache.cleanup()


def cache_stats():

    return _cache.stats()


# ─────────────────────────────────────────────────────────────
# Decorator Cache
# ─────────────────────────────────────────────────────────────

def smart_cache(
    ttl: int = 30,
):

    def decorator(func):

        def wrapper(
            *args,
            **kwargs,
        ):

            raw_key = (
                func.__name__
                + str(args)
                + str(kwargs)
            )

            key = hashlib.md5(
                raw_key.encode()
            ).hexdigest()

            cached = cache_get(key)

            if cached is not None:
                return cached

            result = func(
                *args,
                **kwargs,
            )

            cache_set(
                key,
                result,
                ttl,
            )

            return result

        return wrapper

    return decorator