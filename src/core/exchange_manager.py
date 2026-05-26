import ccxt
import threading
from typing import Optional


class ExchangeManager:
    """
    Singleton Bybit exchange manager.
    Reuses one CCXT instance across the whole app
    لتحسين الأداء وتقليل الـ latency.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.exchange = ccxt.bybit({
            "enableRateLimit": True,
            "timeout": 15000,
            "options": {
                "defaultType": "linear",
                "adjustForTimeDifference": True,
            },
        })

        # تحميل الأسواق مرة واحدة فقط
        self.exchange.load_markets()

    @classmethod
    def get_exchange(cls):

        if cls._instance is None:

            with cls._lock:

                if cls._instance is None:
                    cls._instance = cls()

        return cls._instance.exchange


def get_exchange():
    """
    Shortcut helper.
    """
    return ExchangeManager.get_exchange()
