import json
import threading
import websocket
import time

from typing import Dict, List


# ─────────────────────────────────────────────────────────────
# Binance WebSocket Manager
# ─────────────────────────────────────────────────────────────

class BinanceWebSocketManager:

    def __init__(self):

        self.ws = None

        self.thread = None

        self.running = False

        self.latest_prices: Dict[str, dict] = {}

        self.subscriptions: List[str] = []

        self.last_update = time.time()

    # ─────────────────────────────────────
    # Start WebSocket
    # ─────────────────────────────────────

    def start(
        self,
        symbols: List[str],
    ):

        if self.running:
            return

        self.subscriptions = [
            self._format_stream(s)
            for s in symbols
        ]

        stream_url = (
            "wss://stream.binance.com:9443/stream?streams="
            + "/".join(self.subscriptions)
        )

        self.running = True

        self.ws = websocket.WebSocketApp(
            stream_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self.thread = threading.Thread(
            target=self.ws.run_forever,
            daemon=True,
        )

        self.thread.start()

        print("WebSocket Started")

    # ─────────────────────────────────────
    # Format Stream
    # ─────────────────────────────────────

    def _format_stream(
        self,
        symbol: str,
    ) -> str:

        return (
            symbol.replace("/", "")
            .lower()
            + "@ticker"
        )

    # ─────────────────────────────────────
    # On Message
    # ─────────────────────────────────────

    def _on_message(
        self,
        ws,
        message,
    ):

        try:

            data = json.loads(message)

            payload = data.get("data", {})

            symbol = payload.get(
                "s",
                ""
            )

            if not symbol:
                return

            formatted_symbol = (
                symbol[:-4] + "/USDT"
                if symbol.endswith("USDT")
                else symbol
            )

            self.latest_prices[
                formatted_symbol
            ] = {

                "symbol": formatted_symbol,

                "last": float(
                    payload.get("c", 0)
                ),

                "change_percent": float(
                    payload.get("P", 0)
                ),

                "high": float(
                    payload.get("h", 0)
                ),

                "low": float(
                    payload.get("l", 0)
                ),

                "volume": float(
                    payload.get("v", 0)
                ),

                "quote_volume": float(
                    payload.get("q", 0)
                ),

                "bid": float(
                    payload.get("b", 0)
                ),

                "ask": float(
                    payload.get("a", 0)
                ),

                "open": float(
                    payload.get("o", 0)
                ),

                "timestamp": time.time(),
            }

            self.last_update = time.time()

        except Exception as e:

            print(
                f"WebSocket Message Error -> {e}"
            )

    # ─────────────────────────────────────
    # Error Handler
    # ─────────────────────────────────────

    def _on_error(
        self,
        ws,
        error,
    ):

        print(
            f"WebSocket Error -> {error}"
        )

    # ─────────────────────────────────────
    # Close Handler
    # ─────────────────────────────────────

    def _on_close(
        self,
        ws,
        close_status_code,
        close_msg,
    ):

        print("WebSocket Closed")

        self.running = False

    # ─────────────────────────────────────
    # Stop
    # ─────────────────────────────────────

    def stop(self):

        self.running = False

        try:

            if self.ws:
                self.ws.close()

        except Exception:
            pass

    # ─────────────────────────────────────
    # Get Latest Price
    # ─────────────────────────────────────

    def get_price(
        self,
        symbol: str,
    ) -> Dict:

        return self.latest_prices.get(
            symbol,
            {},
        )

    # ─────────────────────────────────────
    # Get All Prices
    # ─────────────────────────────────────

    def get_all_prices(self) -> Dict:

        return self.latest_prices

    # ─────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────

    def is_alive(self) -> bool:

        if not self.running:
            return False

        return (
            time.time()
            - self.last_update
        ) < 30


# ─────────────────────────────────────────────────────────────
# Global Singleton
# ─────────────────────────────────────────────────────────────

_ws_manager = BinanceWebSocketManager()


# ─────────────────────────────────────────────────────────────
# Public Helpers
# ─────────────────────────────────────────────────────────────

def start_websocket(
    symbols: List[str],
):

    if not _ws_manager.running:

        _ws_manager.start(symbols)


def stop_websocket():

    _ws_manager.stop()


def get_live_price(
    symbol: str,
) -> Dict:

    return _ws_manager.get_price(symbol)


def get_live_prices() -> Dict:

    return _ws_manager.get_all_prices()


def websocket_alive() -> bool:

    return _ws_manager.is_alive()