import ccxt


class ExchangeManager:

    _exchange = None

    @classmethod
    def get_exchange(cls):

        if cls._exchange is not None:
            return cls._exchange

        exchanges_to_try = [
            "kucoin",
            "okx",
            "kraken",
            "bitfinex",
        ]

        for ex_name in exchanges_to_try:

            try:

                exchange_class = getattr(ccxt, ex_name)

                exchange = exchange_class({
                    "enableRateLimit": True,
                })

                exchange.load_markets()

                cls._exchange = exchange

                print(f"Connected to {ex_name}")

                return exchange

            except Exception as e:

                print(f"{ex_name} failed: {e}")

                continue

        raise RuntimeError(
            "No exchange available."
        )


def get_exchange():
    return ExchangeManager.get_exchange()