from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class Broker:
    def __init__(self, api_key, secret):
        self.client = TradingClient(api_key, secret, paper=True)

    def get_account(self):
        return self.client.get_account()

    def submit_order(self, symbol, side, qty):
        qty = round(float(qty), 4)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC 
        )

        return self.client.submit_order(order_data=order)

    def get_position_qty(self, symbol):
        # Fix: Try both formatted and unformatted symbols to ensure sync
        info = self.get_position_info(symbol)
        return info["qty"]

    def get_position_info(self, symbol):
        # Fix: Alpaca API sometimes requires 'BTCUSD' instead of 'BTC/USD'
        formats = [symbol, symbol.replace("/", "")]
        for s in formats:
            try:
                pos = self.client.get_open_position(s)
                return {
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                }
            except Exception:
                continue
        return {
            "qty": 0.0,
            "avg_entry_price": 0.0,
        }
