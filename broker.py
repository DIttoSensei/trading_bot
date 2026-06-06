import math
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class Broker:
    def __init__(self, api_key, secret):
        # paper=True ensures we don't use real money until you're ready
        self.client = TradingClient(api_key, secret, paper=True)

    def get_account(self):
        return self.client.get_account()

    def get_all_positions(self):
        try:
            return self.client.get_all_positions()
        except Exception as e:
            print(f"⚠️ Failed to fetch all positions: {e}")
            return []

    def submit_order(self, symbol, side, qty, type="market", time_in_force="gtc"):
        # CRITICAL: Keep the symbol with slash for Alpaca crypto
        # Alpaca expects "BTC/USD", NOT "BTCUSD"
        
        # Clean symbol - ONLY remove if it has extra slashes, but keep the main one
        if symbol.count('/') > 1:
            symbol = symbol.replace("/", "")  # Only for malformed symbols
        # Otherwise keep the symbol as is (e.g., "BTC/USD")

        if side.lower() == "sell":
            qty = math.floor(float(qty) * 10000) / 10000.0
        else:
            qty = round(float(qty), 6)

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )

        return self.client.submit_order(order_data=order_data)

    def get_position_info(self, symbol):
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
            }
        except Exception:
            return {
                "qty": 0.0,
                "avg_entry_price": 0.0,
            }

    def close_position(self, symbol):
        try:
            return self.client.close_position(symbol)
        except Exception as e:
            print(f"⚠️ Failed to close {symbol}: {e}")
            return None
