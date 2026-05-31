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
        """
        Fetches all currently open positions from Alpaca.
        Required by main.py to track total portfolio exposure.
        """
        try:
            return self.client.get_all_positions()
        except Exception as e:
            print(f"⚠️ Failed to fetch all positions: {e}")
            return []

    def submit_order(self, symbol, side, qty, type="market", time_in_force="gtc"):
        """
        Submits a market order to Alpaca.
        Includes a fix for floating-point 'dust' errors during sells.
        """
        # Clean symbol for Alpaca (removes the '/')
        symbol = symbol.replace("/", "")

        # --- PRECISION FIX ---
        if side.lower() == "sell":
            # We floor (round down) to 4 decimal places. 
            # This ensures we never ask to sell 0.0000001 more than we own.
            qty = math.floor(float(qty) * 10000) / 10000.0
        else:
            # When buying, rounding to 6 is usually safe.
            qty = round(float(qty), 6)
        # ---------------------

        # Convert side to Alpaca Enum
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Create the request
        order_data = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.GTC
        )

        return self.client.submit_order(order_data=order_data)

    def get_position_info(self, symbol):
        # Alpaca API usually expects symbols without the slash for positions
        clean_symbol = symbol.replace("/", "")
        try:
            pos = self.client.get_open_position(clean_symbol)
            return {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
            }
        except Exception:
            # Returns 0 if no position exists
            return {
                "qty": 0.0,
                "avg_entry_price": 0.0,
            }

    def close_position(self, symbol):
        """
        Safety method: Sells the entire position regardless of exact qty.
        Use this in main.py if you want to be 100% sure the sell clears.
        """
        clean_symbol = symbol.replace("/", "")
        return self.client.close_position(clean_symbol)
