from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class Broker:
    def __init__(self, api_key, secret):
        # paper=True ensures we don't use real money until you're ready
        self.client = TradingClient(api_key, secret, paper=True)

    def get_account(self):
        return self.client.get_account()

    def submit_order(self, symbol, side, qty, type="market", time_in_force="gtc"):
        """
        Submits a market order to Alpaca.
        Args:
            symbol (str): e.g., 'BTC/USD'
            side (str): 'buy' or 'sell'
            qty (float): Number of units
            type (str): Order type (default 'market')
            time_in_force (str): e.g., 'gtc'
        """
        # Clean symbol for Alpaca (removes the '/')
        symbol = symbol.replace("/", "")
        
        # Ensure qty is a float and rounded properly for crypto
        qty = round(float(qty), 6)

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
