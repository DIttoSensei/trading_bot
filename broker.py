import traceback
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class Broker:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account(self):
        return self.client.get_account()

    def get_all_positions(self):
        try:
            return self.client.get_all_positions()
        except Exception as e:
            print(f"[Broker] get_all_positions failed: {e}")
            return []

    def submit_order(self, symbol: str, side: str, qty: float):
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
            )
            order = self.client.submit_order(order_data=req)
            print(f"[Broker] {side.upper()} {qty} {symbol} → order {order.id}")
            return order
        except Exception as e:
            print(f"[Broker] submit_order FAILED {symbol}: {e}")
            traceback.print_exc()
            return None

    def close_position(self, symbol: str):
        try:
            order = self.client.close_position(symbol)
            print(f"[Broker] Closed position {symbol}")
            return order
        except Exception as e:
            print(f"[Broker] close_position skipped {symbol}: {e}")
            return None