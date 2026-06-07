from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class Broker:
    def __init__(self, api_key: str, secret_key: str):
        self.client = TradingClient(api_key, secret_key, paper=True)

    def get_account(self):
        return self.client.get_account()

    def get_position_info(self, symbol: str):
        try:
            pos = self.client.get_open_position(symbol)
            return {
                "qty": float(pos.qty),
                "current_price": float(pos.current_price)
            }
        except:
            return {"qty": 0.0, "current_price": 0.0}

    def get_all_positions(self):
        return self.client.get_all_positions()

    def submit_order(self, symbol: str, side: str, qty: float):
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )
            return self.client.submit_order(order)
        except Exception as e:
            print("Order error:", e)
            return None