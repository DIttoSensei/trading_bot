import math
from typing import Dict, Optional
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

class Broker:
    def __init__(self, api_key: str, secret_key: str):
        self.client = TradingClient(api_key, secret_key, paper=True)
    
    def get_account(self):
        """Get account information"""
        return self.client.get_account()
    
    def get_all_positions(self):
        """Get all open positions"""
        try:
            return self.client.get_all_positions()
        except Exception as e:
            print(f"⚠️ Failed to get positions: {e}")
            return []
    
    def submit_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", limit_price: Optional[float] = None):
        """Submit order with proper rounding"""
        # Clean symbol
        symbol = symbol.replace("/USD", "").replace("/", "")
        
        # Round quantities properly
        if side.lower() == "sell":
            # For sells, round down to avoid selling more than we have
            qty = math.floor(qty * 10000) / 10000.0
        else:
            # For buys, standard rounding
            qty = round(qty, 6)
        
        # Ensure minimum quantity
        if qty <= 0:
            raise ValueError(f"Invalid quantity: {qty}")
        
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        
        if order_type == "market":
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY  # Day only, not GTC
            )
        elif order_type == "limit" and limit_price:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY
            )
        else:
            raise ValueError(f"Invalid order type: {order_type}")
        
        try:
            return self.client.submit_order(order)
        except Exception as e:
            print(f"❌ Order failed for {symbol}: {e}")
            raise
    
    def get_position_info(self, symbol: str) -> Dict:
        """Get position info for a symbol"""
        clean_symbol = symbol.replace("/USD", "").replace("/", "")
        try:
            pos = self.client.get_open_position(clean_symbol)
            return {
                "qty": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "unrealized_pl": float(pos.unrealized_pl),
            }
        except Exception:
            return {"qty": 0.0, "avg_entry_price": 0.0, "current_price": 0.0, "unrealized_pl": 0.0}
    
    def close_position(self, symbol: str):
        """Close entire position"""
        clean_symbol = symbol.replace("/USD", "").replace("/", "")
        try:
            return self.client.close_position(clean_symbol)
        except Exception as e:
            print(f"⚠️ Failed to close {symbol}: {e}")
            return None
