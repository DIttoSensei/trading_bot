import pandas as pd
import pandas_ta as ta

class ForesightEngine:
    """
    Calculates dynamic entry and exit targets based on real-time market volatility.
    """
    def __init__(self, min_profit_pct: float = 0.05):
        self.min_profit_pct = min_profit_pct

    def get_dynamic_targets(self, df: pd.DataFrame, entry_price: float) -> dict:
        if len(df) < 15:
            return self._fallback_targets(entry_price)

        # Measure real volatility using 14-period Average True Range
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        if atr is None or atr.empty or pd.isna(atr.iloc[-1]):
            return self._fallback_targets(entry_price)
            
        current_atr = atr.iloc[-1]
        
        # Project a Take Profit at 3x current volatility
        projected_take_profit = entry_price + (current_atr * 3)
        projected_pct = (projected_take_profit - entry_price) / entry_price
        
        # Enforce the absolute minimum profit floor
        if projected_pct < self.min_profit_pct:
            final_take_profit = entry_price * (1 + self.min_profit_pct)
        else:
            final_take_profit = projected_take_profit
            
        # Set a protective stop loss at 1.5x volatility to prevent early shake-outs
        final_stop_loss = entry_price - (current_atr * 1.5)
        
        return {
            "take_profit": final_take_profit,
            "stop_loss": final_stop_loss,
            "atr": current_atr
        }

    def _fallback_targets(self, entry_price: float) -> dict:
        return {
            "take_profit": entry_price * (1 + self.min_profit_pct),
            "stop_loss": entry_price * 0.95,
            "atr": 0.0
        }