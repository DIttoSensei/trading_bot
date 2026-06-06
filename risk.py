import numpy as np
from datetime import datetime, UTC
from typing import Dict, tuple
import config

class RiskManager:
    """
    Manages structural capital drawdown parameters.
    """
    def __init__(self, max_drawdown: float = 0.12):
        self.max_drawdown = max_drawdown
        self.peak_equity = None
        self.current_drawdown = 0.0
    
    def update(self, current_equity: float) -> float:
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        return self.current_drawdown
    
    def allow_trading(self, current_equity: float) -> bool:
        drawdown = self.update(current_equity)
        return drawdown < self.max_drawdown


class TrailingStopTracker:
    """
    Tracks local asset price maxima to calculate mathematical 
    trailing stop thresholds.
    """
    def __init__(self):
        self.peaks = {}  # Map: symbol -> peak high values
        self.activation_pct = config.TRAILING_STOP_ACTIVATION_PCT
        self.trailing_pct = config.TRAILING_STOP_DISTANCE_PCT
    
    def update_peak(self, symbol: str, current_price: float, entry_price: float):
        if entry_price <= 0:
            return
        
        profit_pct = (current_price - entry_price) / entry_price
        
        # Trailing mechanics activate once asset hits structural profit thresholds
        if profit_pct >= self.activation_pct:
            current_peak = self.peaks.get(symbol, current_price)
            if current_price > current_peak:
                self.peaks[symbol] = current_price
    
    def should_exit(self, symbol: str, current_price: float) -> tuple[bool, float]:
        if symbol not in self.peaks:
            return False, 0.0
        
        peak = self.peaks[symbol]
        drop_pct = (peak - current_price) / peak if peak > 0 else 0.0
        
        triggered = drop_pct >= self.trailing_pct
        return triggered, drop_pct
    
    def on_exit(self, symbol: str):
        if symbol in self.peaks:
            del self.peaks[symbol]