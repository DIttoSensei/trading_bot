import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, max_drawdown: float = 0.10):
        self.max_drawdown = max_drawdown
        self.peak_equity = None
        self.current_drawdown = 0.0

    def update(self, equity: float) -> float:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        return self.current_drawdown

    def allow_trading(self, equity: float) -> bool:
        self.update(equity)
        return self.current_drawdown < self.max_drawdown

class TrailingStopTracker:
    def __init__(self, atr_multiplier=None):
        self.peaks = {}
        # 1.5% is the threshold to protect gains without being 'too twitchy'
        self.stop_threshold = 0.015 

    def update_peak(self, symbol, current_price):
        if symbol not in self.peaks or current_price > self.peaks[symbol]:
            self.peaks[symbol] = current_price

    def should_exit(self, symbol, current_price, df=None):
        peak = self.peaks.get(symbol, 0)
        if peak == 0: 
            return False, 0.0
        
        drop_pct = (peak - current_price) / peak
        
        # Priority Check: If price drops 1.5% from peak, we MUST exit.
        if drop_pct >= self.stop_threshold:
            return True, drop_pct
            
        return False, drop_pct

    def on_exit(self, symbol):
        # Clear data so the next trade starts with a fresh peak
        self.peaks.pop(symbol, None)
