import numpy as np
from datetime import datetime, UTC
from typing import Dict, Optional

class RiskManager:
    """Manages overall portfolio risk"""
    
    def __init__(self, max_drawdown: float = 0.15):
        self.max_drawdown = max_drawdown
        self.peak_equity = None
        self.current_drawdown = 0.0
    
    def update(self, current_equity: float) -> float:
        """Update drawdown based on current equity"""
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        return self.current_drawdown
    
    def allow_trading(self, current_equity: float) -> bool:
        """Check if we should continue trading"""
        drawdown = self.update(current_equity)
        return drawdown < self.max_drawdown
    
    def can_trade(self, account_equity: float, daily_pnl: float) -> bool:
        """Additional check for daily limits"""
        # Import config here to avoid circular imports
        import config
        daily_loss_pct = abs(daily_pnl) / account_equity if account_equity > 0 else 0
        return daily_loss_pct < config.MAX_DAILY_LOSS_PCT


class TrailingStopTracker:
    """Tracks trailing stop peaks for each symbol"""
    
    def __init__(self):
        self.peaks = {}  # symbol -> highest price seen
        self.activation_pct = 0.02  # 2% profit needed to activate trailing stop
        self.trailing_pct = 0.01  # 1% trailing distance
    
    def update_peak(self, symbol: str, current_price: float):
        """Update the peak price for a symbol"""
        current_peak = self.peaks.get(symbol, current_price)
        if current_price > current_peak:
            self.peaks[symbol] = current_price
    
    def should_exit(self, symbol: str, current_price: float) -> tuple[bool, float]:
        """Check if trailing stop has been triggered"""
        if symbol not in self.peaks:
            return False, 0.0
        
        peak = self.peaks[symbol]
        
        # Calculate drop from peak
        drop_pct = (peak - current_price) / peak if peak > 0 else 0
        
        # Check if trailing stop is triggered
        triggered = drop_pct >= self.trailing_pct
        
        return triggered, drop_pct
    
    def on_exit(self, symbol: str):
        """Clean up when position is closed"""
        if symbol in self.peaks:
            del self.peaks[symbol]
    
    def get_peak(self, symbol: str) -> float:
        """Get current peak for a symbol"""
        return self.peaks.get(symbol, 0.0)


class PositionTracker:
    """Track open positions across GitHub runs"""
    
    def __init__(self):
        self.positions = {}  # symbol -> position data
    
    def add(self, symbol: str, entry_price: float, qty: float, signal: Dict = None):
        """Add a new position"""
        self.positions[symbol] = {
            'entry_price': entry_price,
            'entry_time': datetime.now(UTC).isoformat(),
            'qty': qty,
            'signal': signal or {}
        }
    
    def remove(self, symbol: str):
        """Remove a position"""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def get(self, symbol: str) -> Optional[Dict]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def get_entry_price(self, symbol: str) -> float:
        """Get entry price for a symbol"""
        pos = self.positions.get(symbol)
        return pos['entry_price'] if pos else 0.0
    
    def get_hold_hours(self, symbol: str) -> float:
        """Get hours held for a symbol"""
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        
        entry_time = datetime.fromisoformat(pos['entry_time'])
        hours = (datetime.now(UTC) - entry_time).total_seconds() / 3600
        return hours
    
    def get_all(self) -> Dict:
        """Get all positions"""
        return self.positions
    
    def load(self, data: Dict):
        """Load positions from state"""
        for symbol, pos_data in data.items():
            self.positions[symbol] = pos_data
    
    def save(self) -> Dict:
        """Save positions to state"""
        return self.positions
    
    def count(self) -> int:
        """Number of active positions"""
        return len(self.positions)
