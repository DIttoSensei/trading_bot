import json
from datetime import datetime, UTC
from typing import Dict, Tuple
import config

class RiskManager:
    """Risk management focused on capital preservation"""
    
    def __init__(self):
        self.consecutive_losses = 0
    
    def can_trade(self, account_equity: float, daily_pnl: float) -> bool:
        """Check if we should continue trading"""
        # Check daily loss limit
        daily_loss_pct = abs(daily_pnl) / account_equity if account_equity > 0 else 0
        if daily_loss_pct > config.MAX_DAILY_LOSS_PCT:
            print(f"⚠️ Daily loss limit reached: {daily_loss_pct:.2%}")
            return False
        
        return True
    
    def update_trade_result(self, pnl: float):
        """Update risk metrics after trade"""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

class PositionTracker:
    """Track open positions with entry prices and timing"""
    
    def __init__(self):
        self.positions = {}
        self.strategy = None  # Will be set later
    
    def set_strategy(self, strategy):
        """Set strategy reference to update entry prices"""
        self.strategy = strategy
    
    def add(self, symbol: str, price: float, qty: float, signal: Dict):
        """Track new position"""
        self.positions[symbol] = {
            'entry_price': price,
            'entry_time': datetime.now(UTC),
            'qty': qty,
            'signal': signal
        }
        # Update strategy's entry price tracker
        if self.strategy:
            self.strategy.entry_prices[symbol] = price
    
    def remove(self, symbol: str):
        """Remove position"""
        if symbol in self.positions:
            del self.positions[symbol]
        # Clean up strategy tracker
        if self.strategy and symbol in self.strategy.entry_prices:
            del self.strategy.entry_prices[symbol]
    
    def get_entry_price(self, symbol: str) -> float:
        """Get entry price for symbol"""
        return self.positions.get(symbol, {}).get('entry_price', 0.0)
    
    def get_hold_hours(self, symbol: str) -> float:
        """Get hours held"""
        if symbol not in self.positions:
            return 0.0
        
        entry_time = self.positions[symbol]['entry_time']
        hours = (datetime.now(UTC) - entry_time).total_seconds() / 3600
        return hours
    
    def load(self, data: Dict):
        """Load from state"""
        for symbol, pos_data in data.items():
            pos_data['entry_time'] = datetime.fromisoformat(pos_data['entry_time'])
            self.positions[symbol] = pos_data
            if self.strategy:
                self.strategy.entry_prices[symbol] = pos_data['entry_price']
    
    def save(self) -> Dict:
        """Save to state"""
        save_data = {}
        for symbol, pos_data in self.positions.items():
            save_data[symbol] = {
                'entry_price': pos_data['entry_price'],
                'entry_time': pos_data['entry_time'].isoformat(),
                'qty': pos_data['qty'],
                'signal': pos_data.get('signal', {})
            }
        return save_data
