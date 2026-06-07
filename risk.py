"""
risk.py
Tracks drawdown and enforces hard limits.
Position sizing based on volatility and config bounds.
"""
import config


class RiskManager:
    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self.peak_equity = 0.0
        self.current_equity = 0.0

    def update(self, equity: float) -> float:
        """Call every cycle. Returns current drawdown fraction."""
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity

        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def allow_trading(self, equity: float) -> bool:
        """False if drawdown exceeds limit."""
        dd = self.update(equity)
        if dd >= self.max_drawdown:
            print(f"[Risk] BLOCKED — drawdown {dd:.2%} >= limit {self.max_drawdown:.2%}")
            return False
        return True

    def position_size(self, equity: float, price: float, volatility: float) -> float:
        """
        Returns qty to buy.
        Uses volatility-adjusted notional, clamped by config min/max.
        Falls back to MIN_NOTIONAL if volatility is zero/missing.
        """
        vol = max(volatility, 0.001)
        notional = equity * config.POSITION_FRACTION / (vol * 100)
        notional = max(config.MIN_NOTIONAL_PER_TRADE, min(config.MAX_NOTIONAL_PER_TRADE, notional))

        # Also enforce equity fraction limits
        max_notional = equity * config.MAX_EQUITY_FRACTION
        min_notional = equity * config.MIN_EQUITY_FRACTION
        notional = max(min_notional, min(max_notional, notional))

        if price <= 0:
            return 0.0

        qty = round(notional / price, 8)
        return qty