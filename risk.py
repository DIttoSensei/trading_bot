import config


class RiskManager:
    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self.peak_equity = 0.0
        self.current_equity = 0.0

    def update(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity
        self.current_equity = equity
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity

    def allow_trading(self, equity: float) -> bool:
        dd = self.update(equity)
        if dd >= self.max_drawdown:
            print(f"[Risk] BLOCKED — drawdown {dd:.2%} >= limit {self.max_drawdown:.2%}")
            return False
        return True

    def position_size(self, equity: float, price: float) -> float:
        """
        Simple fixed fraction of equity.
        Uses POSITION_FRACTION (15% of equity) clamped to min/max notional.
        Volatility scaling removed — was producing $14 orders.
        """
        notional = equity * config.POSITION_FRACTION  # e.g. $96k * 0.15 = $14,400

        notional = max(config.MIN_NOTIONAL_PER_TRADE, notional)   # floor $10
        notional = min(config.MAX_NOTIONAL_PER_TRADE, notional)   # cap $5000

        if price <= 0:
            return 0.0

        qty = round(notional / price, 8)
        print(f"[Risk] Sizing: ${notional:.2f} notional → {qty} units @ ${price:.2f}")
        return qty