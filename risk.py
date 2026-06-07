class RiskManager:
    def __init__(self, max_drawdown: float):
        self.max_drawdown = max_drawdown
        self.peak_equity = 0.0

    def update(self, equity: float) -> float:
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity == 0:
            return 0.0

        return (self.peak_equity - equity) / self.peak_equity

    def allow_trading(self, equity: float) -> bool:
        return self.update(equity) < self.max_drawdown