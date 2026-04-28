class RiskManager:
    def __init__(self, max_drawdown=0.1):
        self.max_drawdown = max_drawdown
        self.peak = None

    def update(self, balance):
        if self.peak is None:
            self.peak = balance
        self.peak = max(self.peak, balance)
        drawdown = (self.peak - balance) / self.peak
        return drawdown

    def allow_trading(self, balance):
        dd = self.update(balance)
        return dd < self.max_drawdown