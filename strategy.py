import time


class Strategy:
    def __init__(self):
        self.position = None
        self.last_trade_time = 0
        self.cooldown = 5  # seconds

    def decide(self, price, prob_up, risk_ok):
        now = time.time()

        if not risk_ok:
            return "HOLD"

        if now - self.last_trade_time < self.cooldown:
            return "HOLD"

        score = prob_up - 0.5

        if score > 0.08 and self.position != "LONG":
            self.last_trade_time = now
            self.position = "LONG"
            return "BUY"

        if score < -0.08 and self.position != "SHORT":
            self.last_trade_time = now
            self.position = "SHORT"
            return "SELL"

        return "HOLD"