import pandas as pd


class LLMJudge:
    def evaluate(self, tech_signal: float, ml_prob: float, df: pd.DataFrame = None) -> dict:

        confidence = (ml_prob * 0.65) + (tech_signal * 0.35)

        if confidence >= 0.65:
            action = "BUY"
        elif confidence <= 0.35:
            action = "SELL"
        else:
            action = "HOLD"

        # Market regime
        regime = "neutral"
        try:
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            ma50 = df["close"].rolling(50).mean().iloc[-1]
            regime = "bull_trend" if ma20 > ma50 else "bear_trend"
        except:
            pass

        # volatility
        volatility = 0.015
        try:
            volatility = df["close"].pct_change().rolling(24).std().iloc[-1]
        except:
            pass

        return {
            "action": action,
            "confidence": float(confidence),
            "threshold": 0.65,
            "regime": regime,
            "volatility": float(volatility)
        }