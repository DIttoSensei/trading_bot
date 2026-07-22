import numpy as np
import pandas as pd
import config

class LLMJudge:
    def evaluate(self, tech_signal: float, ml_prob: float, df: pd.DataFrame) -> dict:
        # Always blend. ML gets more weight the further it is from 0.5 (i.e. the more
        # "opinion" it actually has); tech never gets more than ~45% say.
        ml_weight = 0.55 + min(abs(ml_prob - 0.5), 0.3)   # ranges 0.55–0.85
        tech_weight = 1.0 - ml_weight
        confidence = (ml_prob * ml_weight) + (tech_signal * tech_weight)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence >= config.BASE_THRESHOLD:
            action = "BUY"
        elif confidence <= 0.45:
            action = "SELL"
        else:
            action = "HOLD"

        regime = "neutral"
        volatility = 0.015
        try:
            close = df["close"]
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()
            diff = (ma20 - ma50).tail(3)          # require 3 consecutive confirming bars
            if (diff > 0).all():
                regime = "bull"
            elif (diff < 0).all():
                regime = "bear"
            returns = close.pct_change().dropna()
            if len(returns) >= 24:
                volatility = float(returns.tail(24).std())
        except Exception:
            pass

        return {"action": action, "confidence": confidence, "regime": regime, "volatility": volatility}
