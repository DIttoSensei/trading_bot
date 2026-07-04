import numpy as np
import pandas as pd
import config

class LLMJudge:
    def evaluate(self, tech_signal: float, ml_prob: float, df: pd.DataFrame) -> dict:
        # Profit Hunter Pass
        if ml_prob > 0.60:
            confidence = ml_prob
        elif ml_prob < 0.40:
            confidence = ml_prob 
        else:
            # Defensive Blend: Heavier weight on tech signals when ML is uncertain
            confidence = (ml_prob * 0.40) + (tech_signal * 0.60)
            
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Dynamic Action Engine
        if confidence >= config.BASE_THRESHOLD:
            action = "BUY"
        # If technicals drop below 0.45, it indicates exhaustion. Start selling.
        elif confidence <= 0.45:
            action = "SELL"
        else:
            action = "HOLD"

        regime = "neutral"
        volatility = 0.015
        try:
            close = df["close"]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            regime = "bull" if ma20 > ma50 else "bear"

            returns = close.pct_change().dropna()
            if len(returns) >= 24:
                volatility = float(returns.tail(24).std())
        except Exception:
            pass

        return {
            "action": action,
            "confidence": confidence,
            "regime": regime,
            "volatility": volatility,
        }