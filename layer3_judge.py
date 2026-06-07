"""
layer3_judge.py
Decision engine. Combines ML probability + technical signal.
Regime detection uses MA20/MA50 crossover.
"""
import numpy as np
import pandas as pd
import config


class LLMJudge:
    def evaluate(self, tech_signal: float, ml_prob: float, df: pd.DataFrame) -> dict:
        confidence = (ml_prob * 0.65) + (tech_signal * 0.35)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence >= config.BASE_THRESHOLD:
            action = "BUY"
        elif confidence <= (1.0 - config.BASE_THRESHOLD):
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