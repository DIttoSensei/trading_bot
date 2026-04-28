# ===== LAYER 3: DOMINANCE-BASED TRADING ENGINE =====

import numpy as np


class LLMJudge:
    def __init__(self):
        self.base_threshold = 0.55   # easier to trigger trades
        self.risk_per_trade = 0.02

    def detect_regime(self, df):
        returns = df['close'].pct_change().dropna()

        volatility = returns.std()
        sma = df['close'].rolling(20).mean()
        trend = (sma.iloc[-1] - sma.iloc[-20]) / sma.iloc[-20]

        return float(volatility), float(trend)

    def evaluate(self, tech, ml, df):
        volatility, trend = self.detect_regime(df)

        # Normalize tech to probability space
        tech_prob = (tech + 1) / 2

        # ===== 🔥 DOMINANCE SCORE =====
        tech_strength = abs(tech)
        ml_strength = abs(ml - 0.5)

        # Weight whichever is stronger more
        if ml_strength > tech_strength:
            confidence = 0.7 * ml + 0.3 * tech_prob
        else:
            confidence = 0.7 * tech_prob + 0.3 * ml

        # Trend boost
        if (tech > 0 and trend > 0) or (tech < 0 and trend < 0):
            confidence *= 1.05

        # Volatility penalty (only if extreme)
        if volatility > 0.03:
            confidence *= 0.85

        confidence = float(np.clip(confidence, 0, 1))

        # ===== 🔥 NEW DECISION RULE =====
        if confidence > self.base_threshold:
            action = 'BUY'
        elif confidence < (1 - self.base_threshold):
            action = 'SELL'
        else:
            action = 'HOLD'

        size = self.risk_per_trade * confidence

        return {
            "action": action,
            "confidence": confidence,
            "volatility": volatility,
            "trend": trend,
            "tech_strength": tech_strength,
            "ml_strength": ml_strength
        }