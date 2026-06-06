import numpy as np
import pandas as pd
import pandas_ta as ta
import config

class LLMJudge:
    def __init__(self):
        self.base_threshold = config.BASE_THRESHOLD
        self.max_threshold = config.MAX_THRESHOLD

    def evaluate(self, tech_signal: float, ml_prob: float, df: pd.DataFrame) -> dict:
        """
        Executes structural market classification and synthesizes indicators
        into unified programmatic instructions.
        """
        close_series = df['close'].astype(float)
        current_close = close_series.iloc[-1]
        
        # Trend and regime classifications
        ema_50 = ta.ema(close_series, length=50)
        ema_50_val = ema_50.iloc[-1] if (ema_50 is not None and not pd.isna(ema_50.iloc[-1])) else current_close
        
        # Volatility index estimation
        returns = close_series.pct_change(1)
        volatility = float(returns.rolling(14).std().iloc[-1])
        if pd.isna(volatility):
            volatility = 0.015

        # Regime determination logic
        if current_close < ema_50_val:
            regime = "bear_trend"
            dynamic_threshold = self.max_threshold  # Raise execution barrier
        else:
            # Check for sideways range profile via rolling standard deviations
            bb = ta.bbands(close_series, length=20, std=2)
            if bb is not None:
                bandwidth = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1]
                if bandwidth.iloc[-1] < 0.04:
                    regime = "range"
                    dynamic_threshold = self.base_threshold
                else:
                    regime = "bull_trend"
                    dynamic_threshold = self.base_threshold
            else:
                regime = "bull_trend"
                dynamic_threshold = self.base_threshold

        # Weigh algorithmic models
        # Scale: 60% Machine Learning prediction, 40% Momentum signals
        confidence = (0.60 * ml_prob) + (0.40 * tech_signal)

        # Dynamic adjustments based on market environment
        if regime == "bear_trend" and tech_signal > 0.70:
            # Adjust threshold downward for strong oversold capitulation patterns
            dynamic_threshold = float(np.clip(dynamic_threshold - 0.08, 0.60, 0.80))

        # Assign trade routing directions
        sell_gate = 1.0 - dynamic_threshold
        if confidence >= dynamic_threshold:
            action = "BUY"
        elif confidence <= sell_gate:
            action = "SELL"
        else:
            action = "HOLD"

        return {
            "action": action,
            "confidence": float(confidence),
            "threshold": float(dynamic_threshold),
            "regime": regime,
            "volatility": float(volatility)
        }