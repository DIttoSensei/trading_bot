# ===== LAYER 3: DOMINANCE-BASED TRADING ENGINE + SHADOW FORESIGHT =====
import numpy as np

class LLMJudge:
    def __init__(self):
        self.base_threshold = 0.55
        self.risk_per_trade = 0.02

    def run_shadow_simulations(self, current_price, df, hours_ahead=12, simulations=1000):
        """
        Runs 1,000 parallel futures based on historical volatility.
        This is the 'Foreshadowing' engine.
        """
        returns = df['close'].pct_change().dropna()
        if returns.empty:
            return 0.5, current_price, current_price * 0.98
            
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate 1000 random paths for the next 4 hours
        shocks = np.random.normal(mu, sigma, (simulations, hours_ahead))
        # Log-normal price walk
        price_paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        final_prices = price_paths[:, -1]
        
        # Calculate Shadow Metrics
        win_prob = np.sum(final_prices > current_price) / simulations
        expected_price = np.mean(final_prices)
        worst_case = np.percentile(final_prices, 5) # 5% probability 'Crash Floor'
        
        return float(win_prob), float(expected_price), float(worst_case)

    def detect_regime(self, df):
        returns = df['close'].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.015
        sma = df['close'].rolling(20).mean()
        if len(sma) < 20:
            trend = 0.0
        else:
            trend = float((sma.iloc[-1] - sma.iloc[-20]) / (sma.iloc[-20] if sma.iloc[-20] != 0 else 1))
        return volatility, trend

    def evaluate(self, tech, ml, df):
        current_price = float(df.iloc[-1]['close'])
        volatility, trend = self.detect_regime(df)
        
        # 1. RUN SHADOW FORESIGHT
        # This simulates 1000 futures to see if the 'Trend' is a trap
        shadow_win_prob, exp_price, shadow_risk = self.run_shadow_simulations(current_price, df)

        # 2. COMBINE SIGNALS (Tech + ML + Shadow Probability)
        tech_prob = (tech + 1) / 2
        
        # DYNAMIC CONFIDENCE: 40% Shadow Prediction, 30% ML, 30% Technicals
        confidence = (0.4 * shadow_win_prob) + (0.3 * ml) + (0.3 * tech_prob)

        # Trend and Volatility adjustments
        if (tech > 0 and trend > 0) or (tech < 0 and trend < 0):
            confidence *= 1.05
        if volatility > 0.03:
            confidence *= 0.85

        confidence = float(np.clip(confidence, 0, 1))

        # 3. DYNAMIC THRESHOLDING
        regime = "trend" if abs(trend) >= 0.01 else "range"
        dynamic_threshold = self.base_threshold
        
        # If the 'Shadows' show a massive worst-case drop (>3%), we raise the bar to enter
        risk_gap = (current_price - shadow_risk) / current_price
        if risk_gap > 0.03:
            dynamic_threshold += 0.05 

        dynamic_threshold = float(np.clip(dynamic_threshold, 0.51, 0.65))

        # 4. FINAL DECISION
        if confidence > dynamic_threshold:
            action = 'BUY'
        elif confidence < (1 - dynamic_threshold) or current_price < shadow_risk:
            # If the current price falls below the 5th percentile 'Worst Case', panic exit
            action = 'SELL'
        else:
            action = 'HOLD'

        return {
            "action": action,
            "confidence": confidence,
            "threshold": dynamic_threshold,
            "regime": regime,
            "shadow_win_prob": shadow_win_prob,
            "expected_future_price": exp_price,
            "shadow_risk_floor": shadow_risk
        }
