import numpy as np
import pandas as pd
import pandas_ta as ta
import config

class LLMJudge:
    def __init__(self):
        self.base_threshold = config.BASE_THRESHOLD
        self.max_threshold = config.MAX_THRESHOLD

    def run_shadow_simulations(self, current_price, df, hours_ahead=8, simulations=1000):
        """
        Regime-aware Monte Carlo. Optimized for 3-year data context.
        Caps the 'historical' memory to 1000 hours so the simulation 
        stays relevant to the current market epoch.
        """
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()

        if len(returns) < 72:
            return 0.5, current_price, current_price * 0.97

        recent_returns = returns.iloc[-72:]
        hist_window = returns.iloc[-1000:] 

        mu_recent, sigma_recent = recent_returns.mean(), recent_returns.std()
        mu_hist, sigma_hist = hist_window.mean(), hist_window.std()

        mu = 0.80 * mu_recent + 0.20 * mu_hist
        sigma = 0.80 * sigma_recent + 0.20 * sigma_hist

        sma_20 = df["close"].rolling(20).mean().iloc[-1]
        sma_200 = df["close"].rolling(200).mean().iloc[-1]

        if current_price > sma_20 and sma_20 > sma_200:
            mu += abs(mu) * 0.1  
        elif current_price < sma_20 and sma_20 < sma_200:
            mu -= abs(mu) * 0.1  

        sigma = max(sigma, 1e-6)

        shocks = np.random.normal(mu, sigma, (simulations, hours_ahead))
        price_paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        final_prices = price_paths[:, -1]

        win_prob = float(np.sum(final_prices > current_price) / simulations)
        expected_price = float(np.mean(final_prices))
        worst_case = float(np.percentile(final_prices, 5))

        return win_prob, expected_price, worst_case

    def detect_regime(self, df):
        returns = df["close"].pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(24 * 365)) 
        raw_vol = float(returns.tail(48).std())

        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()

        trend_strength = 0.0
        trend_direction = 0
        is_trending = False

        if len(df) >= 50:
            s20 = sma_20.iloc[-1]
            s50 = sma_50.iloc[-1]
            trend_strength = float(abs((s20 - s50) / s50))
            trend_direction = 1 if s20 > s50 else -1
            is_trending = trend_strength > 0.008 

        return raw_vol, trend_strength, trend_direction, is_trending

    def evaluate(self, tech, ml, df):
        current_price = float(df.iloc[-1]["close"])
        vol, trend_str, trend_dir, is_trending = self.detect_regime(df)
        shadow_win_prob, exp_price, shadow_risk = self.run_shadow_simulations(current_price, df)

        regime_label = "bull_trend" if (is_trending and trend_dir > 0) else \
                       "bear_trend" if (is_trending and trend_dir < 0) else "range"

        # --- BRANCH A: RANGE MEAN REVERSION SCALPER ---
        if regime_label == "range":
            bb = ta.bbands(df["close"], length=20, std=2)
            if bb is not None and not bb.empty:
                lower_band = float(bb.iloc[-1, 0])
                
                # Condition: Trigger buy directly when price tags or dips past lower BB support
                if current_price <= lower_band * 1.001:
                    action = "BUY"
                    confidence = 0.85  
                    dynamic_threshold = 0.35
                else:
                    action = "HOLD"
                    confidence = 0.20
                    dynamic_threshold = 0.35
            else:
                action = "HOLD"
                confidence = 0.0
                dynamic_threshold = 0.35

        # --- BRANCH B: CONSERVATIVE TREND SNIPER ---
        else:
            tech_prob = (tech + 1) / 2.0
            confidence = (0.40 * shadow_win_prob) + (0.35 * ml) + (0.25 * tech_prob)

            if is_trending and trend_dir > 0 and tech > 0:
                confidence *= 1.05 

            if vol > 0.03:
                confidence *= 0.85

            confidence = float(np.clip(confidence, 0.0, 1.0))
            dynamic_threshold = self.max_threshold if regime_label == "bear_trend" else self.base_threshold

            risk_gap = (current_price - shadow_risk) / current_price
            if risk_gap > 0.05: dynamic_threshold += 0.08
            elif risk_gap > 0.03: dynamic_threshold += 0.04

            dynamic_threshold = float(np.clip(dynamic_threshold, 0.30, self.max_threshold))
            sell_threshold = 1.0 - dynamic_threshold

            if confidence >= dynamic_threshold:
                action = "BUY"
            elif confidence <= sell_threshold:
                if exp_price < current_price:
                    action = "SELL"
                else:
                    action = "HOLD"
            else:
                action = "HOLD"

        return {
            "action": action,
            "confidence": confidence,
            "threshold": dynamic_threshold,
            "regime": regime_label,
            "volatility": vol
        }
