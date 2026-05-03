import numpy as np
import config


class LLMJudge:
    """
    Layer 3 decision engine with improvements:

    1. Regime-aware Monte Carlo simulation
       - Weights recent volatility (last 48h) 3x more than historical
       - Detects if market is in uptrend/downtrend and biases paths accordingly
       - Pure random walk was giving no real edge

    2. Smarter thresholding
       - Base threshold raised to 0.54 (was 0.51 - too easy to pass)
       - Threshold rises in high-volatility regimes (more caution needed)
       - Threshold drops slightly in confirmed uptrends (trust the trend)

    3. Cleaner signal combination
       - Explicit regime detection drives weight allocation
       - Added sell signal strength check (avoids panic exits on weak signals)
    """

    def __init__(self):
        self.base_threshold = config.BASE_THRESHOLD
        self.max_threshold = config.MAX_THRESHOLD

    def run_shadow_simulations(self, current_price, df, hours_ahead=8, simulations=1000):
        """
        Regime-aware Monte Carlo price simulation.
        Uses recent volatility weighted 3x vs full history.
        """
        returns = df["close"].pct_change().dropna()
        if len(returns) < 48:
            return 0.5, current_price, current_price * 0.97

        # Recent regime (last 48h) vs long-term
        recent_returns = returns.iloc[-48:]
        long_returns = returns

        mu_recent = recent_returns.mean()
        sigma_recent = recent_returns.std()
        mu_long = long_returns.mean()
        sigma_long = long_returns.std()

        # Blend: 70% recent, 30% historical - weights recent conditions heavier
        mu = 0.70 * mu_recent + 0.30 * mu_long
        sigma = 0.70 * sigma_recent + 0.30 * sigma_long

        # Trend adjustment: if price is above 20h MA, drift slightly positive
        ma_20 = df["close"].rolling(20).mean().iloc[-1]
        if current_price > ma_20 * 1.005:
            mu += sigma * 0.05   # slight upward bias in confirmed uptrend
        elif current_price < ma_20 * 0.995:
            mu -= sigma * 0.05   # slight downward bias in downtrend

        sigma = max(sigma, 1e-6)

        # Simulate paths
        shocks = np.random.normal(mu, sigma, (simulations, hours_ahead))
        price_paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        final_prices = price_paths[:, -1]

        win_prob = float(np.sum(final_prices > current_price) / simulations)
        expected_price = float(np.mean(final_prices))
        worst_case = float(np.percentile(final_prices, 5))

        return win_prob, expected_price, worst_case

    def detect_regime(self, df):
        """
        Returns (volatility, trend_strength, trend_direction, is_trending).
        trend_direction: +1 uptrend, -1 downtrend, 0 neutral
        """
        returns = df["close"].pct_change().dropna()
        volatility = float(returns.std()) if not returns.empty else 0.015

        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        current = df["close"].iloc[-1]

        trend_strength = 0.0
        trend_direction = 0
        is_trending = False

        if len(sma_20) >= 20 and len(sma_50) >= 50:
            s20 = sma_20.iloc[-1]
            s50 = sma_50.iloc[-1]
            if s50 > 0:
                trend_strength = float(abs((s20 - s50) / s50))
                trend_direction = 1 if s20 > s50 else -1
                is_trending = trend_strength > 0.005  # 0.5% separation = trending

        return volatility, trend_strength, trend_direction, is_trending

    def evaluate(self, tech, ml, df):
        current_price = float(df.iloc[-1]["close"])
        volatility, trend_strength, trend_direction, is_trending = self.detect_regime(df)

        # 1. Shadow simulation
        shadow_win_prob, exp_price, shadow_risk = self.run_shadow_simulations(
            current_price, df
        )

        # 2. Combine signals
        tech_prob = (tech + 1) / 2.0  # convert -1..1 to 0..1

        # Weighting: ML gets more trust when it's been trained on enough data
        # Shadow foresight: 35%, ML: 35%, Technicals: 30%
        confidence = (0.35 * shadow_win_prob) + (0.35 * ml) + (0.30 * tech_prob)

        # Trend confirmation bonus
        if is_trending:
            if (tech > 0.1 and trend_direction > 0) or (tech < -0.1 and trend_direction < 0):
                confidence *= 1.06  # signals agree with trend

        # Volatility penalty: high vol = uncertainty = reduce confidence
        if volatility > 0.04:
            confidence *= 0.80
        elif volatility > 0.025:
            confidence *= 0.90

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # 3. Dynamic threshold
        dynamic_threshold = self.base_threshold

        # Raise threshold if downside risk is large (shadow shows >3% crash floor gap)
        risk_gap = (current_price - shadow_risk) / current_price
        if risk_gap > 0.04:
            dynamic_threshold += 0.06
        elif risk_gap > 0.03:
            dynamic_threshold += 0.03

        # Raise threshold in high volatility (more noise, need stronger signal)
        if volatility > 0.035:
            dynamic_threshold += 0.04

        # Lower threshold slightly in confirmed uptrend with good ML signal
        if is_trending and trend_direction > 0 and ml > 0.58:
            dynamic_threshold -= 0.02

        dynamic_threshold = float(np.clip(dynamic_threshold, self.base_threshold, self.max_threshold))

        # 4. Decision
        sell_threshold = 1.0 - dynamic_threshold

        if confidence >= dynamic_threshold:
            action = "BUY"
        elif confidence <= sell_threshold and current_price < shadow_risk:
            # Only sell on signal AND shadow confirms danger (avoids weak-signal exits)
            action = "SELL"
        elif confidence <= (sell_threshold - 0.05):
            # Very strong sell signal even without shadow confirmation
            action = "SELL"
        else:
            action = "HOLD"

        regime_label = "trending_up" if (is_trending and trend_direction > 0) else \
                       "trending_down" if (is_trending and trend_direction < 0) else "ranging"

        return {
            "action": action,
            "confidence": confidence,
            "threshold": dynamic_threshold,
            "regime": regime_label,
            "shadow_win_prob": shadow_win_prob,
            "expected_future_price": exp_price,
            "shadow_risk_floor": shadow_risk,
            "volatility": volatility,
            "trend_direction": trend_direction,
        }