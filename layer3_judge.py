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
        # Calculate log returns for better statistical properties over long periods
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()

        if len(returns) < 72:
            return 0.5, current_price, current_price * 0.97

        # 3-Year Fix: We use the last 1000h as "Historical" and 72h as "Recent"
        recent_returns = returns.iloc[-72:]
        hist_window = returns.iloc[-1000:] 

        mu_recent, sigma_recent = recent_returns.mean(), recent_returns.std()
        mu_hist, sigma_hist = hist_window.mean(), hist_window.std()

        # Blend: 80% Recent / 20% Historical (Aggressive for Crypto)
        mu = 0.80 * mu_recent + 0.20 * mu_hist
        sigma = 0.80 * sigma_recent + 0.20 * sigma_hist

        # Trend Bias based on 20h vs 200h (Golden/Death Cross logic)
        sma_20 = df["close"].rolling(20).mean().iloc[-1]
        sma_200 = df["close"].rolling(200).mean().iloc[-1]

        if current_price > sma_20 and sma_20 > sma_200:
            mu += abs(mu) * 0.1  # Strengthen upward drift in bull alignment
        elif current_price < sma_20 and sma_20 < sma_200:
            mu -= abs(mu) * 0.1  # Strengthen downward drift in bear alignment

        sigma = max(sigma, 1e-6)

        # Simulate paths using Geometric Brownian Motion logic
        shocks = np.random.normal(mu, sigma, (simulations, hours_ahead))
        price_paths = current_price * np.exp(np.cumsum(shocks, axis=1))
        final_prices = price_paths[:, -1]

        win_prob = float(np.sum(final_prices > current_price) / simulations)
        expected_price = float(np.mean(final_prices))
        # 95% Confidence floor (Value at Risk)
        worst_case = float(np.percentile(final_prices, 5))

        return win_prob, expected_price, worst_case

    def detect_regime(self, df):
        returns = df["close"].pct_change().dropna()
        # Annualized volatility approx
        volatility = float(returns.std() * np.sqrt(24 * 365)) 
        # Using a raw std for the threshold logic
        raw_vol = float(returns.tail(48).std())

        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        current = df["close"].iloc[-1]

        trend_strength = 0.0
        trend_direction = 0
        is_trending = False

        if len(df) >= 50:
            s20 = sma_20.iloc[-1]
            s50 = sma_50.iloc[-1]
            trend_strength = float(abs((s20 - s50) / s50))
            trend_direction = 1 if s20 > s50 else -1
            is_trending = trend_strength > 0.008 # 0.8% threshold for trending

        return raw_vol, trend_strength, trend_direction, is_trending

    def evaluate(self, tech, ml, df):
        current_price = float(df.iloc[-1]["close"])
        vol, trend_str, trend_dir, is_trending = self.detect_regime(df)

        shadow_win_prob, exp_price, shadow_risk = self.run_shadow_simulations(current_price, df)

        # Signal Combination
        tech_prob = (tech + 1) / 2.0

        # FINAL CONFIDENCE FORMULA
        # Logic: Shadow (Monte Carlo) is the anchor, ML is the momentum, Tech is the trigger
        confidence = (0.40 * shadow_win_prob) + (0.35 * ml) + (0.25 * tech_prob)

        if is_trending and trend_dir > 0 and tech > 0:
            confidence *= 1.05 

        # Volatility Squelch (If market is erratic, trust signals less)
        if vol > 0.03:
            confidence *= 0.85

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # --- DYNAMIC THRESHOLDING ---
        dynamic_threshold = self.base_threshold
        risk_gap = (current_price - shadow_risk) / current_price

        # Adjust threshold based on risk gaps
        if risk_gap > 0.05: dynamic_threshold += 0.08
        elif risk_gap > 0.03: dynamic_threshold += 0.04

        # In trending markets, we can be slightly more aggressive
        if is_trending and trend_dir > 0:
            dynamic_threshold -= 0.02
            
        # Determine the accurate structural label
        regime_label = "bull_trend" if (is_trending and trend_dir > 0) else \
                       "bear_trend" if (is_trending and trend_dir < 0) else "range"
                       
        # Lower the barrier inside ranges directly inside the judge to affect the action
        if regime_label == "range":
            dynamic_threshold = 0.35

        dynamic_threshold = float(np.clip(dynamic_threshold, 0.30, self.max_threshold))

        # --- DECISION MATRIX ---
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


def technical_bot(df):
    """
    Returns a composite signal from -1.0 to +1.0.
    """
    if df is None or len(df) < 50:
        return 0.0

    data = df.copy()

    # 1. RSI (14)
    data["rsi"] = ta.rsi(data["close"], length=14)
    rsi = data["rsi"].iloc[-1]
    rsi_signal = (rsi - 50) / 50 if pd.notna(rsi) else 0

    # 2. MACD (12,26,9)
    macd = ta.macd(data["close"], fast=12, slow=26, signal=9)
    macd_signal = 0.0
    if macd is not None and not macd.empty:
        m_line = macd.iloc[:, 0]
        s_line = macd.iloc[:, 2]
        if pd.notna(m_line.iloc[-1]) and pd.notna(s_line.iloc[-1]):
            macd_signal = np.tanh(
                (m_line.iloc[-1] - s_line.iloc[-1]) / (data["close"].iloc[-1] * 0.001)
            )

    # 3. Bollinger Bands %B
    bb = ta.bbands(data["close"], length=20, std=2)
    bb_signal = 0.0
    if bb is not None and not bb.empty:
        upper = bb.iloc[:, 2]
        lower = bb.iloc[:, 0]
        curr = data["close"].iloc[-1]
        band_width = upper.iloc[-1] - lower.iloc[-1]
        if pd.notna(upper.iloc[-1]) and band_width > 0:
            pct_b = (curr - lower.iloc[-1]) / band_width
            bb_signal = (0.5 - pct_b) * 2

    # 4. Stochastic RSI
    stoch = ta.stochrsi(data["close"], length=14, rsi_length=14, k=3, d=3)
    stoch_signal = 0.0
    if stoch is not None and not stoch.empty:
        k, d = stoch.iloc[-1, 0], stoch.iloc[-1, 1]
        if pd.notna(k) and pd.notna(d):
            stoch_signal = (k - d) / 100

    # 5. ADX - trend strength filter
    adx_result = ta.adx(data["high"], data["low"], data["close"], length=14)
    adx_value = 20.0  
    adx_directional = 0.0
    if adx_result is not None and not adx_result.empty:
        adx_col = [c for c in adx_result.columns if c.startswith("ADX_")]
        dmp_col = [c for c in adx_result.columns if c.startswith("DMP_")]
        dmn_col = [c for c in adx_result.columns if c.startswith("DMN_")]
        if adx_col and pd.notna(adx_result[adx_col[0]].iloc[-1]):
            adx_value = float(adx_result[adx_col[0]].iloc[-1])
        if dmp_col and dmn_col:
            dmp = adx_result[dmp_col[0]].iloc[-1]
            dmn = adx_result[dmn_col[0]].iloc[-1]
            if pd.notna(dmp) and pd.notna(dmn) and (dmp + dmn) > 0:
                adx_directional = (dmp - dmn) / (dmp + dmn)

    # 6. EMA crossover (fast 9 / slow 21)
    ema_fast = ta.ema(data["close"], length=9)
    ema_slow = ta.ema(data["close"], length=21)
    ema_signal = 0.0
    if ema_fast is not None and ema_slow is not None:
        ef = ema_fast.iloc[-1]
        es = ema_slow.iloc[-1]
        if pd.notna(ef) and pd.notna(es) and es > 0:
            ema_signal = np.tanh((ef - es) / es * 100)

    # 7. Volume confirmation
    data["vol_ma"] = data["volume"].rolling(20).mean()
    vol_signal = 0.0
    if data["vol_ma"].iloc[-1] > 0:
        vol_ratio = data["volume"].iloc[-1] / data["vol_ma"].iloc[-1]
        price_change = np.sign(data["close"].iloc[-1] - data["close"].iloc[-2])
        vol_signal = np.tanh(vol_ratio - 1) * price_change

    # --- DYNAMIC WEIGHTING based on market regime ---
    is_trending = adx_value > 25
    trend_intensity = abs(macd_signal)

    if is_trending and trend_intensity > 0.3:
        weights = {
            "rsi": 0.10, "macd": 0.30, "bb": 0.05,
            "stoch": 0.08, "adx_dir": 0.25, "ema": 0.15, "vol": 0.07
        }
    elif is_trending:
        weights = {
            "rsi": 0.15, "macd": 0.25, "bb": 0.10,
            "stoch": 0.10, "adx_dir": 0.20, "ema": 0.12, "vol": 0.08
        }
    else:
        weights = {
            "rsi": 0.25, "macd": 0.15, "bb": 0.22,
            "stoch": 0.15, "adx_dir": 0.05, "ema": 0.10, "vol": 0.08
        }

    composite = (
        weights["rsi"] * rsi_signal
        + weights["macd"] * macd_signal
        + weights["bb"] * bb_signal
        + weights["stoch"] * stoch_signal
        + weights["adx_dir"] * adx_directional
        + weights["ema"] * ema_signal
        + weights["vol"] * vol_signal
    )

    return float(np.clip(composite, -1.0, 1.0))
