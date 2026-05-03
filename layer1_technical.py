import pandas as pd
import pandas_ta as ta
import numpy as np


def technical_bot(df):
    """
    Returns a composite signal from -1.0 to +1.0.
    Improvements over original:
    - Added ADX trend strength filter (avoids false signals in choppy markets)
    - Added EMA crossover as primary trend direction
    - Tightened dynamic weighting logic
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

    # 5. ADX - NEW: trend strength filter
    # ADX > 25 = trending market, weight trend-following signals more
    # ADX < 20 = ranging market, weight mean-reversion signals more
    adx_result = ta.adx(data["high"], data["low"], data["close"], length=14)
    adx_value = 20.0  # neutral default
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
                # +1 = strong uptrend, -1 = strong downtrend
                adx_directional = (dmp - dmn) / (dmp + dmn)

    # 6. EMA crossover (fast 9 / slow 21) - NEW: primary trend direction
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
        # Trending: trust EMA crossover + MACD, reduce mean-reversion
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
        # Ranging: lean on RSI + BB mean reversion
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