# ===== LAYER 1: ADVANCED TECHNICAL BOT (Optimized for Shadow Engine) =====
import pandas as pd
import pandas_ta as ta
import numpy as np

def technical_bot(df):
    """
    Returns a composite signal from -1.0 to +1.0.
    Optimized for stability during rapid parallel simulations.
    """
    if df is None or len(df) < 50:
        return 0.0

    data = df.copy()
    
    # 1. RSI (14) - Overbought/Oversold
    data['rsi'] = ta.rsi(data['close'], length=14)
    rsi = data['rsi'].iloc[-1]
    rsi_signal = (rsi - 50) / 50 if pd.notna(rsi) else 0

    # 2. MACD (12,26,9) - Momentum
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    macd_signal = 0
    if macd is not None and not macd.empty:
        # Using specific pandas_ta naming conventions for robustness
        m_line = macd.iloc[:, 0] # MACD Line
        s_line = macd.iloc[:, 2] # Signal Line
        if pd.notna(m_line.iloc[-1]) and pd.notna(s_line.iloc[-1]):
            # Use tanh to squash the difference into a readable signal
            macd_signal = np.tanh((m_line.iloc[-1] - s_line.iloc[-1]) / (data['close'].iloc[-1] * 0.001))

    # 3. Bollinger Bands %B - Position relative to volatility bands
    bb = ta.bbands(data['close'], length=20, std=2)
    bb_signal = 0
    if bb is not None and not bb.empty:
        # Calculate %B manually if column is missing: (Price - Lower) / (Upper - Lower)
        upper = bb.iloc[:, 2]
        lower = bb.iloc[:, 0]
        curr = data['close'].iloc[-1]
        if pd.notna(upper.iloc[-1]) and (upper.iloc[-1] - lower.iloc[-1]) > 0:
            pct_b = (curr - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            bb_signal = (0.5 - pct_b) * 2 # Contrarian: Sell high band, buy low band
            
    # 4. Stochastic RSI - Fast sensitivity
    stoch = ta.stochrsi(data['close'], length=14, rsi_length=14, k=3, d=3)
    stoch_signal = 0
    if stoch is not None and not stoch.empty:
        k, d = stoch.iloc[-1, 0], stoch.iloc[-1, 1]
        if pd.notna(k) and pd.notna(d):
            stoch_signal = (k - d) / 100

    # 5. ATR Trend Strength - Volatility Filter
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)
    atr_signal = 0
    if atr is not None and not atr.empty:
        atr_ratio = atr.iloc[-1] / atr.rolling(50).mean().iloc[-1]
        # Diminish signal strength if volatility is exploding (uncertainty)
        atr_signal = 1.0 - min(atr_ratio, 2.0) / 2.0

    # 6. Volume Force - Confirmation
    data['vol_ma'] = data['volume'].rolling(20).mean()
    vol_signal = 0
    if data['vol_ma'].iloc[-1] > 0:
        vol_ratio = data['volume'].iloc[-1] / data['vol_ma'].iloc[-1]
        price_change = np.sign(data['close'].iloc[-1] - data['close'].iloc[-2])
        vol_signal = np.tanh(vol_ratio - 1) * price_change

    # --- DYNAMIC WEIGHTING ---
    # If MACD shows a strong trend, prioritize trend-following indicators
    trend_intensity = abs(macd_signal)
    if trend_intensity > 0.5:
        weights = {'rsi': 0.15, 'macd': 0.40, 'bb': 0.10, 'stoch': 0.10, 'atr': 0.10, 'vol': 0.15}
    else:
        weights = {'rsi': 0.25, 'macd': 0.20, 'bb': 0.20, 'stoch': 0.15, 'atr': 0.10, 'vol': 0.10}

    composite = (weights['rsi'] * rsi_signal +
                 weights['macd'] * macd_signal +
                 weights['bb'] * bb_signal +
                 weights['stoch'] * stoch_signal +
                 weights['atr'] * atr_signal +
                 weights['vol'] * vol_signal)

    return np.clip(composite, -1.0, 1.0)
