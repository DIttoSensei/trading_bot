# ===== LAYER 1: ADVANCED TECHNICAL BOT =====
# Uses 6 indicators with dynamic weighting
import pandas as pd
import pandas_ta as ta
import numpy as np

def technical_bot(df):
    """
    Returns a signal from -1.0 (strong sell) to +1.0 (strong buy).
    Uses: RSI, MACD, Bollinger Bands, Stochastic RSI, ATR trend, Volume Profile.
    """
    data = df.copy()
    
    # 1. RSI (14)
    data['rsi'] = ta.rsi(data['close'], length=14)
    rsi = data['rsi'].iloc[-1]
    if pd.notna(rsi):
        rsi_signal = (rsi - 50) / 50
    else:
        rsi_signal = 0

    # 2. MACD (12,26,9)
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # Find the correct column names
        macd_col = [c for c in macd.columns if 'MACD_' in c and 'signal' not in c.lower()]
        signal_col = [c for c in macd.columns if 'signal' in c.lower()]
        hist_col = [c for c in macd.columns if 'hist' in c.lower()]
        
        if macd_col and signal_col:
            macd_line = macd[macd_col[0]].iloc[-1]
            signal_line = macd[signal_col[0]].iloc[-1]
            hist = macd[hist_col[0]].iloc[-1] if hist_col else 0
            
            if pd.notna(macd_line) and pd.notna(signal_line):
                macd_signal = np.tanh((macd_line - signal_line) * 100)
                hist_strength = np.tanh(hist * 200) if hist else 1
                macd_signal = macd_signal * (0.5 + 0.5 * abs(hist_strength))
            else:
                macd_signal = 0
        else:
            macd_signal = 0
    else:
        macd_signal = 0

    # 3. Bollinger Bands %B (20,2) - FIXED VERSION
    bb = ta.bbands(data['close'], length=20, std=2)
    if bb is not None and not bb.empty:
        # Try different possible column names
        upper_col = None
        lower_col = None
        percent_b_col = None
        
        for col in bb.columns:
            if 'upper' in col.lower() or 'BBU' in col:
                upper_col = col
            elif 'lower' in col.lower() or 'BBL' in col:
                lower_col = col
            elif 'percent' in col.lower() or 'BBP' in col:
                percent_b_col = col
        
        if percent_b_col and percent_b_col in bb.columns:
            percent_b = bb[percent_b_col].iloc[-1]
        elif upper_col and lower_col:
            upper = bb[upper_col].iloc[-1]
            lower = bb[lower_col].iloc[-1]
            current_price = data['close'].iloc[-1]
            if pd.notna(upper) and pd.notna(lower) and (upper - lower) > 0:
                percent_b = (current_price - lower) / (upper - lower)
            else:
                percent_b = 0.5
        else:
            percent_b = 0.5
        
        if pd.notna(percent_b):
            bb_signal = (0.5 - percent_b) * 2
        else:
            bb_signal = 0
    else:
        bb_signal = 0

    # 4. Stochastic RSI (14,14,3,3) – more sensitive
    try:
        stoch = ta.stochrsi(data['close'], length=14, rsi_length=14, k=3, d=3)
        if stoch is not None and not stoch.empty:
            # Find k and d columns
            k_col = [c for c in stoch.columns if 'k' in c.lower()][0] if stoch.columns else None
            d_col = [c for c in stoch.columns if 'd' in c.lower()][0] if stoch.columns else None
            
            if k_col and d_col:
                k = stoch[k_col].iloc[-1]
                d = stoch[d_col].iloc[-1]
                if pd.notna(k) and pd.notna(d):
                    stoch_signal = (k - d) / 100
                else:
                    stoch_signal = 0
            else:
                stoch_signal = 0
        else:
            stoch_signal = 0
    except:
        stoch_signal = 0

    # 5. ATR Trend Strength (volatility-adjusted)
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)
    if atr is not None and not atr.empty:
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(50).mean().iloc[-1]
        if pd.notna(current_atr) and pd.notna(avg_atr) and avg_atr > 0:
            atr_ratio = current_atr / avg_atr
            # High ATR = high volatility -> reduce position size signal
            atr_signal = 1 - min(atr_ratio, 1.5) / 1.5
        else:
            atr_signal = 0
    else:
        atr_signal = 0

    # 6. Volume Profile (volume compared to 20-period average)
    data['volume_ma'] = data['volume'].rolling(20).mean()
    price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
    vol_signal = np.tanh(vol_ratio - 1) * np.sign(price_change)

    # Dynamic weights: more weight to RSI and MACD in trending markets
    trend_strength = abs(macd_signal)  # MACD magnitude indicates trend
    weights = {
        'rsi': 0.20,
        'macd': 0.25,
        'bb': 0.15,
        'stoch': 0.15,
        'atr': 0.10,
        'vol': 0.15
    }
    # Adjust: if strong trend, favor MACD and RSI
    if trend_strength > 0.3:
        weights['macd'] = 0.35
        weights['rsi'] = 0.25
        weights['bb'] = 0.10
        weights['stoch'] = 0.10

    composite = (weights['rsi'] * rsi_signal +
                 weights['macd'] * macd_signal +
                 weights['bb'] * bb_signal +
                 weights['stoch'] * stoch_signal +
                 weights['atr'] * atr_signal +
                 weights['vol'] * vol_signal)

    return np.clip(composite, -1.0, 1.0)