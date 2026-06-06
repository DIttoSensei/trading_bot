import numpy as np
import pandas as pd
import pandas_ta as ta

def technical_bot(df: pd.DataFrame) -> float:
    """
    Transforms multiple historical technical frames into a normalized 
    directional confidence vector bounded between 0.0 and 1.0.
    """
    try:
        if len(df) < 50:
            return 0.5

        close = df['close'].astype(float).values
        high = df['high'].astype(float).values
        low = df['low'].astype(float).values

        # Compute Technical Matrix Core
        ema_9 = ta.ema(pd.Series(close), length=9).values
        ema_21 = ta.ema(pd.Series(close), length=21).values
        rsi = ta.rsi(pd.Series(close), length=14).values
        macd = ta.macd(pd.Series(close))
        bb = ta.bbands(pd.Series(close), length=20, std=2)

        # Extraction layers
        current_close = close[-1]
        current_rsi = rsi[-1] if not pd.isna(rsi[-1]) else 50.0

        # Sub-component scoring vectors
        trend_score = 0.0
        if ema_9[-1] > ema_21[-1]:
            trend_score += 0.3
        if macd is not None and len(macd) > 0:
            macd_line = macd.iloc[:, 0].values[-1]
            signal_line = macd.iloc[:, 1].values[-1]
            if macd_line > signal_line:
                trend_score += 0.2

        # Mean-reversion scoring vectors
        reversion_score = 0.0
        if bb is not None and len(bb) > 0:
            bbl = bb.iloc[:, 0].values[-1]  # Lower Band
            bbu = bb.iloc[:, 2].values[-1]  # Upper Band
            
            # Position inside the structural bands
            if current_close <= bbl * 1.002:
                reversion_score += 0.4  # Oversold structural floor
            elif current_close >= bbu * 0.998:
                reversion_score -= 0.3  # Overbought structural ceiling

        # RSI relative scaling
        rsi_score = 0.0
        if current_rsi < 35:
            rsi_score += 0.1
        elif current_rsi > 70:
            rsi_score -= 0.2

        # Map scores directly to a normalized 0-1 scale
        raw_signal = 0.5 + trend_score + reversion_score + rsi_score
        return float(np.clip(raw_signal, 0.0, 1.0))

    except Exception as e:
        print(f"⚠️ Exception inside Layer 1 Processing: {e}")
        return 0.5