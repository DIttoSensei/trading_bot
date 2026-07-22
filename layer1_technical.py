import numpy as np
import pandas as pd
import pandas_ta as ta

def technical_bot(df: pd.DataFrame) -> float:
    try:
        if len(df) < 50:
            return 0.5

        close = df['close'].astype(float).values
        ema_9 = ta.ema(pd.Series(close), length=9).values
        ema_21 = ta.ema(pd.Series(close), length=21).values
        rsi = ta.rsi(pd.Series(close), length=14).values
        macd = ta.macd(pd.Series(close))
        bb = ta.bbands(pd.Series(close), length=20, std=2)

        current_close = close[-1]
        current_rsi = rsi[-1] if not pd.isna(rsi[-1]) else 50.0

        # Continuous trend score (bounded via tanh, never hard-clips)
        ema_spread = (ema_9[-1] - ema_21[-1]) / current_close
        macd_hist = 0.0
        if macd is not None and len(macd) > 0:
            macd_hist = (macd.iloc[:, 0].values[-1] - macd.iloc[:, 1].values[-1]) / current_close
        trend_score = np.tanh((ema_spread * 50) + (macd_hist * 50)) * 0.3

        # Continuous mean-reversion score using %B position in the bands
        reversion_score = 0.0
        if bb is not None and len(bb) > 0:
            bbl = bb.iloc[:, 0].values[-1]
            bbu = bb.iloc[:, 2].values[-1]
            if bbu > bbl:
                pct_b = (current_close - bbl) / (bbu - bbl)
                reversion_score = (0.5 - pct_b) * 0.4

        rsi_score = ((50 - current_rsi) / 50) * 0.15

        raw_signal = 0.5 + trend_score + reversion_score + rsi_score
        return float(np.clip(raw_signal, 0.05, 0.95))  # never lets it hit hard 0/1
    except Exception as e:
        print(f"⚠️ Exception inside Layer 1 Processing: {e}")
        return 0.5
