import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

class TradingStrategy:
    """Generates entry signals - exits handled by position tracker"""
    
    def __init__(self):
        pass
    
    def analyze(self, df: pd.DataFrame, has_position: bool, symbol: str) -> Optional[Dict]:
        """Generate entry signals (exits handled separately)"""
        if len(df) < 100:
            return None
        
        indicators = self._calculate_indicators(df)
        if not indicators:
            return None
        
        # Only generate entry signals when not holding
        if not has_position:
            return self._get_entry_signal(indicators)
        
        # For exits, just return SELL if conditions met (main.py will check)
        exit_signal = self._get_exit_signal(indicators)
        if exit_signal:
            return exit_signal
        
        return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate technical indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            ema_9 = ta.ema(close, length=9)
            ema_21 = ta.ema(close, length=21)
            ema_50 = ta.ema(close, length=50)
            rsi = ta.rsi(close, length=14)
            macd = ta.macd(close)
            bb = ta.bbands(close, length=20, std=2)
            atr = ta.atr(high, low, close, length=14)
            volume_sma = ta.sma(volume, length=20)
            adx = ta.adx(high, low, close, length=14)
            
            return {
                'close': close[-1],
                'ema_9': ema_9[-1] if not pd.isna(ema_9[-1]) else close[-1],
                'ema_21': ema_21[-1] if not pd.isna(ema_21[-1]) else close[-1],
                'ema_50': ema_50[-1] if not pd.isna(ema_50[-1]) else close[-1],
                'rsi': rsi[-1] if not pd.isna(rsi[-1]) else 50,
                'macd': macd['MACD_12_26_9'][-1] if macd is not None and not pd.isna(macd['MACD_12_26_9'][-1]) else 0,
                'macd_signal': macd['MACDs_12_26_9'][-1] if macd is not None and not pd.isna(macd['MACDs_12_26_9'][-1]) else 0,
                'bb_lower': bb['BBL_20_2.0'][-1] if bb is not None and not pd.isna(bb['BBL_20_2.0'][-1]) else close[-1] * 0.98,
                'bb_upper': bb['BBU_20_2.0'][-1] if bb is not None and not pd.isna(bb['BBU_20_2.0'][-1]) else close[-1] * 1.02,
                'atr_pct': (atr[-1] / close[-1]) if not pd.isna(atr[-1]) else 0.02,
                'volume_ratio': volume[-1] / volume_sma[-1] if volume_sma[-1] > 0 else 1,
                'adx': adx['ADX_14'][-1] if adx is not None and not pd.isna(adx['ADX_14'][-1]) else 20,
            }
        except Exception as e:
            return None
    
    def _get_entry_signal(self, indicators: Dict) -> Optional[Dict]:
        """Generate BUY signals for scalping entries"""
        confidence = 0.5
        
        # SCALP SETUP 1: Oversold bounce (for bear market)
        at_lower_bb = indicators['close'] <= indicators['bb_lower'] * 1.002
        rsi_oversold = indicators['rsi'] < 40
        
        if at_lower_bb and rsi_oversold:
            confidence = 0.70
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': 'oversold_bounce'
            }
        
        # SCALP SETUP 2: EMA pullback (for bull market)
        price_near_ema = abs(indicators['close'] - indicators['ema_21']) / indicators['close'] < 0.008
        rsi_good = 40 < indicators['rsi'] < 65
        macd_bullish = indicators['macd'] > indicators['macd_signal']
        
        if price_near_ema and rsi_good and macd_bullish:
            confidence = 0.75
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': 'ema_pullback'
            }
        
        # SCALP SETUP 3: Volume breakout
        volume_surge = indicators['volume_ratio'] > 1.5
        price_above_ema = indicators['close'] > indicators['ema_9']
        
        if volume_surge and price_above_ema:
            confidence = 0.65
            return {
                'action': 'BUY',
                'confidence': confidence,
                'reason': 'volume_breakout'
            }
        
        return None
    
    def _get_exit_signal(self, indicators: Dict) -> Optional[Dict]:
        """Generate SELL signals (main.py will also check P&L)"""
        # Overbought exit
        if indicators['rsi'] > 75:
            return {
                'action': 'SELL',
                'confidence': 0.80,
                'reason': 'overbought'
            }
        
        # Upper BB exit
        if indicators['close'] >= indicators['bb_upper'] * 0.999:
            return {
                'action': 'SELL',
                'confidence': 0.75,
                'reason': 'bb_upper'
            }
        
        return None
