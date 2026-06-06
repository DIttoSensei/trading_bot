import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Optional

class TradingStrategy:
    """Unified trading strategy focused on trend following"""
    
    def __init__(self):
        self.entry_prices = {}  # Track entry prices per symbol
    
    def analyze(self, df: pd.DataFrame, has_position: bool, symbol: str) -> Optional[Dict]:
        """Main analysis function - returns actionable signal"""
        if len(df) < 100:
            return None
        
        # Calculate all indicators
        indicators = self._calculate_indicators(df)
        if not indicators:
            return None
        
        # Detect market regime
        regime = self._detect_regime(df, indicators)
        
        # Calculate signal based on regime
        if regime == 'TRENDING_UP':
            signal = self._trend_following_signal(df, indicators, has_position, symbol)
        elif regime == 'TRENDING_DOWN':
            signal = self._avoid_down_trend(has_position)
        else:
            signal = self._range_market_signal(indicators, has_position, symbol)
        
        if signal:
            signal['regime'] = regime
            return signal
        
        return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Calculate all technical indicators"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Trend indicators
            ema_20 = ta.ema(close, length=20)
            ema_50 = ta.ema(close, length=50)
            ema_200 = ta.ema(close, length=200)
            
            # Momentum
            rsi = ta.rsi(close, length=14)
            macd = ta.macd(close)
            
            # Volatility
            bb = ta.bbands(close, length=20, std=2)
            atr = ta.atr(high, low, close, length=14)
            
            # Volume
            volume_sma = ta.sma(volume, length=20)
            
            # ADX for trend strength
            adx = ta.adx(high, low, close, length=14)
            
            return {
                'close': close[-1],
                'ema_20': ema_20[-1] if not pd.isna(ema_20[-1]) else close[-1],
                'ema_50': ema_50[-1] if not pd.isna(ema_50[-1]) else close[-1],
                'ema_200': ema_200[-1] if not pd.isna(ema_200[-1]) else close[-1],
                'rsi': rsi[-1] if not pd.isna(rsi[-1]) else 50,
                'macd': macd['MACD_12_26_9'][-1] if macd is not None and not pd.isna(macd['MACD_12_26_9'][-1]) else 0,
                'macd_signal': macd['MACDs_12_26_9'][-1] if macd is not None and not pd.isna(macd['MACDs_12_26_9'][-1]) else 0,
                'bb_upper': bb['BBU_20_2.0'][-1] if bb is not None and not pd.isna(bb['BBU_20_2.0'][-1]) else close[-1] * 1.02,
                'bb_lower': bb['BBL_20_2.0'][-1] if bb is not None and not pd.isna(bb['BBL_20_2.0'][-1]) else close[-1] * 0.98,
                'bb_middle': bb['BBM_20_2.0'][-1] if bb is not None and not pd.isna(bb['BBM_20_2.0'][-1]) else close[-1],
                'atr': atr[-1] if not pd.isna(atr[-1]) else close[-1] * 0.02,
                'atr_pct': (atr[-1] / close[-1]) if not pd.isna(atr[-1]) else 0.02,
                'volume': volume[-1],
                'volume_sma': volume_sma[-1] if not pd.isna(volume_sma[-1]) else volume[-1],
                'adx': adx['ADX_14'][-1] if adx is not None and not pd.isna(adx['ADX_14'][-1]) else 20,
            }
        except Exception as e:
            print(f"Indicator error: {e}")
            return None
    
    def _detect_regime(self, df: pd.DataFrame, indicators: Dict) -> str:
        """Detect market regime with trend strength"""
        # Calculate trend strength
        price_above_50 = indicators['close'] > indicators['ema_50']
        price_above_200 = indicators['close'] > indicators['ema_200']
        ema_50_above_200 = indicators['ema_50'] > indicators['ema_200']
        
        # ADX > 25 indicates trending market
        is_trending = indicators['adx'] > 25
        
        if is_trending and price_above_50 and price_above_200 and ema_50_above_200:
            return 'TRENDING_UP'
        elif is_trending and not price_above_50 and not price_above_200:
            return 'TRENDING_DOWN'
        else:
            return 'RANGE'
    
    def _trend_following_signal(self, df: pd.DataFrame, indicators: Dict, has_position: bool, symbol: str) -> Optional[Dict]:
        """Generate signal for trending markets"""
        confidence = 0.5
        
        # Buy signals (only if no position)
        if not has_position:
            # Check for pullback to EMA in uptrend
            price_near_ema = abs(indicators['close'] - indicators['ema_20']) / indicators['close'] < 0.01
            
            # RSI not overbought (below 70)
            rsi_good = indicators['rsi'] < 70 and indicators['rsi'] > 40
            
            # MACD bullish
            macd_bullish = indicators['macd'] > indicators['macd_signal']
            
            # Volume confirmation
            volume_surge = indicators['volume'] > indicators['volume_sma'] * 1.2
            
            if price_near_ema and rsi_good and macd_bullish:
                confidence = 0.75
                if volume_surge:
                    confidence = 0.85
                
                return {
                    'action': 'BUY',
                    'confidence': confidence,
                }
        
        # Sell signals (for existing positions)
        elif has_position:
            # Get entry price
            entry_price = self.entry_prices.get(symbol, indicators['close'])
            current_return = (indicators['close'] - entry_price) / entry_price
            
            # Take profit at 3% gain
            if current_return > 0.03:
                return {
                    'action': 'SELL',
                    'confidence': 0.80,
                    'reason': f'take_profit_{current_return:.2%}'
                }
            
            # Stop loss at 2%
            if current_return < -0.02:
                return {
                    'action': 'SELL',
                    'confidence': 0.90,
                    'reason': f'stop_loss_{current_return:.2%}'
                }
        
        return None
    
    def _avoid_down_trend(self, has_position: bool) -> Optional[Dict]:
        """In downtrend - don't buy, only sell if holding"""
        if has_position:
            return {
                'action': 'SELL',
                'confidence': 0.70,
                'reason': 'downtrend_exit'
            }
        return None
    
    def _range_market_signal(self, indicators: Dict, has_position: bool, symbol: str) -> Optional[Dict]:
        """In ranging markets - buy low, sell high with tight targets"""
        if not has_position:
            # Buy at support (lower Bollinger Band)
            at_lower_bb = indicators['close'] <= indicators['bb_lower'] * 1.001
            
            # RSI oversold
            rsi_oversold = indicators['rsi'] < 35
            
            if at_lower_bb and rsi_oversold:
                return {
                    'action': 'BUY',
                    'confidence': 0.70,
                }
        
        elif has_position:
            # Get entry price
            entry_price = self.entry_prices.get(symbol, indicators['close'])
            current_return = (indicators['close'] - entry_price) / entry_price
            
            # Take profit quickly in ranges (1% target)
            if current_return > 0.01:
                return {
                    'action': 'SELL',
                    'confidence': 0.75,
                    'reason': f'range_profit_{current_return:.2%}'
                }
            
            # Tight stop in ranges (1.5%)
            if current_return < -0.015:
                return {
                    'action': 'SELL',
                    'confidence': 0.80,
                    'reason': f'range_stop_{current_return:.2%}'
                }
        
        return None
