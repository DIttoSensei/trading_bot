import csv
import os
import sys
import json
import traceback
from datetime import UTC, datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from broker import Broker
from strategy import TradingStrategy
from risk import RiskManager, PositionTracker

class TradingBot:
    def __init__(self):
        """Initialize bot with conservative, profit-focused settings"""
        self._validate_config()
        
        self.symbols = config.TRADE_SYMBOLS
        self.broker = Broker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()
        self.strategy = TradingStrategy()
        self.risk = RiskManager()
        self.positions = PositionTracker()
        
        # Track daily stats
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_reset_date = datetime.now(UTC).date()
        
        # Load state
        self.state_path = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self._load_state()
        
        # Setup logging
        self.log_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._init_log()
        
        print(f"🤖 Bot initialized for {', '.join(self.symbols)}")
        print(f"📊 Max exposure: {config.MAX_TOTAL_EXPOSURE_PCT*100}%")
        print(f"🎯 Min profit target: {config.MIN_PROFIT_TARGET_PCT*100}%")
    
    def _validate_config(self):
        """Ensure config values are profitable"""
        if config.MIN_PROFIT_TARGET_PCT < 0.01:
            raise ValueError("MIN_PROFIT_TARGET_PCT must be at least 1% to cover fees")
        if config.MAX_POSITION_SIZE_PCT > 0.25:
            raise ValueError("MAX_POSITION_SIZE_PCT too high - limit to 25% for safety")
    
    def _load_state(self):
        """Load persistent state"""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    state = json.load(f)
                    self.positions.load(state.get('positions', {}))
                    self.daily_pnl = state.get('daily_pnl', 0.0)
                    self.daily_trades = state.get('daily_trades', 0)
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
    
    def _save_state(self):
        """Save persistent state"""
        state = {
            'positions': self.positions.save(),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'last_date': self.last_reset_date.isoformat()
        }
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _init_log(self):
        """Initialize trade log"""
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'action', 'price', 'qty',
                    'pnl', 'total_pnl', 'reason', 'confidence'
                ])
    
    def _log_trade(self, symbol: str, action: str, price: float, qty: float,
                   pnl: float, total_pnl: float, reason: str, confidence: float):
        """Log trade with P&L tracking"""
        row = [
            datetime.now(UTC).isoformat(),
            symbol, action, round(price, 2), round(qty, 6),
            round(pnl, 2), round(total_pnl, 2), reason, round(confidence, 3)
        ]
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        
        # Console output for monitoring
        if action == 'SELL':
            print(f"💰 {symbol}: {action} at ${price:,.2f} | P&L: ${pnl:+.2f} | Total: ${total_pnl:,.2f} | {reason}")
        else:
            print(f"📈 {symbol}: {action} at ${price:,.2f} | {reason}")
    
    def _reset_daily_stats(self):
        """Reset daily P&L at midnight UTC"""
        today = datetime.now(UTC).date()
        if today != self.last_reset_date:
            print(f"📅 Daily reset - Yesterday P&L: ${self.daily_pnl:,.2f} over {self.daily_trades} trades")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_reset_date = today
            self._save_state()
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data with sufficient length"""
        try:
            end = datetime.now(UTC)
            start = end - timedelta(hours=config.LOOKBACK_HOURS)
            
            # Remove /USD for Alpaca API
            clean_symbol = symbol.replace('/USD', '')
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[clean_symbol],
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            
            bars = self.data_client.get_crypto_bars(request).df
            if bars.empty:
                return None
            
            bars = bars.reset_index()
            bars['symbol'] = symbol
            bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            bars = bars.sort_values('timestamp').drop_duplicates('timestamp')
            
            return bars
            
        except Exception as e:
            print(f"❌ Data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_position_size(self, symbol: str, price: float, confidence: float,
                                account_equity: float, current_exposure: float) -> float:
        """Calculate position size with profit focus"""
        # Base size from config
        base_size_pct = config.MAX_POSITION_SIZE_PCT
        
        # Adjust for confidence (higher confidence = larger position)
        confidence_multiplier = min(1.5, max(0.5, confidence / config.MIN_CONFIDENCE_THRESHOLD))
        
        # Adjust for current exposure (less room = smaller position)
        remaining_capacity = config.MAX_TOTAL_EXPOSURE_PCT - current_exposure
        if remaining_capacity <= 0:
            return 0.0
        
        exposure_multiplier = min(1.0, remaining_capacity / config.MAX_POSITION_SIZE_PCT)
        
        # Final size
        target_equity_pct = base_size_pct * confidence_multiplier * exposure_multiplier
        target_equity_pct = min(target_equity_pct, remaining_capacity)
        
        target_notional = account_equity * target_equity_pct
        
        # Minimum trade size check (must be worth the fees)
        min_profit_dollars = target_notional * config.MIN_PROFIT_TARGET_PCT
        if min_profit_dollars < 5.0:
            return 0.0
        
        qty = target_notional / price
        return round(qty, 6)
    
    def run_cycle(self):
        """Main trading cycle - quality over quantity"""
        print(f"\n{'='*60}")
        print(f"🔄 Trading Cycle: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")
        
        # Reset daily stats
        self._reset_daily_stats()
        
        # Get account status
        try:
            account = self.broker.get_account()
            account_equity = float(account.equity)
            buying_power = float(account.buying_power)
        except Exception as e:
            print(f"❌ Cannot fetch account: {e}")
            return
        
        # Check if we should trade today
        if not self.risk.can_trade(account_equity, self.daily_pnl):
            print("⏸️ Risk limits reached - skipping cycle")
            return
        
        # Calculate current exposure
        current_positions = self.broker.get_all_positions()
        current_exposure = 0.0
        for p in current_positions:
            try:
                current_exposure += float(p.qty) * float(p.current_price)
            except:
                pass
        current_exposure = current_exposure / account_equity if account_equity > 0 else 0
        
        print(f"💰 Account Equity: ${account_equity:,.2f}")
        print(f"📊 Current Exposure: {current_exposure*100:.1f}%")
        print(f"📈 Daily P&L: ${self.daily_pnl:+.2f}")
        
        # Track total P&L for this cycle
        cycle_pnl = 0.0
        
        # Process each symbol
        for symbol in self.symbols:
            print(f"\n--- {symbol} ---")
            
            # Fetch data
            df = self.fetch_data(symbol)
            if df is None or len(df) < 50:
                print(f"⚠️ Insufficient data for {symbol}")
                continue
            
            # Get current position
            position = self.broker.get_position_info(symbol)
            current_qty = float(position.get('qty', 0.0))
            current_price = float(df.iloc[-1]['close'])
            position_value = current_qty * current_price
            
            # Get trading signal
            signal = self.strategy.analyze(df, current_qty > 0, symbol)
            
            if signal is None:
                print(f"⏸️ No clear signal for {symbol}")
                continue
            
            print(f"🎯 Signal: {signal['action']} | Confidence: {signal['confidence']:.2%}")
            print(f"📊 Regime: {signal['regime']}")
            
            # --- EXIT LOGIC (Close positions first) ---
            if current_qty > 0 and signal['action'] == 'SELL':
                should_exit = False
                exit_reason = ""
                
                # Force exit after max hold time
                hold_hours = self.positions.get_hold_hours(symbol)
                if hold_hours >= config.MAX_HOLD_HOURS:
                    should_exit = True
                    exit_reason = f"max_hold_{config.MAX_HOLD_HOURS}h"
                
                # Judge says sell
                elif signal['action'] == 'SELL':
                    should_exit = True
                    exit_reason = signal.get('reason', 'signal_exit')
                
                if should_exit:
                    try:
                        # Execute sell
                        self.broker.submit_order(symbol, qty=current_qty, side='sell')
                        
                        # Calculate P&L
                        avg_entry = self.positions.get_entry_price(symbol)
                        pnl = (current_price - avg_entry) * current_qty
                        self.daily_pnl += pnl
                        self.daily_trades += 1
                        cycle_pnl += pnl
                        
                        self._log_trade(symbol, 'SELL', current_price, current_qty,
                                      pnl, self.daily_pnl, exit_reason, signal['confidence'])
                        
                        # Remove from position tracker
                        self.positions.remove(symbol)
                        
                    except Exception as e:
                        print(f"❌ Sell failed: {e}")
            
            # --- ENTRY LOGIC (Only if no position) ---
            elif current_qty == 0 and signal['action'] == 'BUY':
                # Check minimum requirements
                if signal['confidence'] < config.MIN_CONFIDENCE_THRESHOLD:
                    print(f"❌ Confidence too low: {signal['confidence']:.2%} < {config.MIN_CONFIDENCE_THRESHOLD:.2%}")
                    continue
                
                # Calculate position size
                qty = self.calculate_position_size(
                    symbol, current_price, signal['confidence'],
                    account_equity, current_exposure
                )
                
                if qty <= 0:
                    print(f"❌ Position size too small (${qty * current_price:.2f})")
                    continue
                
                # Execute buy
                try:
                    self.broker.submit_order(symbol, qty=qty, side='buy')
                    
                    self._log_trade(symbol, 'BUY', current_price, qty, 0, self.daily_pnl,
                                  f"signal_{signal['regime']}", signal['confidence'])
                    
                    # Track position
                    self.positions.add(symbol, current_price, qty, signal)
                    
                    # Update exposure
                    current_exposure += (qty * current_price) / account_equity
                    
                except Exception as e:
                    print(f"❌ Buy failed: {e}")
            
            else:
                # Hold or wait
                if current_qty > 0:
                    hold_hours = self.positions.get_hold_hours(symbol)
                    avg_entry = self.positions.get_entry_price(symbol)
                    unrealized_pnl = (current_price - avg_entry) * current_qty
                    print(f"💎 Holding {symbol}: {hold_hours:.1f}h | Unrealized: ${unrealized_pnl:+.2f}")
                else:
                    print(f"⏸️ Waiting for entry signal")
        
        # End of cycle summary
        print(f"\n{'='*60}")
        print(f"✅ Cycle Complete | Today's P&L: ${self.daily_pnl:+.2f}")
        print(f"📊 Total Trades Today: {self.daily_trades}")
        print(f"{'='*60}\n")
        
        # Save state
        self._save_state()

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run_cycle()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
