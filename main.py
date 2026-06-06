import csv
import os
import sys
import json
import traceback
from datetime import UTC, datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from broker import Broker
from strategy import TradingStrategy

class TradingBot:
    def __init__(self):
        self.symbols = config.TRADE_SYMBOLS
        self.broker = Broker(config.APCA_API_KEY_ID, config.APCA_API_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()
        self.strategy = TradingStrategy()
        
        # State file for remembering positions across runs
        self.state_file = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self.positions = {}
        
        # Load previous state
        self._load_state()
        
        # Setup logging
        self.log_file = os.path.join(os.path.dirname(__file__), "trades.csv")
        self._init_log()
        
        print(f"🤖 Bot started at {datetime.now(UTC)}")
        print(f"📊 Active positions from last run: {len(self.positions)}")
        for sym, pos in self.positions.items():
            print(f"   └─ {sym}: {pos['qty']} @ ${pos['entry_price']:.2f}")
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert USD to USDT for Alpaca crypto"""
        if symbol.endswith("/USD"):
            return symbol.replace("/USD", "/USDT")
        return symbol
    
    def _load_state(self):
        """Load positions from previous runs"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.positions = data.get('positions', {})
                    print(f"✅ Loaded {len(self.positions)} positions from previous run")
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
                self.positions = {}
        else:
            self.positions = {}
    
    def _save_state(self):
        """Save positions for next run"""
        state = {
            'positions': self.positions,
            'last_update': datetime.now(UTC).isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Saved {len(self.positions)} positions for next run")
    
    def _init_log(self):
        """Initialize trade log"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'action', 'price', 'qty', 'pnl', 'reason'])
    
    def _log_trade(self, symbol: str, action: str, price: float, qty: float, pnl: float, reason: str):
        """Log trade"""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(UTC).isoformat(), symbol, action, round(price, 2), qty, round(pnl, 2), reason])
        print(f"📝 {symbol}: {action} ${price:.2f} | P&L: ${pnl:.2f} | {reason}")
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data"""
        try:
            end = datetime.now(UTC)
            start = end - timedelta(hours=168)
            
            clean_symbol = self._convert_symbol(symbol)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[clean_symbol],
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )
            
            bars = self.data_client.get_crypto_bars(request).df
            if bars.empty:
                print(f"⚠️ No data for {clean_symbol}")
                return None
            
            bars = bars.reset_index()
            bars = bars[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return bars.sort_values('timestamp')
            
        except Exception as e:
            print(f"❌ Data fetch failed for {symbol}: {e}")
            return None
    
    def run_cycle(self):
        """Main trading cycle"""
        print(f"\n{'='*60}")
        print(f"🔄 Trading Cycle at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")
        
        # Get account info
        try:
            account = self.broker.get_account()
            equity = float(account.equity)
            print(f"💰 Account Equity: ${equity:,.2f}")
        except Exception as e:
            print(f"❌ Cannot get account: {e}")
            return
        
        # Check existing positions
        print(f"\n📊 Checking {len(self.positions)} existing positions...")
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            entry_price = position['entry_price']
            entry_time = datetime.fromisoformat(position['entry_time'])
            qty = position['qty']
            hold_hours = (datetime.now(UTC) - entry_time).total_seconds() / 3600
            
            # Get current price
            df = self.fetch_data(symbol)
            if df is None:
                continue
            current_price = float(df.iloc[-1]['close'])
            
            # Calculate P&L
            pnl_pct = (current_price - entry_price) / entry_price
            pnl_usd = pnl_pct * (qty * entry_price)
            
            print(f"   └─ {symbol}: Held {hold_hours:.1f}h | P&L: {pnl_pct:.2%} (${pnl_usd:+.2f})")
            
            # Exit conditions
            exit_signal = False
            exit_reason = ""
            
            if pnl_pct >= 0.025:  # 2.5% profit
                exit_signal = True
                exit_reason = f"take_profit_{pnl_pct:.2%}"
            elif pnl_pct <= -0.02:  # 2% loss
                exit_signal = True
                exit_reason = f"stop_loss_{pnl_pct:.2%}"
            elif hold_hours >= 48:  # 48 hour max
                exit_signal = True
                exit_reason = f"max_hold_{hold_hours:.0f}h"
            
            if exit_signal:
                try:
                    self.broker.submit_order(symbol, qty=qty, side='sell')
                    self._log_trade(symbol, 'SELL', current_price, qty, pnl_usd, exit_reason)
                    del self.positions[symbol]
                    print(f"   ✅ EXITED {symbol}: {exit_reason}")
                except Exception as e:
                    print(f"   ❌ Exit failed: {e}")
            else:
                print(f"   💎 HOLDING {symbol}: Target 2.5% profit, currently {pnl_pct:.2%}")
        
        # Look for new entries
        if len(self.positions) < 3:
            print(f"\n🔍 Looking for new entries (current positions: {len(self.positions)}/3)...")
            
            for symbol in self.symbols:
                if symbol in self.positions:
                    continue
                
                df = self.fetch_data(symbol)
                if df is None or len(df) < 50:
                    continue
                
                current_price = float(df.iloc[-1]['close'])
                signal = self.strategy.analyze(df, has_position=False, symbol=symbol)
                
                if signal and signal.get('action') == 'BUY':
                    confidence = signal.get('confidence', 0)
                    
                    if confidence >= config.MIN_CONFIDENCE_THRESHOLD:
                        position_size_pct = 0.075
                        qty = (equity * position_size_pct) / current_price
                        qty = round(qty, 6)
                        
                        if qty > 0:
                            try:
                                self.broker.submit_order(symbol, qty=qty, side='buy')
                                
                                self.positions[symbol] = {
                                    'entry_price': current_price,
                                    'entry_time': datetime.now(UTC).isoformat(),
                                    'qty': qty
                                }
                                
                                self._log_trade(symbol, 'BUY', current_price, qty, 0, signal.get('reason', 'entry'))
                                print(f"   ✅ ENTERED {symbol}: {qty} @ ${current_price:.2f}")
                                
                            except Exception as e:
                                print(f"   ❌ Entry failed: {e}")
        
        self._save_state()
        
        print(f"\n{'='*60}")
        print(f"✅ Cycle Complete")
        print(f"📊 Active positions: {len(self.positions)}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run_cycle()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
