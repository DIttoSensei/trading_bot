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
from risk_manager import RiskManager
from position_tracker import PositionTracker

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
            
            request = CryptoBarsRequest(
                symbol_or_symbols=[symbol.replace('/USD', '')],
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
        """
        Calculate position size with profit focus:
        - Never over-allocate
        - Scale with confidence
        - Respect max exposure
        """
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
        if min_profit_dollars < 5.0:  # Less than $5 expected profit isn't worth it
            return 0.0
        
        qty = target_notional / price
        return round(qty, 6)
    
    def check_spread(self, symbol: str) -> bool:
        """Check if spread is acceptable (prevents losing money to spread)"""
        try:
            # Get current bid/ask (simplified - Alpaca provides this)
            # For now, return True - implement with real data
            return True
        except:
            return True  # Assume good if can't check
    
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
        current_exposure = sum(float(p.qty) * float(p.current_price) for p in current_positions) / account_equity
        
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
            signal = self.strategy.analyze(df, current_qty > 0)
            
            if signal is None:
                print(f"⏸️ No clear signal for {symbol}")
                continue
            
            print(f"🎯 Signal: {signal['action']} | Confidence: {signal['confidence']:.2%}")
            print(f"📊 Regime: {signal['regime']} | Expected P&L: {signal.get('expected_return', 0):.2%}")
            
            # --- EXIT LOGIC (Close positions first) ---
            if current_qty > 0 and signal['action'] == 'SELL':
                # Check if we should exit
                should_exit = False
                exit_reason = ""
                
                # Force exit after max hold time
                hold_hours = self.positions.get_hold_hours(symbol)
                if hold_hours >= config.MAX_HOLD_HOURS:
                    should_exit = True
                    exit_reason = f"max_hold_{config.MAX_HOLD_HOURS}h"
                
                # Trail stop or take profit
                elif signal.get('take_profit') or signal.get('trailing_stop'):
                    should_exit = True
                    exit_reason = "take_profit" if signal.get('take_profit') else "trailing_stop"
                
                # Judge says sell
                elif signal['action'] == 'SELL':
                    should_exit = True
                    exit_reason = "signal_exit"
                
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
                
                if signal.get('expected_return', 0) < config.MIN_PROFIT_TARGET_PCT:
                    print(f"❌ Expected return too low: {signal.get('expected_return', 0):.2%} < {config.MIN_PROFIT_TARGET_PCT:.2%}")
                    continue
                
                if not self.check_spread(symbol):
                    print(f"❌ Spread too high for {symbol}")
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
                    unrealized_pnl = (current_price - self.positions.get_entry_price(symbol)) * current_qty
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
        sys.exit(1)import csv
import os
import sys
import json
import traceback
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from broker import Broker
from layer1_technical import technical_bot
from layer3_judge import LLMJudge
from ml_layer import MLSpecialist
from risk import RiskManager, TrailingStopTracker
from sheet_logger import GoogleSheetLogger

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

class TradingBot:
    def __init__(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise ValueError("Missing Alpaca API keys.")

        self.symbols = config.TRADE_SYMBOLS
        self.broker = Broker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()

        self.ml_models: dict[str, MLSpecialist] = {}
        self.judge = LLMJudge()
        self.risk = RiskManager(max_drawdown=config.MAX_DRAWDOWN)
        self.trailing_stop = TrailingStopTracker()

        # Define path for memory persistence layer
        self.state_path = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self.load_bot_state()

        self.day_start_equity = None
        self.journal_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._ensure_journal()

        self.sheet_logger = GoogleSheetLogger(
            credentials_file=os.path.join(os.path.dirname(__file__), config.GOOGLE_CREDENTIALS_FILE),
            sheet_name=config.GOOGLE_SHEETS_NAME,
        )

    def load_bot_state(self):
        """Loads trailing stop peaks and trade cooldowns across separate GitHub runs."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    self.trailing_stop.peaks = state.get("peaks", {})
                    self.cooldowns = state.get("cooldowns", {})
            except Exception as e:
                print(f"⚠️ Failed to parse state file, resetting memory: {e}")
                self.cooldowns = {}
        else:
            self.cooldowns = {}

    def save_bot_state(self):
        """Saves memory state so the next 4-hour workflow run remembers it."""
        state = {
            "peaks": self.trailing_stop.peaks,
            "cooldowns": self.cooldowns
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=4)

    def _ensure_journal(self):
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp_utc", "symbol", "price", "action",
                    "confidence", "tech_signal", "ml_prob",
                    "position_qty", "equity", "drawdown", "traded",
                    "regime", "threshold", "note",
                ])

    def _log(self, symbol, price, action, confidence, tech_signal, ml_prob,
             position_qty, equity, drawdown, traded, regime, threshold, note):
        row = [
            datetime.now(UTC).isoformat(),
            symbol,
            round(float(price), 6),
            action,
            round(float(confidence), 6),
            round(float(tech_signal), 6),
            round(float(ml_prob), 6),
            round(float(position_qty), 6),
            round(float(equity), 2),
            round(float(drawdown), 6),
            int(bool(traded)),
            regime,
            round(float(threshold), 6),
            note,
        ]
        with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        try:
            self.sheet_logger.log_row(row)
        except Exception: pass

    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
            end = datetime.now(UTC)
            start = end - timedelta(hours=config.LOOKBACK_HOURS)
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Hour, start=start, end=end)
            bars = self.data_client.get_crypto_bars(req).df.reset_index()
            bars = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
            return bars.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"❌ Data fetch failed for {symbol}: {e}")
            return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["return_1h"] = d["close"].pct_change(1)
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1
        if "timestamp" in d.columns:
            d["hour_sin"] = np.sin(2 * np.pi * pd.to_datetime(d["timestamp"]).dt.hour / 24)
        return d

    def compute_buy_qty(self, price, buying_power, equity, confidence, volatility, drawdown) -> float:
        baseline_vol = 0.015
        vol_multiplier = np.clip(baseline_vol / max(volatility, 1e-6), 0.5, 1.4)
        confidence_multiplier = np.clip((confidence - config.BASE_THRESHOLD) / 0.10, 0.35, 1.25)

        drawdown_multiplier = 1.0
        if drawdown > 0.08: drawdown_multiplier = 0.5
        elif drawdown > 0.05: drawdown_multiplier = 0.75

        dynamic_fraction = (config.POSITION_FRACTION * vol_multiplier * confidence_multiplier * drawdown_multiplier)
        dynamic_fraction = float(np.clip(dynamic_fraction, config.MIN_EQUITY_FRACTION, config.MAX_EQUITY_FRACTION))

        target_notional = min(equity * dynamic_fraction, config.MAX_NOTIONAL_PER_TRADE)
        target_notional = min(target_notional, buying_power * 0.95)

        if target_notional < config.MIN_NOTIONAL_PER_TRADE: return 0.0
        return round(max(target_notional / price, 0.0), 6)

    def run_cycle(self):
        print(f"--- [STARTUP] Run Cycle: {datetime.now(UTC)} ---")
        acc = self.broker.get_account()
        equity = float(acc.equity)
        buying_power = float(acc.buying_power)
        drawdown = self.risk.update(equity)

        if self.day_start_equity is None: self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)

        now_ts = datetime.now(UTC).timestamp()

        # --- STRUCTURAL RISK CONTROLS ---
        # 1. Total Portfolio Allocation Limit (Absolute Cap: 25% of total equity)
        MAX_PORTFOLIO_EXPOSURE = equity * 0.25 

        # 2. Bear Market Allocation Limit (Strict Cap: 10% of total equity)
        MAX_BEAR_EXPOSURE = equity * 0.10

        for symbol in self.symbols:
            df = self.fetch_data(symbol)
            if df is None or len(df) < config.ML_TRAIN_MIN_ROWS:
                print(f"⚠️ {symbol}: Warming up.")
                continue

            # Cooldown Memory Gate: Sideline asset if exited recently
            cooldown_until = self.cooldowns.get(symbol, 0)
            if now_ts < cooldown_until:
                remaining_mins = int((cooldown_until - now_ts) / 60)
                self._log(symbol, float(df.iloc[-1]["close"]), "OUT", 0.0, 0.0, 0.0, 0.0, equity, drawdown, False, "cooldown", config.BASE_THRESHOLD, f"cooldown_{remaining_mins}m_left")
                continue

            # Original training engine preserved intact
            model = MLSpecialist()
            model.train_price_model(df)
            feat_df = self.engineer_features(df).dropna()
            if feat_df.empty: continue

            ml_prob = float(model.predict(feat_df.iloc[-1]))
            tech_signal = float(technical_bot(df))

            decision = self.judge.evaluate(tech_signal, ml_prob, df)
            action, confidence, threshold, regime = decision["action"], float(decision["confidence"]), float(decision["threshold"]), decision["regime"]
            volatility = float(decision.get("volatility", 0.015))

            pos = self.broker.get_position_info(symbol)
            qty = float(pos.get("qty", 0.0))
            price = float(df.iloc[-1]["close"])

            risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)

            if not risk_ok:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "risk_guard")
                continue

            # Calculate actual dollar value of position to dynamically bypass exchange dust
            position_value = qty * price
            has_real_position = position_value > 1.00

            # --- THE PRECISION DECISION CHAIN ---

            # 1. Entry Logic: Only try to execute a buy if we aren't holding a real position
            if action == "BUY" and confidence >= threshold and not has_real_position:

                # Fetch fresh account state to see current total exposure before buying
                current_positions = self.broker.get_all_positions()

                # FIXED: Dot notation usage for native Alpaca Position objects
                total_deployed_capital = sum(float(p.qty) * float(p.current_price) for p in current_positions)

                # Check exposure limits based on market conditions
                if regime == "bear_trend" and total_deployed_capital >= MAX_BEAR_EXPOSURE:
                    self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "skip_bear_exposure_limit")
                    continue
                elif total_deployed_capital >= MAX_PORTFOLIO_EXPOSURE:
                    self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "skip_max_portfolio_limit")
                    continue

                buy_qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)

                # Dynamic allocation scaling: Cut buy sizes by 60% if inside a bear market
                if regime == "bear_trend":
                    buy_qty = round(buy_qty * 0.40, 6)

                if buy_qty > 0:
                    self.broker.submit_order(symbol=symbol, qty=buy_qty, side="buy", type="market")
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, buy_qty, equity, drawdown, True, regime, threshold, "entry")

            # 2. Exit Logic: Manage open positions using the saved trailing peaks
            elif has_real_position and qty > 0.0001:
                avg_entry = float(pos.get("avg_entry_price", price))

                # --- NEW HYBRID SWITCH: SCALPER MODE PROFIT TARGET ---
                if regime == "range" and price >= avg_entry * 1.0075:
                    self.broker.submit_order(symbol, "sell", qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (4 * 3600)  # Shorter 4h range cooldown
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, regime, threshold, "RANGE_SCALP_TAKE_PROFIT_0.75%")
                    continue

                # --- TREND MODE FALLBACK: TRAILING STOP ---
                self.trailing_stop.update_peak(symbol, price) 
                trailing_triggered, drop_amt = self.trailing_stop.should_exit(symbol, price)

                if trailing_triggered:
                    self.broker.submit_order(symbol, "sell", qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (12 * 3600)  # Lock memory for 12 hours
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, regime, threshold, f"PROFIT_LOCK_{round(drop_amt*100, 2)}%")

                elif action == "SELL":
                    self.broker.submit_order(symbol, "sell", qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (12 * 3600)  # Lock memory for 12 hours
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, regime, threshold, "judge_exit")

                else:
                    self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "shadow_holding")

            # 3. Flat Cache: No open position and no strong buy signal (or it's just exchange dust)
            else:
                self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "waiting_for_entry")

        # Persist memory right before the container shuts down
        self.save_bot_state()


if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run_cycle()
        print(f"--- [SUCCESS] Cycle finished at {datetime.now(UTC)} ---")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
