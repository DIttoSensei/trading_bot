import csv
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

        # State file for persistence across GitHub runs
        self.state_path = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self.load_bot_state()

        self.day_start_equity = None
        self.journal_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._ensure_journal()

        # Track positions for hold logic
        self.positions = {}  # {symbol: {'entry_price': float, 'entry_time': str, 'qty': float}}

        self.sheet_logger = GoogleSheetLogger(
            credentials_file=os.path.join(os.path.dirname(__file__), config.GOOGLE_CREDENTIALS_FILE),
            sheet_name=config.GOOGLE_SHEETS_NAME,
        )

    def load_bot_state(self):
        """Loads trailing stop peaks, trade cooldowns, and positions across GitHub runs."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    self.trailing_stop.peaks = state.get("peaks", {})
                    self.cooldowns = state.get("cooldowns", {})
                    self.positions = state.get("positions", {})
                    print(f"✅ Loaded {len(self.positions)} positions from previous run")
            except Exception as e:
                print(f"⚠️ Failed to parse state file: {e}")
                self.cooldowns = {}
                self.positions = {}
        else:
            self.cooldowns = {}
            self.positions = {}

    def save_bot_state(self):
        """Saves memory state so the next run remembers everything."""
        state = {
            "peaks": self.trailing_stop.peaks,
            "cooldowns": self.cooldowns,
            "positions": self.positions
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
        except Exception:
            pass

    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
            end = datetime.now(UTC)
            start = end - timedelta(hours=config.LOOKBACK_HOURS)
            # Keep symbol with slash for Alpaca
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
        if drawdown > 0.08:
            drawdown_multiplier = 0.5
        elif drawdown > 0.05:
            drawdown_multiplier = 0.75

        dynamic_fraction = (config.POSITION_FRACTION * vol_multiplier * confidence_multiplier * drawdown_multiplier)
        dynamic_fraction = float(np.clip(dynamic_fraction, config.MIN_EQUITY_FRACTION, config.MAX_EQUITY_FRACTION))

        target_notional = min(equity * dynamic_fraction, config.MAX_NOTIONAL_PER_TRADE)
        target_notional = min(target_notional, buying_power * 0.95)

        if target_notional < config.MIN_NOTIONAL_PER_TRADE:
            return 0.0
        return round(max(target_notional / price, 0.0), 6)

    def run_cycle(self):
        print(f"--- [STARTUP] Run Cycle: {datetime.now(UTC)} ---")
        acc = self.broker.get_account()
        equity = float(acc.equity)
        buying_power = float(acc.buying_power)
        drawdown = self.risk.update(equity)

        if self.day_start_equity is None:
            self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)

        now_ts = datetime.now(UTC).timestamp()

        MAX_PORTFOLIO_EXPOSURE = equity * config.MAX_TOTAL_EXPOSURE_PCT
        MAX_BEAR_EXPOSURE = equity * 0.10

        for symbol in self.symbols:
            # Check if we have an existing position from previous run
            existing_position = self.positions.get(symbol)
            
            df = self.fetch_data(symbol)
            if df is None or len(df) < config.ML_TRAIN_MIN_ROWS:
                print(f"⚠️ {symbol}: Warming up.")
                continue

            # Cooldown Memory Gate
            cooldown_until = self.cooldowns.get(symbol, 0)
            if now_ts < cooldown_until:
                remaining_mins = int((cooldown_until - now_ts) / 60)
                self._log(symbol, float(df.iloc[-1]["close"]), "OUT", 0.0, 0.0, 0.0, 
                         existing_position.get('qty', 0) if existing_position else 0, 
                         equity, drawdown, False, "cooldown", config.BASE_THRESHOLD, 
                         f"cooldown_{remaining_mins}m_left")
                continue

            # Train ML model (reuse if exists)
            if symbol not in self.ml_models:
                model = MLSpecialist()
                model.train_price_model(df)
                self.ml_models[symbol] = model
            else:
                model = self.ml_models[symbol]
                # Retrain occasionally (every 24 hours)
                if np.random.random() < 0.25:  # 25% chance each run
                    model.train_price_model(df)

            feat_df = self.engineer_features(df).dropna()
            if feat_df.empty:
                continue

            ml_prob = float(model.predict(feat_df.iloc[-1]))
            tech_signal = float(technical_bot(df))

            decision = self.judge.evaluate(tech_signal, ml_prob, df)
            action, confidence, threshold, regime = (decision["action"], float(decision["confidence"]), 
                                                      float(decision["threshold"]), decision["regime"])
            volatility = float(decision.get("volatility", 0.015))

            pos = self.broker.get_position_info(symbol)
            current_qty = float(pos.get("qty", 0.0))
            
            # If we have an existing position from state, use that qty
            if existing_position and current_qty == 0:
                # Position was remembered but not in broker? Something's wrong
                print(f"⚠️ {symbol}: Position in state but not in broker - removing from state")
                del self.positions[symbol]
                current_qty = 0
            elif existing_position and current_qty > 0:
                # Use the state's entry price for P&L calculation
                entry_price = existing_position['entry_price']
                entry_time = datetime.fromisoformat(existing_position['entry_time'])
                hold_hours = (datetime.now(UTC) - entry_time).total_seconds() / 3600
                current_price = float(df.iloc[-1]["close"])
                pnl_pct = (current_price - entry_price) / entry_price
                
                print(f"   └─ HOLDING {symbol}: {hold_hours:.1f}h | P&L: {pnl_pct:.2%}")

            price = float(df.iloc[-1]["close"])
            risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)

            if not risk_ok:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, 
                         current_qty, equity, drawdown, False, regime, threshold, "risk_guard")
                continue

            position_value = current_qty * price
            has_real_position = position_value > 1.00

            # --- EXIT LOGIC (Check if we should sell) ---
            if has_real_position:
                avg_entry = float(pos.get("avg_entry_price", price))
                
                # Use stored entry price if available
                if existing_position:
                    avg_entry = existing_position['entry_price']
                    entry_time = datetime.fromisoformat(existing_position['entry_time'])
                    hold_hours = (datetime.now(UTC) - entry_time).total_seconds() / 3600
                    
                    # Force exit after max hold time
                    if hold_hours >= config.MAX_HOLD_HOURS:
                        self.broker.submit_order(symbol, "sell", current_qty, "market")
                        self.trailing_stop.on_exit(symbol)
                        self.cooldowns[symbol] = now_ts + (12 * 3600)
                        if symbol in self.positions:
                            del self.positions[symbol]
                        self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, 
                                 current_qty, equity, drawdown, True, regime, threshold, 
                                 f"max_hold_{hold_hours:.0f}h")
                        continue
                
                # Range scalper mode profit target
                if regime == "range" and price >= avg_entry * 1.0075:
                    self.broker.submit_order(symbol, "sell", current_qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (4 * 3600)
                    if symbol in self.positions:
                        del self.positions[symbol]
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, True, regime, threshold, 
                             "RANGE_SCALP_TAKE_PROFIT_0.75%")
                    continue

                # Trailing stop logic
                self.trailing_stop.update_peak(symbol, price)
                trailing_triggered, drop_amt = self.trailing_stop.should_exit(symbol, price)

                if trailing_triggered:
                    self.broker.submit_order(symbol, "sell", current_qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (12 * 3600)
                    if symbol in self.positions:
                        del self.positions[symbol]
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, True, regime, threshold, 
                             f"TRAILING_STOP_{round(drop_amt*100, 2)}%")

                elif action == "SELL":
                    self.broker.submit_order(symbol, "sell", current_qty, "market")
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + (12 * 3600)
                    if symbol in self.positions:
                        del self.positions[symbol]
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, True, regime, threshold, "judge_exit")

                else:
                    self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, False, regime, threshold, "shadow_holding")

            # --- ENTRY LOGIC ---
            elif action == "BUY" and confidence >= threshold and not has_real_position:
                current_positions = self.broker.get_all_positions()
                total_deployed_capital = sum(float(p.qty) * float(p.current_price) for p in current_positions)

                if regime == "bear_trend" and total_deployed_capital >= MAX_BEAR_EXPOSURE:
                    self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, False, regime, threshold, "skip_bear_exposure_limit")
                    continue
                elif total_deployed_capital >= MAX_PORTFOLIO_EXPOSURE:
                    self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, 
                             current_qty, equity, drawdown, False, regime, threshold, "skip_max_portfolio_limit")
                    continue

                buy_qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)

                if regime == "bear_trend":
                    buy_qty = round(buy_qty * 0.40, 6)

                if buy_qty > 0:
                    self.broker.submit_order(symbol=symbol, qty=buy_qty, side="buy", type="market")
                    # Store position for future runs
                    self.positions[symbol] = {
                        'entry_price': price,
                        'entry_time': datetime.now(UTC).isoformat(),
                        'qty': buy_qty
                    }
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, 
                             buy_qty, equity, drawdown, True, regime, threshold, "entry")

            else:
                self._log(symbol, price, "OUT", confidence, tech_signal, ml_prob, 
                         current_qty, equity, drawdown, False, regime, threshold, "waiting_for_entry")

        # Persist memory before the container shuts down
        self.save_bot_state()

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run_cycle()
        print(f"--- [SUCCESS] Cycle finished at {datetime.now(UTC)} ---")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
