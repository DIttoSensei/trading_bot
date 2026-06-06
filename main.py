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
from ml_layer import MLSpecialist
from layer3_judge import LLMJudge
from risk import RiskManager, TrailingStopTracker
from sheet_logger import GoogleSheetLogger

class TradingBot:
    def __init__(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise ValueError("Missing essential Alpaca API access keys.")

        self.symbols = config.TRADE_SYMBOLS
        self.broker = Broker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()

        self.ml_models: dict[str, MLSpecialist] = {}
        self.judge = LLMJudge()
        self.risk = RiskManager(max_drawdown=config.MAX_DRAWDOWN)
        self.trailing_stop = TrailingStopTracker()

        # Local persistence state across detached cron actions
        self.state_path = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self.cooldowns = {}
        self.positions = {}
        self.load_bot_state()

        self.day_start_equity = None
        self.journal_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._ensure_journal()

        self.sheet_logger = GoogleSheetLogger(
            credentials_file=os.path.join(os.path.dirname(__file__), config.GOOGLE_CREDENTIALS_FILE),
            sheet_name=config.GOOGLE_SHEETS_NAME,
        )

    def load_bot_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                    self.trailing_stop.peaks = state.get("peaks", {})
                    self.cooldowns = state.get("cooldowns", {})
                    self.positions = state.get("positions", {})
                    print(f"✅ State loaded successfully. Current active tracked pairs: {list(self.positions.keys())}")
            except Exception as e:
                print(f"⚠️ State load corrupted. Reinitializing: {e}")

    def save_bot_state(self):
        try:
            state = {
                "peaks": self.trailing_stop.peaks,
                "cooldowns": self.cooldowns,
                "positions": self.positions
            }
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"❌ Failed to persist engine context state to disk: {e}")

    def _ensure_journal(self):
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp_utc", "symbol", "price", "action",
                    "confidence", "tech_signal", "ml_prob",
                    "position_qty", "equity", "drawdown", "traded",
                    "regime", "threshold", "note"
                ])

    def _log(self, *args):
        try:
            row = [
                datetime.now(UTC).isoformat(), args[0], round(float(args[1]), 4),
                args[2], round(float(args[3]), 4), round(float(args[4]), 4),
                round(float(args[5]), 4), round(float(args[6]), 4), round(float(args[7]), 2),
                round(float(args[8]), 4), int(bool(args[9])), args[10],
                round(float(args[11]), 4), str(args[12])
            ]
            with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            self.sheet_logger.log_row(row)
        except Exception as e:
            print(f"⚠️ Logging matrix indexing error: {e}")

    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
            end = datetime.now(UTC)
            start = end - timedelta(hours=config.LOOKBACK_HOURS)
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Hour, start=start, end=end)
            bars = self.data_client.get_crypto_bars(req).df.reset_index()
            return bars[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"❌ Historical ingestion dropped for target {symbol}: {e}")
            return None

    def compute_buy_qty(self, price: float, buying_power: float, equity: float, confidence: float, volatility: float) -> float:
        vol_scale = np.clip(0.015 / max(volatility, 1e-5), 0.6, 1.5)
        conf_scale = np.clip((confidence - config.BASE_THRESHOLD) / 0.12, 0.4, 1.5)
        
        target_fraction = config.POSITION_FRACTION * vol_scale * conf_scale
        target_fraction = float(np.clip(target_fraction, config.MIN_EQUITY_FRACTION, config.MAX_EQUITY_FRACTION))
        
        target_allocation = min(equity * target_fraction, config.MAX_NOTIONAL_PER_TRADE)
        target_allocation = min(target_allocation, buying_power * 0.92)
        
        if target_allocation < config.MIN_NOTIONAL_PER_TRADE:
            return 0.0
        return round(max(target_allocation / price, 0.0), 5)

    def run_cycle(self):
        print(f"⚡ Execution Initialization: {datetime.now(UTC).isoformat()}")
        acc = self.broker.get_account()
        equity, buying_power = float(acc.equity), float(acc.buying_power)
        drawdown = self.risk.update(equity)

        if self.day_start_equity is None:
            self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)
        
        now_ts = datetime.now(UTC).timestamp()

        for symbol in self.symbols:
            df = self.fetch_data(symbol)
            if df is None or len(df) < 30:
                continue

            current_price = float(df.iloc[-1]["close"])
            existing_pos = self.positions.get(symbol)
            
            # Sync local database with state of broker
            broker_pos = self.broker.get_position_info(symbol)
            current_qty = float(broker_pos.get("qty", 0.0))

            if current_qty == 0 and existing_pos:
                del self.positions[symbol]
                existing_pos = None

            # Cooldown execution gate
            if now_ts < self.cooldowns.get(symbol, 0.0):
                continue

            # Layer 2 Matrix Processing
            model = MLSpecialist()
            model.train_price_model(df)
            
            # Feature extraction vector mapping
            d_feat = df.copy()
            d_feat["return_1h"] = d_feat["close"].pct_change(1)
            d_feat["volatility_6h"] = d_feat["return_1h"].rolling(6).std()
            d_feat["ma_20_dist"] = (d_feat["close"] / d_feat["close"].rolling(20).mean()) - 1.0
            d_feat = d_feat.dropna()
            
            if d_feat.empty:
                continue
                
            ml_prob = float(model.predict(d_feat.iloc[-1]))
            tech_signal = float(technical_bot(df))

            # Consolidated Layer 3 Judge analysis
            decision = self.judge.evaluate(tech_signal, ml_prob, df)
            action, confidence, threshold, regime = decision["action"], decision["confidence"], decision["threshold"], decision["regime"]
            volatility = decision["volatility"]

            # General circuit evaluation
            if not self.risk.allow_trading(equity) or (daily_loss >= config.MAX_DAILY_LOSS_PCT):
                print("🛑 Circuit breakers triggered. Halting trade cycle additions.")
                continue

            # --- LIQUIDATION & EXIT FLOWS ---
            if current_qty > 0 and existing_pos:
                entry_price = float(existing_pos["entry_price"])
                hold_hours = (datetime.now(UTC) - datetime.fromisoformat(existing_pos["entry_time"])).total_seconds() / 3600.0
                
                # Update rolling highs
                self.trailing_stop.update_peak(symbol, current_price, entry_price)
                stop_triggered, drop_amt = self.trailing_stop.should_exit(symbol, current_price)

                # Fixed-range profit target routing
                range_take_profit = (regime == "range" and current_price >= entry_price * (1.0 + config.MIN_PROFIT_TARGET_PCT))

                if hold_hours >= config.MAX_HOLD_HOURS:
                    self.broker.submit_order(symbol, "sell", current_qty)
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + 14400
                    del self.positions[symbol]
                    self._log(symbol, current_price, "SELL", confidence, tech_signal, ml_prob, 0, equity, drawdown, True, regime, threshold, "max_hold_expiration")
                
                elif stop_triggered:
                    self.broker.submit_order(symbol, "sell", current_qty)
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + 14400
                    del self.positions[symbol]
                    self._log(symbol, current_price, "SELL", confidence, tech_signal, ml_prob, 0, equity, drawdown, True, regime, threshold, f"trailing_stop_triggered_{round(drop_amt*100,2)}%")
                
                elif range_take_profit:
                    self.broker.submit_order(symbol, "sell", current_qty)
                    self.trailing_stop.on_exit(symbol)
                    self.cooldowns[symbol] = now_ts + 7200
                    del self.positions[symbol]
                    self._log(symbol, current_price, "SELL", confidence, tech_signal, ml_prob, 0, equity, drawdown, True, regime, threshold, "range_scalp_take_profit")

            # --- CAPITAL ENGAGEMENT & ENTRIES ---
            elif action == "BUY" and current_qty == 0:
                current_portfolio = self.broker.get_all_positions()
                allocated_capital = sum(float(p.qty) * float(p.current_price) for p in current_portfolio)
                
                if allocated_capital >= (equity * config.MAX_TOTAL_EXPOSURE_PCT):
                    continue

                buy_qty = self.compute_buy_qty(current_price, buying_power, equity, confidence, volatility)
                if regime == "bear_trend":
                    buy_qty = round(buy_qty * 0.50, 5)  # Halve contract sizes during macro down trends

                if buy_qty > 0:
                    order = self.broker.submit_order(symbol, "buy", buy_qty)
                    if order:
                        self.positions[symbol] = {
                            "entry_price": current_price,
                            "entry_time": datetime.now(UTC).isoformat(),
                            "qty": buy_qty
                        }
                        self._log(symbol, current_price, "BUY", confidence, tech_signal, ml_prob, buy_qty, equity, drawdown, True, regime, threshold, "strategic_entry")

        # Atomic structural serialization commit
        self.save_bot_state()

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run_cycle()
    except Exception:
        traceback.print_exc()
        sys.exit(1)