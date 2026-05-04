import csv
import os
import sys
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
        self.trailing_stop = TrailingStopTracker(atr_multiplier=config.ATR_STOP_MULTIPLIER)

        self.entry_price: dict[str, float] = {}
        self.day_start_equity = None
        self.journal_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._ensure_journal()
        
        self.sheet_logger = GoogleSheetLogger(
            credentials_file=os.path.join(os.path.dirname(__file__), config.GOOGLE_CREDENTIALS_FILE),
            sheet_name=config.GOOGLE_SHEETS_NAME,
        )

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
        # Your specific feature engineering logic
        d["return_1h"] = d["close"].pct_change(1)
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1
        # (Add other features here as needed)
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
        
        # Track daily loss
        if self.day_start_equity is None: self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)

        for symbol in self.symbols:
            df = self.fetch_data(symbol)
            if df is None or len(df) < config.ML_TRAIN_MIN_ROWS:
                print(f"⚠️ {symbol}: Warming up.")
                continue

            # Retrain and Predict
            model = MLSpecialist()
            model.train_price_model(df)
            feat_df = self.engineer_features(df).dropna()
            if feat_df.empty: continue
            
            ml_prob = float(model.predict(feat_df.iloc[-1]))
            tech_signal = float(technical_bot(df))
            
            # Judge Decision
            decision = self.judge.evaluate(tech_signal, ml_prob, df)
            action, confidence, threshold, regime = decision["action"], float(decision["confidence"]), float(decision["threshold"]), decision["regime"]
            volatility = float(decision.get("volatility", 0.015))

            # Current State
            pos = self.broker.get_position_info(symbol)
            qty = float(pos.get("qty", 0.0))
            price = float(df.iloc[-1]["close"])
            
            risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)

            if not risk_ok:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, "risk_guard")
                continue

            # --- BUY LOGIC ---
            if action == "BUY" and confidence >= threshold and qty <= 0:
                buy_qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)
                if buy_qty > 0:
                    self.broker.submit_order(
                        symbol=symbol, qty=buy_qty, side="buy", type="market", time_in_force="gtc",
                        order_class="bracket",
                        take_profit={"limit_price": round(price * (1 + config.TAKE_PROFIT_PCT), 2)},
                        stop_loss={"stop_price": round(price * (1 - config.STOP_LOSS_PCT), 2)},
                    )
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, buy_qty, equity, drawdown, True, regime, threshold, "bracket_entry")

            # --- PROBE LOGIC ---
            elif (config.ENABLE_PROBE_ENTRY and action == "HOLD" and confidence >= config.PROBE_CONFIDENCE and ml_prob > 0.58 and qty <= 0):
                full_qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)
                probe_qty = round(full_qty * config.PROBE_SIZE_MULTIPLIER, 6)
                if probe_qty > 0:
                    self.broker.submit_order(symbol=symbol, qty=probe_qty, side="buy", type="market", time_in_force="gtc")
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, probe_qty, equity, drawdown, True, regime, threshold, "probe_entry")

            # --- EXIT LOGIC ---
            elif qty > 0:
                self.trailing_stop.update_peak(symbol, price) # Sync tracker
                trailing_triggered, _ = self.trailing_stop.should_exit(symbol, price, df)
                
                if (action == "SELL" or trailing_triggered) and qty > 0:
                    note = "trailing_stop" if trailing_triggered else "shadow_exit"
                    self.broker.submit_order(symbol, "sell", qty, "market", "gtc")
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, regime, threshold, note)
                elif (action == "SELL" or trailing_triggered):
                     print(f"ℹ️ {symbol}: Model signaled SELL, but quantity is 0. Skipping.")
                else:
                    # If we don't own it and aren't buying, it's just 'WAITING' or 'MONITORING'
                    display_action = "HOLD" if qty > 0 else "OUT" 
                    note = "shadow_holding" if qty > 0 else "waiting_for_entry"
                    
                    self._log(symbol, price, display_action, confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, note)

if __name__ == "__main__":
    try:
        TradingBot().run_cycle()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
