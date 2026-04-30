import csv
import os
import time
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
from risk import RiskManager
from sheet_logger import GoogleSheetLogger

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

class TradingBot:
    def __init__(self):
        if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
            raise ValueError("Missing Alpaca API keys in environment variables.")

        self.symbols = config.TRADE_SYMBOLS
        self.broker = Broker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()

        self.ml_models = {}
        self.judge = LLMJudge()
        self.risk = RiskManager(max_drawdown=config.MAX_DRAWDOWN)

        self.last_retrain_at = {}
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
                writer.writerow(["timestamp_utc", "symbol", "price", "action", "confidence", "tech_signal", "ml_prob", "position_qty", "equity", "drawdown", "traded", "regime", "threshold", "note"])

    def _log(self, symbol, price, action, confidence, tech_signal, ml_prob, position_qty, equity, drawdown, traded, regime, threshold, note):
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
            note
        ]
        with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.sheet_logger.log_row(row)

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        end = datetime.now(UTC)
        start = end - timedelta(hours=config.LOOKBACK_HOURS)
        try:
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Hour, start=start, end=end)
            bars = self.data_client.get_crypto_bars(req).df.reset_index()
            return bars[["timestamp", "open", "high", "low", "close", "volume"]].copy().sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None

    def get_account_state(self):
        acc = self.broker.get_account()
        equity = float(acc.equity)
        buying_power = float(acc.buying_power)
        if self.day_start_equity is None or datetime.now(UTC).hour == 0:
            self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity) if self.day_start_equity else 0.0
        return equity, buying_power, daily_loss

    def compute_buy_qty(self, price, buying_power, equity, confidence, market_vol, drawdown):
        """
        DYNAMIC POSITION SIZING:
        Scales the trade based on a % of total account equity.
        """
        # 1. Base Risk (Percentage of total account per trade)
        risk_fraction = config.POSITION_FRACTION # Suggest 0.05 to 0.10
        
        # 2. Confidence Adjustment
        # If confidence is 0.7, multiplier is ~1.3. If 0.5, multiplier is 1.0.
        conf_multiplier = np.clip((confidence - 0.5) / 0.15, 0.5, 1.5)
        
        # 3. Market Volatility Adjustment
        vol_multiplier = np.clip(0.015 / max(market_vol, 1e-6), 0.7, 1.3)

        # 4. Final Notional Calculation
        target_notional = equity * risk_fraction * conf_multiplier * vol_multiplier
        
        # 5. Cap by configured MAX and available buying power
        target_notional = min(target_notional, config.MAX_NOTIONAL_PER_TRADE)
        
        # Hard safety: Never use more than 95% of actual available cash
        max_allowed = buying_power * 0.95
        final_notional = min(target_notional, max_allowed)

        if final_notional < config.MIN_NOTIONAL_PER_TRADE:
            return 0.0

        return round(final_notional / price, 6)

    def evaluate_and_trade(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty: return

        price = float(df.iloc[-1]["close"])
        equity, buying_power, daily_loss = self.get_account_state()
        drawdown = self.risk.update(equity)
        pos = self.broker.get_position_info(symbol)
        position_qty = float(pos.get("qty", 0.0))
        entry_price = float(pos.get("avg_entry_price", 0.0))

        if self.should_retrain(symbol): self.train_model(symbol, df)

        # 1. SIGNAL GENERATION
        tech_signal = float(technical_bot(df))
        
        feat = df.copy()
        feat["return"] = feat["close"].pct_change()
        feat["momentum_3h"] = feat["close"] - feat["close"].shift(3)
        feat["momentum_24h"] = feat["close"] - feat["close"].shift(24)
        feat["volatility_24h"] = feat["return"].rolling(24).std()
        feat["ma_50_dist"] = feat["close"] / feat["close"].rolling(50).mean()
        
        ml_prob = float(self.ml_models[symbol].predict(feat.dropna().iloc[-1]))

        # 2. JUDGE & SHADOW FORECAST
        decision = self.judge.evaluate(symbol, tech_signal, ml_prob, df)
        action = decision["action"]
        confidence = decision["confidence"]
        threshold = decision["threshold"]
        
        # GLITCH PROTECTION: Prevent explosive forecasts from tiny price decimals
        shadow_risk = decision.get("shadow_risk_floor", price * 0.95)
        forecast_price = decision['expected_future_price']
        if forecast_price > (price * 2): # If forecast predicts 100% gain in 8 hours, it's a bug
            forecast_price = price
            confidence = min(confidence, 0.50)

        print(f"[{symbol}] Price: {price:.4f} | Conf: {confidence:.3f} | Forecast: {forecast_price:.4f}")

        if not (self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)):
            print(f"Risk Guard Active for {symbol}.")
            return

        # 3. EXECUTION LOGIC
        if action == "BUY" and position_qty <= 0:
            qty = self.compute_buy_qty(price, buying_power, equity, confidence, 0.015, drawdown)
            if qty > 0:
                try:
                    self.broker.submit_order(symbol=symbol, qty=qty, side="buy")
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, decision['regime'], threshold, "shadow_entry")
                except Exception as e:
                    print(f"Order Error: {e}")

        elif position_qty > 0:
            pnl = (price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Exit Conditions
            exit_sl = pnl <= -config.STOP_LOSS_PCT
            exit_tp = pnl >= config.TAKE_PROFIT_PCT
            exit_sig = (action == "SELL")
            exit_shadow = price < shadow_risk # Price fell below the simulated worst-case floor

            if exit_sl or exit_tp or exit_sig or exit_shadow:
                try:
                    self.broker.submit_order(symbol=symbol, qty=position_qty, side="sell")
                    reason = "stop_loss" if exit_sl else ("take_profit" if exit_tp else ("shadow_floor" if exit_shadow else "shadow_exit"))
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, True, decision['regime'], threshold, reason)
                except Exception as e:
                    print(f"Exit Error: {e}")
            else:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, decision['regime'], threshold, "shadow_holding")

    def should_retrain(self, symbol):
        last = self.last_retrain_at.get(symbol)
        return last is None or (datetime.now(UTC) - last) >= timedelta(hours=config.MODEL_RETRAIN_HOURS)

    def train_model(self, symbol, df):
        if symbol not in self.ml_models: self.ml_models[symbol] = MLSpecialist()
        self.ml_models[symbol].train_price_model(df)
        self.last_retrain_at[symbol] = datetime.now(UTC)

    def run(self):
        print("=== Self-Reliant Trading Bot Started ===")
        
        # Main Portfolio Loop: Always evaluates all targets
        for s in self.symbols:
            print(f"\n--- Checking {s} ---")
            df = self.fetch_data(s)
            if df is not None:
                self.evaluate_and_trade(s, df)
        
        print("\n--- Cycle Complete ---")
        
        if config.BOT_RUN_ONCE: 
            return
        
        while True:
            time.sleep(config.DECISION_INTERVAL_HOURS * 3600)
            for s in self.symbols:
                df = self.fetch_data(s)
                if df is not None: 
                    self.evaluate_and_trade(s, df)

if __name__ == "__main__":
    TradingBot().run()
