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
if config.ENABLE_BACKTEST_GATE:
    from backtester import walk_forward_gate

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
        self.next_decision_at = datetime.now(UTC)
        self.entry_price = {}
        self.day_start_equity = None

        self.journal_path = os.path.join(os.path.dirname(__file__), config.TRADE_LOG_CSV)
        self._ensure_journal()
        self.sheet_logger = GoogleSheetLogger(
            credentials_file=os.path.join(os.path.dirname(__file__), config.GOOGLE_CREDENTIALS_FILE),
            sheet_name=config.GOOGLE_SHEETS_NAME,
        )

    def _ensure_journal(self):
        if os.path.exists(self.journal_path):
            return
        with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_utc",
                    "symbol",
                    "price",
                    "action",
                    "confidence",
                    "tech_signal",
                    "ml_prob",
                    "position_qty",
                    "equity",
                    "drawdown",
                    "traded",
                    "regime",
                    "threshold",
                    "note",
                ]
            )

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
            note,
        ]
        with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.sheet_logger.log_row(row)

    def fetch_data(self, symbol: str) -> pd.DataFrame:
        end = datetime.now(UTC)
        start = end - timedelta(hours=config.LOOKBACK_HOURS)
        req = CryptoBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start,
            end=end,
        )

        bars = self.data_client.get_crypto_bars(req).df.reset_index()
        bars = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        bars = bars.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return bars

    def train_model(self, symbol: str, df: pd.DataFrame):
        model = self.ml_models.get(symbol)
        if model is None:
            model = MLSpecialist()
            self.ml_models[symbol] = model
        model.train_price_model(df)
        self.last_retrain_at[symbol] = datetime.now(UTC)

    def should_retrain(self, symbol: str):
        last = self.last_retrain_at.get(symbol)
        if last is None:
            return True
        age = datetime.now(UTC) - last
        return age >= timedelta(hours=config.MODEL_RETRAIN_HOURS)

    def get_account_state(self):
        acc = self.broker.get_account()
        equity = float(acc.equity)
        buying_power = float(acc.buying_power)
        if self.day_start_equity is None or datetime.now(UTC).hour == 0:
            self.day_start_equity = equity
        daily_loss = 0.0
        if self.day_start_equity:
            daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)
        return equity, buying_power, daily_loss

    def compute_buy_qty(self, price, buying_power, equity, confidence, market_vol, drawdown):
        baseline_vol = 0.015
        vol_multiplier = np.clip(baseline_vol / max(market_vol, 1e-6), 0.5, 1.4)
        confidence_multiplier = np.clip((confidence - 0.5) / 0.15, 0.35, 1.25)
        drawdown_multiplier = 1.0
        if drawdown > 0.08:
            drawdown_multiplier = 0.6
        elif drawdown > 0.05:
            drawdown_multiplier = 0.8

        dynamic_fraction = config.POSITION_FRACTION * vol_multiplier * confidence_multiplier * drawdown_multiplier
        dynamic_fraction = min(
            max(dynamic_fraction, config.MIN_EQUITY_FRACTION),
            config.MAX_EQUITY_FRACTION,
        )
        target_from_equity = equity * dynamic_fraction
        target_from_buying_power = buying_power * dynamic_fraction
        target_notional = min(
            max(target_from_equity, target_from_buying_power),
            config.MAX_NOTIONAL_PER_TRADE,
        )
        if target_notional < config.MIN_NOTIONAL_PER_TRADE:
            return 0.0
        qty = target_notional / price
        return round(max(qty, 0.0), 6)

    def evaluate_and_trade(self, symbol: str, df: pd.DataFrame):
        if df is None or df.empty:
            print(f"{symbol}: No market data. HOLD.")
            return

        price = float(df.iloc[-1]["close"])
        equity, buying_power, daily_loss = self.get_account_state()
        drawdown = self.risk.update(equity)
        position_info = self.broker.get_position_info(symbol)
        position_qty = float(position_info.get("qty", 0.0))
        live_entry_price = float(position_info.get("avg_entry_price", 0.0))

        if len(df) < config.ML_TRAIN_MIN_ROWS:
            print("Warming up data... HOLD.")
            return

        if self.should_retrain(symbol):
            self.train_model(symbol, df)

        # Layers
        tech_signal = float(technical_bot(df))
        
        # Feature engineering for ML
        feat = df.copy()
        feat["return"] = feat["close"].pct_change()
        feat["momentum_3h"] = feat["close"] - feat["close"].shift(3)
        feat["momentum_24h"] = feat["close"] - feat["close"].shift(24)
        feat["volatility_24h"] = feat["return"].rolling(24).std()
        feat["ma_50_dist"] = feat["close"] / feat["close"].rolling(50).mean()
        feat = feat.dropna()
        
        ml_prob = float(self.ml_models[symbol].predict(feat.iloc[-1]))
        
        # --- THE SHADOW JUDGE ---
        decision = self.judge.evaluate(tech_signal, ml_prob, df)
        action = decision["action"]
        confidence = float(decision["confidence"])
        decision_threshold = float(decision.get("threshold", config.MIN_BUY_CONFIDENCE))
        
        # Display the Monster's Thought Process
        print("\n==============================")
        print(f"Time: {datetime.now(UTC).isoformat()}")
        print(f"{symbol} | Price: {price:.2f} | Balance: {equity:.2f}")
        print(f"Shadow Win Prob: {decision.get('shadow_win_prob', 0):.3f} | Confidence: {confidence:.3f}")
        print(f"Shadow Target: ${decision.get('expected_future_price', 0):.2f}")
        print(f"Regime: {decision.get('regime', 'unknown')} | Threshold: {decision_threshold:.3f}")
        print(f"Action: {action} | Drawdown: {drawdown:.3%}")

        risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)
        if not risk_ok:
            print("Risk Guard Active. HOLD.")
            return

        # Buy Logic
        if action == "BUY" and confidence >= decision_threshold and position_qty <= 0:
            qty = self.compute_buy_qty(price, buying_power, equity, confidence, 0.015, drawdown)
            if qty > 0:
                self.broker.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="gtc",
                    order_class="bracket", # This links the buy to the safety exits
                    stop_loss={'stop_price': price * (1 - config.STOP_LOSS_PCT)},
                    take_profit={'limit_price': price * (1 + config.TAKE_PROFIT_PCT)}
                )
        # Sell/Hold Logic
        elif position_qty > 0:
            entry_ref = live_entry_price if live_entry_price > 0 else self.entry_price.get(symbol)
            pnl = (price - entry_ref) / entry_ref if entry_ref else 0
            
            # The bot sells if the Shadow Signal turns negative OR if hard stops hit
            exit_on_signal = action == "SELL"
            exit_on_sl = pnl <= -config.STOP_LOSS_PCT
            
            if exit_on_signal or exit_on_sl:
                self.broker.submit_order(symbol, "SELL", position_qty)
                note = "shadow_exit" if exit_on_signal else "stop_loss"
                self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, True, decision['regime'], decision_threshold, note)
            else:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, decision['regime'], decision_threshold, "shadow_holding")
    # Fix: Added pre-flight check to sync with Alpaca wallet at startup
    def pre_flight_check(self):
        print("\n--- [STARTUP] Pre-Flight Position Audit ---")
        for symbol in self.symbols:
            position_info = self.broker.get_position_info(symbol)
            qty = position_info["qty"]
            
            if qty > 0:
                print(f"Found existing position: {qty} units of {symbol}")
                df = self.fetch_data(symbol)
                if df is not None and not df.empty:
                    self.evaluate_and_trade(symbol, df)
                else:
                    print(f"Could not fetch data for {symbol}. Keeping position.")
            else:
                print(f"No existing position found for {symbol}.")
        print("--- Audit Complete. Entering Main Loop. ---\n")

    def run(self):
        print("=== Self-Reliant Paper Trading Bot Started ===")
        print(f"Symbols: {', '.join(self.symbols)} | Decision interval: {config.DECISION_INTERVAL_HOURS}h")

        # Fix: Run the audit before the main loop
        self.pre_flight_check()

        if config.BOT_RUN_ONCE:
            print("BOT_RUN_ONCE enabled. Finished single cycle.")
            return

        while True:
            try:
                now = datetime.now(UTC)
                if now >= self.next_decision_at:
                    for symbol in self.symbols:
                        df = self.fetch_data(symbol)
                        self.evaluate_and_trade(symbol, df)
                    self.next_decision_at = now + timedelta(hours=config.DECISION_INTERVAL_HOURS)
                else:
                    print(f"Waiting for decision time. Next: {self.next_decision_at.isoformat()}")
            except Exception as exc:
                print(f"Loop error: {exc}")
            time.sleep(max(60, config.DATA_REFRESH_MINUTES * 60))


if __name__ == "__main__":
    TradingBot().run()
