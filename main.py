import csv
import os
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from backtester import walk_forward_gate
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

        self.symbol = config.SYMBOL
        self.broker = Broker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient()

        self.ml = MLSpecialist()
        self.judge = LLMJudge()
        self.risk = RiskManager(max_drawdown=config.MAX_DRAWDOWN)

        self.last_retrain_at = None
        self.next_decision_at = datetime.now(UTC)
        self.entry_price = None
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
                    "note",
                ]
            )

    def _log(self, price, action, confidence, tech_signal, ml_prob, position_qty, equity, drawdown, note):
        row = [
            datetime.now(UTC).isoformat(),
            self.symbol,
            round(float(price), 6),
            action,
            round(float(confidence), 6),
            round(float(tech_signal), 6),
            round(float(ml_prob), 6),
            round(float(position_qty), 6),
            round(float(equity), 2),
            round(float(drawdown), 6),
            note,
        ]
        with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        self.sheet_logger.log_row(row)

    def _log_hold_event(self, note: str, price: float = 0.0):
        try:
            equity, _, _ = self.get_account_state()
            drawdown = self.risk.update(equity)
            position_qty = self.get_position_qty()
        except Exception:
            equity = 0.0
            drawdown = 0.0
            position_qty = 0.0
        self._log(
            price=price,
            action="HOLD",
            confidence=0.0,
            tech_signal=0.0,
            ml_prob=0.0,
            position_qty=position_qty,
            equity=equity,
            drawdown=drawdown,
            note=note,
        )

    def fetch_data(self) -> pd.DataFrame:
        end = datetime.now(UTC)
        start = end - timedelta(hours=config.LOOKBACK_HOURS)
        req = CryptoBarsRequest(
            symbol_or_symbols=[self.symbol],
            timeframe=TimeFrame.Hour,
            start=start,
            end=end,
        )

        bars = self.data_client.get_crypto_bars(req).df.reset_index()
        bars = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        bars = bars.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return bars

    def train_model(self, df: pd.DataFrame):
        self.ml.train_price_model(df)
        self.last_retrain_at = datetime.now(UTC)

    def should_retrain(self):
        if self.last_retrain_at is None:
            return True
        age = datetime.now(UTC) - self.last_retrain_at
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

    def get_position_qty(self):
        return self.broker.get_position_qty(self.symbol)

    def compute_buy_qty(self, price, buying_power, equity):
        dynamic_fraction = min(
            max(config.POSITION_FRACTION, config.MIN_EQUITY_FRACTION),
            config.MAX_EQUITY_FRACTION,
        )
        target_from_equity = equity * dynamic_fraction
        target_from_buying_power = buying_power * config.POSITION_FRACTION
        target_notional = min(
            max(target_from_equity, target_from_buying_power),
            config.MAX_NOTIONAL_PER_TRADE,
        )
        if target_notional < config.MIN_NOTIONAL_PER_TRADE:
            return 0.0
        qty = target_notional / price
        return round(max(qty, 0.0), 6)

    def evaluate_and_trade(self, df: pd.DataFrame):
        if len(df) < config.ML_TRAIN_MIN_ROWS:
            print("Not enough data for safe decision. HOLD.")
            last_price = float(df.iloc[-1]["close"]) if len(df) else 0.0
            self._log_hold_event("not_enough_data", last_price)
            return

        if self.should_retrain():
            print("Retraining ML model...")
            self.train_model(df)

        gate = walk_forward_gate(
            df,
            min_rows=config.ML_TRAIN_MIN_ROWS,
            test_window=config.BACKTEST_TEST_WINDOW,
            min_signals=config.BACKTEST_MIN_SIGNALS,
            min_winrate=config.BACKTEST_MIN_WINRATE,
        )
        if not gate["pass"]:
            print(f"Backtest gate failed ({gate['reason']}). HOLD.")
            gate_price = float(df.iloc[-1]["close"]) if len(df) else 0.0
            self._log_hold_event(f"gate_{gate['reason']}", gate_price)
            return

        row = df.iloc[-1]
        price = float(row["close"])
        tech_signal = float(technical_bot(df))

        feat = df.copy()
        feat["return"] = feat["close"].pct_change()
        feat["momentum"] = feat["close"] - feat["close"].shift(3)
        feat["volatility"] = feat["return"].rolling(5).std()
        feat = feat.dropna()
        if feat.empty:
            print("Feature frame empty after engineering. HOLD.")
            self._log_hold_event("feature_frame_empty", price)
            return

        ml_prob = float(self.ml.predict(feat.iloc[-1]))
        decision = self.judge.evaluate(tech_signal, ml_prob, df)
        action = decision["action"]
        confidence = float(decision["confidence"])

        equity, buying_power, daily_loss = self.get_account_state()
        drawdown = self.risk.update(equity)
        risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)

        position_qty = self.get_position_qty()
        print("\n==============================")
        print(f"Time: {datetime.now(UTC).isoformat()}")
        print(f"Price: {price:.2f} | Position qty: {position_qty:.6f}")
        print(f"Tech: {tech_signal:.3f} | ML: {ml_prob:.3f} | Confidence: {confidence:.3f}")
        print(f"Action: {action} | Drawdown: {drawdown:.3%} | Daily loss: {daily_loss:.3%}")

        if not risk_ok:
            self._log(price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, "risk_pause")
            print("Risk guard active. HOLD.")
            return

        if action == "BUY" and confidence >= config.MIN_BUY_CONFIDENCE and position_qty <= 0:
            qty = self.compute_buy_qty(price, buying_power, equity)
            if qty > 0:
                self.broker.submit_order(self.symbol, "BUY", qty)
                self.entry_price = price
                print(f"BUY executed: {qty} (notional ~ {qty * price:.2f})")
                self._log(price, "BUY", confidence, tech_signal, ml_prob, qty, equity, drawdown, "entry")
            else:
                self._log(price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, "small_notional")
        elif position_qty > 0:
            pnl = 0.0 if not self.entry_price else (price - self.entry_price) / self.entry_price
            exit_on_signal = action == "SELL" and confidence > 0.5
            exit_on_tp = pnl >= config.TAKE_PROFIT_PCT
            exit_on_sl = pnl <= -config.STOP_LOSS_PCT
            if exit_on_signal or exit_on_tp or exit_on_sl:
                self.broker.submit_order(self.symbol, "SELL", position_qty)
                note = "signal_exit" if exit_on_signal else ("take_profit" if exit_on_tp else "stop_loss")
                print(f"SELL executed: {position_qty} ({note})")
                self.entry_price = None
                self._log(price, "SELL", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, note)
            else:
                self._log(price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, "in_position")
        else:
            self._log(price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, "no_setup")

    def run(self):
        print("=== Self-Reliant Paper Trading Bot Started ===")
        print(f"Symbol: {self.symbol} | Decision interval: {config.DECISION_INTERVAL_HOURS}h")

        if config.BOT_RUN_ONCE:
            print("BOT_RUN_ONCE enabled. Running a single cycle.")
            df = self.fetch_data()
            self.evaluate_and_trade(df)
            return

        while True:
            try:
                df = self.fetch_data()
                now = datetime.now(UTC)
                if now >= self.next_decision_at:
                    self.evaluate_and_trade(df)
                    self.next_decision_at = now + timedelta(hours=config.DECISION_INTERVAL_HOURS)
                else:
                    print(f"Data refreshed at {now.isoformat()}. Next decision: {self.next_decision_at.isoformat()}")
            except Exception as exc:
                print(f"Loop error: {exc}")
            time.sleep(max(60, config.DATA_REFRESH_MINUTES * 60))


if __name__ == "__main__":
    TradingBot().run()