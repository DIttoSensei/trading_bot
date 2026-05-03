import csv
import os
import time
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

        self.ml_models: dict[str, MLSpecialist] = {}
        self.judge = LLMJudge()
        self.risk = RiskManager(max_drawdown=config.MAX_DRAWDOWN)
        self.trailing_stop = TrailingStopTracker(atr_multiplier=config.ATR_STOP_MULTIPLIER)

        self.last_retrain_at: dict[str, datetime] = {}
        self.next_decision_at = datetime.now(UTC)
        self.entry_price: dict[str, float] = {}
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
            writer = csv.writer(f)
            writer.writerow(row)
        self.sheet_logger.log_row(row)

    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
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
        except Exception as e:
            print(f"❌ Data fetch failed for {symbol}: {e}")
            return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["return_1h"] = d["close"].pct_change(1)
        d["return_4h"] = d["close"].pct_change(4)
        d["return_24h"] = d["close"].pct_change(24)
        d["momentum_3h"] = d["close"] - d["close"].shift(3)
        d["momentum_12h"] = d["close"] - d["close"].shift(12)
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["volatility_24h"] = d["return_1h"].rolling(24).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1
        d["ma_50_dist"] = (d["close"] / d["close"].rolling(50).mean()) - 1

        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi"] = (100 - (100 / (1 + rs))) / 100
        d["vol_spike"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)

        if "timestamp" in d.columns:
            d["hour_sin"] = np.sin(2 * np.pi * pd.to_datetime(d["timestamp"]).dt.hour / 24)
        else:
            d["hour_sin"] = 0.0
        return d

    def train_model(self, symbol: str, df: pd.DataFrame):
        model = self.ml_models.get(symbol)
        if model is None:
            model = MLSpecialist()
            self.ml_models[symbol] = model
        model.train_price_model(df)
        self.last_retrain_at[symbol] = datetime.now(UTC)

    def should_retrain(self, symbol: str) -> bool:
        last = self.last_retrain_at.get(symbol)
        if last is None: return True
        return (datetime.now(UTC) - last) >= timedelta(hours=config.MODEL_RETRAIN_HOURS)

    def get_account_state(self) -> tuple[float, float, float]:
        acc = self.broker.get_account()
        equity = float(acc.equity)
        buying_power = float(acc.buying_power)
        if self.day_start_equity is None or datetime.now(UTC).hour == 0:
            self.day_start_equity = equity
        daily_loss = 0.0
        if self.day_start_equity:
            daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)
        return equity, buying_power, daily_loss

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

    def evaluate_and_trade(self, symbol: str, df: pd.DataFrame):
        equity, buying_power, daily_loss = self.get_account_state()
        drawdown = self.risk.update(equity)
        position_info = self.broker.get_position_info(symbol)
        position_qty = float(position_info.get("qty", 0.0))
        live_entry_price = float(position_info.get("avg_entry_price", 0.0))

        if df is None or df.empty:
            print(f"{symbol}: No market data. Skipping.")
            self._log(symbol, 0.0, "ERROR", 0.0, 0.0, 0.5, position_qty, equity, drawdown, False, "unknown", 0.0, "missing_data")
            return

        price = float(df.iloc[-1]["close"])

        # FIX: Warmup logging now happens BEFORE the return
        if len(df) < config.ML_TRAIN_MIN_ROWS:
            remaining = config.ML_TRAIN_MIN_ROWS - len(df)
            print(f"{symbol}: Warming up... ({len(df)} bars). HOLD.")
            self._log(symbol, price, "HOLD", 0.0, 0.0, 0.5, position_qty, equity, drawdown, False, "warmup", 0.0, f"warmup_{remaining}")
            return

        if self.should_retrain(symbol):
            self.train_model(symbol, df)

        tech_signal = float(technical_bot(df))
        feat_df = self.engineer_features(df).dropna()

        if feat_df.empty:
            self._log(symbol, price, "HOLD", 0.0, 0.0, 0.5, position_qty, equity, drawdown, False, "unknown", 0.0, "empty_features")
            return

        ml_prob = float(self.ml_models[symbol].predict(feat_df.iloc[-1]))
        decision = self.judge.evaluate(tech_signal, ml_prob, df)
        action, confidence, threshold, regime = decision["action"], float(decision["confidence"]), float(decision["threshold"]), decision["regime"]
        volatility = float(decision.get("volatility", 0.015))

        print(f"\n--- {symbol} | ${price:,.4f} | Conf: {confidence:.3f} | Regime: {regime} ---")

        risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)
        if not risk_ok:
            self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, regime, threshold, "risk_guard")
            return

        # BUY LOGIC
        if action == "BUY" and confidence >= threshold and position_qty <= 0:
            qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)
            if qty > 0:
                try:
                    self.broker.submit_order(
                        symbol=symbol, qty=qty, side="buy", type="market", time_in_force="gtc",
                        order_class="bracket",
                        take_profit={"limit_price": round(price * (1 + config.TAKE_PROFIT_PCT), 2)},
                        stop_loss={"stop_price": round(price * (1 - config.STOP_LOSS_PCT), 2)},
                    )
                    self.trailing_stop.on_entry(symbol, price)
                    self.entry_price[symbol] = price
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, regime, threshold, "bracket_entry")
                except Exception as e:
                    self._log(symbol, price, "BUY_FAILED", confidence, tech_signal, ml_prob, qty, equity, drawdown, False, regime, threshold, str(e))
            else:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, 0.0, equity, drawdown, False, regime, threshold, "zero_qty")

        # PROBE ENTRY
        elif (config.ENABLE_PROBE_ENTRY and action == "HOLD" and confidence >= config.PROBE_CONFIDENCE and ml_prob > 0.58 and tech_signal > 0.15 and position_qty <= 0):
            qty = self.compute_buy_qty(price, buying_power, equity, confidence, volatility, drawdown)
            probe_qty = round(qty * config.PROBE_SIZE_MULTIPLIER, 6)
            if probe_qty > 0:
                try:
                    self.broker.submit_order(symbol=symbol, qty=probe_qty, side="buy", type="market", time_in_force="gtc")
                    self.trailing_stop.on_entry(symbol, price)
                    self.entry_price[symbol] = price
                    self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, probe_qty, equity, drawdown, True, regime, threshold, "probe_entry")
                except Exception as e:
                    self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, 0.0, equity, drawdown, False, regime, threshold, f"probe_failed")

        # EXIT LOGIC
        elif position_qty > 0:
            entry_ref = live_entry_price if live_entry_price > 0 else self.entry_price.get(symbol, price)
            pnl_pct = (price - entry_ref) / entry_ref if entry_ref else 0.0
            trailing_triggered, trailing_stop_price = False, 0.0
            if config.ENABLE_TRAILING_STOP:
                trailing_triggered, trailing_stop_price = self.trailing_stop.should_exit(symbol, price, df)

            exit_on_signal = action == "SELL"
            exit_on_sl = pnl_pct <= -config.STOP_LOSS_PCT
            exit_on_trailing = trailing_triggered and pnl_pct > 0

            if exit_on_signal or exit_on_sl or exit_on_trailing:
                note = "trailing_stop" if exit_on_trailing else "shadow_exit" if exit_on_signal else "stop_loss"
                try:
                    self.broker.submit_order(symbol, position_qty, "sell", "market", "gtc")
                    self.trailing_stop.on_exit(symbol)
                    self.entry_price.pop(symbol, None)
                    self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, True, regime, threshold, note)
                except Exception: pass
            else:
                self.trailing_stop.update_peak(symbol, price)
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, regime, threshold, "shadow_holding")
        else:
            self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, 0.0, equity, drawdown, False, regime, threshold, "no_signal")

    def pre_flight_check(self):
        print("\n--- [STARTUP] Position Audit ---")
        for symbol in self.symbols:
            position_info = self.broker.get_position_info(symbol)
            qty = float(position_info.get("qty", 0.0))
            if qty > 0:
                avg_entry = float(position_info.get("avg_entry_price", 0.0))
                self.entry_price[symbol] = avg_entry
                self.trailing_stop.on_entry(symbol, avg_entry)
            df = self.fetch_data(symbol)
            self.evaluate_and_trade(symbol, df)

    def run(self):
        print("Trading Bot Active...")
        self.pre_flight_check()
        while True:
            now = datetime.now(UTC)
            if now >= self.next_decision_at:
                for symbol in self.symbols:
                    try:
                        df = self.fetch_data(symbol)
                        self.evaluate_and_trade(symbol, df)
                    except Exception as exc:
                        print(f"Error {symbol}: {exc}")
                self.next_decision_at = now + timedelta(hours=config.DECISION_INTERVAL_HOURS)
            else:
                time.sleep(60)

if __name__ == "__main__":
    TradingBot().run()