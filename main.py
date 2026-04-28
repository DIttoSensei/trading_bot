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
            print(f"{symbol}: No market data fetched. HOLD.")
            return

        price = float(df.iloc[-1]["close"])
        equity, buying_power, daily_loss = self.get_account_state()
        drawdown = self.risk.update(equity)
        position_info = self.broker.get_position_info(symbol)
        position_qty = float(position_info.get("qty", 0.0))
        live_entry_price = float(position_info.get("avg_entry_price", 0.0))

        if len(df) < config.ML_TRAIN_MIN_ROWS:
            print("Not enough data for safe decision. HOLD.")
            self._log(symbol, price, "HOLD", 0.5, 0.0, 0.5, position_qty, equity, drawdown, False, "unknown", config.MIN_BUY_CONFIDENCE, "not_enough_data")
            return

        if self.should_retrain(symbol):
            print(f"{symbol}: Retraining ML model...")
            self.train_model(symbol, df)

        tech_signal = float(technical_bot(df))

        feat = df.copy()
        feat["return"] = feat["close"].pct_change()
        feat["momentum_3h"] = feat["close"] - feat["close"].shift(3)
        feat["momentum_24h"] = feat["close"] - feat["close"].shift(24)
        feat["volatility_24h"] = feat["return"].rolling(24).std()
        feat["ma_50_dist"] = feat["close"] / feat["close"].rolling(50).mean()
        feat = feat.dropna()
        if feat.empty:
            print("Feature frame empty after engineering. HOLD.")
            self._log(symbol, price, "HOLD", 0.5, tech_signal, 0.5, position_qty, equity, drawdown, False, "unknown", config.MIN_BUY_CONFIDENCE, "feature_frame_empty")
            return

        ml_prob = float(self.ml_models[symbol].predict(feat.iloc[-1]))
        decision = self.judge.evaluate(tech_signal, ml_prob, df)
        action = decision["action"]
        confidence = float(decision["confidence"])
        atr_value = float(
            (df["high"] - df["low"])
            .rolling(config.ATR_WINDOW)
            .mean()
            .iloc[-1]
        )
        atr_pct = atr_value / price if price > 0 else 0.0

        if config.ENABLE_BACKTEST_GATE:
            gate = walk_forward_gate(
                df,
                min_rows=config.ML_TRAIN_MIN_ROWS,
                test_window=config.BACKTEST_TEST_WINDOW,
                min_signals=config.BACKTEST_MIN_SIGNALS,
                min_winrate=config.BACKTEST_MIN_WINRATE,
            )
            if not gate["pass"]:
                print(f"Backtest gate failed ({gate['reason']}). HOLD.")
                self._log(
                    symbol,
                    price,
                    "HOLD",
                    confidence,
                    tech_signal,
                    ml_prob,
                    position_qty,
                    equity,
                    drawdown,
                    False,
                    str(decision.get("regime", "unknown")),
                    float(decision.get("threshold", config.MIN_BUY_CONFIDENCE)),
                    f"gate_{gate['reason']}",
                )
                return

        decision_threshold = float(decision.get("threshold", config.MIN_BUY_CONFIDENCE))
        market_vol = float(decision.get("volatility", 0.015))
        risk_ok = self.risk.allow_trading(equity) and (daily_loss < config.MAX_DAILY_LOSS_PCT)

        print("\n==============================")
        print(f"Time: {datetime.now(UTC).isoformat()}")
        print(f"{symbol} | Price: {price:.2f} | Position qty: {position_qty:.6f}")
        print(f"Tech: {tech_signal:.3f} | ML: {ml_prob:.3f} | Confidence: {confidence:.3f}")
        print(
            f"Regime: {decision.get('regime', 'unknown')} | Threshold: {float(decision.get('threshold', config.MIN_BUY_CONFIDENCE)):.3f}"
        )
        print(f"ATR%: {atr_pct:.3%}")
        print(f"Action: {action} | Drawdown: {drawdown:.3%} | Daily loss: {daily_loss:.3%}")

        if not risk_ok:
            self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, str(decision.get("regime", "unknown")), decision_threshold, "risk_pause")
            print("Risk guard active. HOLD.")
            return

        full_entry_ok = action == "BUY" and confidence >= max(config.MIN_BUY_CONFIDENCE, decision_threshold) and position_qty <= 0
        probe_entry_ok = (
            config.ENABLE_PROBE_ENTRY
            and position_qty <= 0
            and confidence >= config.PROBE_CONFIDENCE
            and tech_signal >= config.PROBE_TECH_MIN
            and ml_prob >= config.PROBE_ML_MIN
        )

        if full_entry_ok or probe_entry_ok:
            qty = self.compute_buy_qty(price, buying_power, equity, confidence, market_vol, drawdown)
            if probe_entry_ok and not full_entry_ok:
                qty = round(max(qty * config.PROBE_SIZE_MULTIPLIER, 0.0), 6)
            if qty > 0:
                self.broker.submit_order(symbol, "BUY", qty)
                self.entry_price[symbol] = price
                print(f"BUY executed: {qty} (notional ~ {qty * price:.2f})")
                note = "entry" if full_entry_ok else "probe_entry"
                self._log(symbol, price, "BUY", confidence, tech_signal, ml_prob, qty, equity, drawdown, True, str(decision.get("regime", "unknown")), decision_threshold, note)
            else:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, str(decision.get("regime", "unknown")), decision_threshold, "small_notional")
        elif position_qty > 0:
            entry_ref = live_entry_price if live_entry_price > 0 else self.entry_price.get(symbol)
            pnl = 0.0 if not entry_ref else (price - entry_ref) / entry_ref
            exit_on_signal = action == "SELL" and confidence > 0.5
            stop_loss_pct = config.STOP_LOSS_PCT
            take_profit_pct = config.TAKE_PROFIT_PCT
            if config.ENABLE_ATR_EXITS and atr_pct > 0:
                stop_loss_pct = max(config.STOP_LOSS_PCT, config.ATR_STOP_MULTIPLIER * atr_pct)
                take_profit_pct = max(config.TAKE_PROFIT_PCT, config.ATR_TP_MULTIPLIER * atr_pct)
            exit_on_tp = pnl >= take_profit_pct
            exit_on_sl = pnl <= -stop_loss_pct
            if exit_on_signal or exit_on_tp or exit_on_sl:
                self.broker.submit_order(symbol, "SELL", position_qty)
                note = "signal_exit" if exit_on_signal else ("take_profit" if exit_on_tp else "stop_loss")
                print(f"SELL executed: {position_qty} ({note})")
                self.entry_price[symbol] = None
                self._log(symbol, price, "SELL", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, True, str(decision.get("regime", "unknown")), decision_threshold, note)
            else:
                self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, str(decision.get("regime", "unknown")), decision_threshold, "in_position")
        else:
            self._log(symbol, price, "HOLD", confidence, tech_signal, ml_prob, position_qty, equity, drawdown, False, str(decision.get("regime", "unknown")), decision_threshold, "no_setup")

    def run(self):
        print("=== Self-Reliant Paper Trading Bot Started ===")
        print(f"Symbols: {', '.join(self.symbols)} | Decision interval: {config.DECISION_INTERVAL_HOURS}h")

        if config.BOT_RUN_ONCE:
            print("BOT_RUN_ONCE enabled. Running a single cycle.")
            for symbol in self.symbols:
                df = self.fetch_data(symbol)
                self.evaluate_and_trade(symbol, df)
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
                    print(f"Data refreshed at {now.isoformat()}. Next decision: {self.next_decision_at.isoformat()}")
            except Exception as exc:
                print(f"Loop error: {exc}")
            time.sleep(max(60, config.DATA_REFRESH_MINUTES * 60))


if __name__ == "__main__":
    TradingBot().run()