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
from risk import RiskManager
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

        self.state_path = os.path.join(os.path.dirname(__file__), "bot_state.json")
        self.json_dataset_path = os.path.join(os.path.dirname(__file__), "training_data.json")
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
                    self.cooldowns = state.get("cooldowns", {})
                    self.positions = state.get("positions", {})
                    print(f"✅ State loaded successfully. Tracked portfolio pairs: {list(self.positions.keys())}")
            except Exception as e:
                print(f"⚠️ State load corrupted: {e}")

    def save_bot_state(self):
        try:
            state = {"cooldowns": self.cooldowns, "positions": self.positions}
            with open(self.state_path, "w") as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            print(f"❌ Failed to persist context to disk: {e}")

    def _ensure_journal(self):
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp_utc", "symbol", "price", "action",
                    "confidence", "tech_signal", "ml_prob",
                    "position_qty", "equity", "drawdown", "traded",
                    "regime", "threshold", "note"
                ])

    def _save_json_training_snapshot(self, symbol, d_feat, tech_signal, ml_prob):
        """Accumulates structured states over time into a training JSON matrix."""
        try:
            dataset = []
            if os.path.exists(self.json_dataset_path):
                try:
                    with open(self.json_dataset_path, "r") as f:
                        dataset = json.load(f)
                except Exception:
                    dataset = []

            last_row = d_feat.iloc[-1]
            snapshot = {
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": symbol,
                "features": {
                    "return_4h": float(last_row.get("return_4h", 0.0)),
                    "volatility_24h": float(last_row.get("volatility_24h", 0.0)),
                    "ma_20_dist": float(last_row.get("ma_20_dist", 0.0)),
                    "close": float(last_row.get("close", 0.0))
                },
                "metrics": {
                    "technical_bot_score": float(tech_signal),
                    "ml_probability_score": float(ml_prob)
                }
            }
            dataset.append(snapshot)
            with open(self.json_dataset_path, "w") as f:
                json.dump(dataset, f, indent=4)
            print(f"💾 [JSON MATRIX] Persistent training data logged for {symbol}.")
        except Exception as e:
            print(f"❌ Failed writing JSON snapshot: {e}")

    def _log(self, symbol, price, action, confidence, tech_signal, ml_prob, position_qty, equity, drawdown, traded, regime, threshold, note):
        try:
            row = [
                datetime.now(UTC).isoformat(),
                str(symbol), round(float(price), 4), str(action),
                round(float(confidence), 4), round(float(tech_signal), 4), round(float(ml_prob), 4),
                round(float(position_qty), 4), round(float(equity), 2), round(float(drawdown), 4),
                int(bool(traded)), str(regime), round(float(threshold), 4), str(note)
            ]
            with open(self.journal_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            
            print(f"📡 [SHEET API] Appending initial verification row for {symbol}...")
            return self.sheet_logger.log_row(row)
        except Exception as e:
            print(f"❌ Primary logging allocation failed: {e}")
            return None

    def _update_log(self, row_index, symbol, price, action, confidence, tech_signal, ml_prob, position_qty, equity, drawdown, traded, regime, threshold, note):
        if row_index is None:
            return
        try:
            row = [
                datetime.now(UTC).isoformat(),
                str(symbol), round(float(price), 4), str(action),
                round(float(confidence), 4), round(float(tech_signal), 4), round(float(ml_prob), 4),
                round(float(position_qty), 4), round(float(equity), 2), round(float(drawdown), 4),
                int(bool(traded)), str(regime), round(float(threshold), 4), str(note)
            ]
            self.sheet_logger.update_row(row_index, row)
        except Exception as e:
            print(f"❌ Spreadsheet matrix update tracking failure: {e}")

    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
            end = datetime.now(UTC)
            # scaled baseline configuration parameter to extract true data depth
            lookback_hours = max(getattr(config, "LOOKBACK_HOURS", 168), 720) 
            start = end - timedelta(hours=lookback_hours)
            req = CryptoBarsRequest(symbol_or_symbols=[symbol], timeframe=TimeFrame.Hour, start=start, end=end)
            bars = self.data_client.get_crypto_bars(req).df.reset_index()
            return bars[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            print(f"❌ Ingestion drop for {symbol}: {e}")
            return None

    def compute_buy_qty(self, price: float, buying_power: float, equity: float, confidence: float, volatility: float) -> float:
        safe_vol = max(volatility, 0.0001)
        vol_scale = np.clip(0.015 / safe_vol, 0.6, 1.5)
        base_thresh = getattr(config, "BASE_THRESHOLD", 0.58)
        conf_scale = np.clip((confidence - base_thresh) / 0.12, 0.4, 1.5)
        
        pos_fraction = getattr(config, "POSITION_FRACTION", 0.15)
        min_fraction = getattr(config, "MIN_EQUITY_FRACTION", 0.05)
        max_fraction = getattr(config, "MAX_EQUITY_FRACTION", 0.25)

        target_fraction = pos_fraction * vol_scale * conf_scale
        target_fraction = float(np.clip(target_fraction, min_fraction, max_fraction))

        target_allocation = min(equity * target_fraction, getattr(config, "MAX_NOTIONAL_PER_TRADE", 5000.0))
        target_allocation = min(target_allocation, buying_power * 0.95)

        if target_allocation < getattr(config, "MIN_NOTIONAL_PER_TRADE", 10.0):
            return 0.0
        return round(max(target_allocation / price, 0.0), 5)

    def run_cycle(self):
        print(f"⚡ Execution Initialization Window: {datetime.now(UTC).isoformat()}")
        try:
            acc = self.broker.get_account()
            equity, buying_power = float(acc.equity), float(acc.buying_power)
        except Exception as e:
            print(f"❌ Could not reach broker REST layer: {e}")
            return

        drawdown = self.risk.update(equity)
        if self.day_start_equity is None:
            self.day_start_equity = equity
        daily_loss = max(0.0, (self.day_start_equity - equity) / self.day_start_equity)
        now_ts = datetime.now(UTC).timestamp()

        for symbol in self.symbols:
            sheet_row_index = None
            current_price = 0.0
            current_qty = 0.0
            
            try:
                df = self.fetch_data(symbol)
                if df is None or len(df) < 30:
                    self._log(symbol, 0.0, "DROP", 0.0, 0.0, 0.0, 0.0, equity, drawdown, False, "error", 0.0, "insufficient_data_history")
                    continue

                current_price = float(df.iloc[-1]["close"])
                broker_pos = self.broker.get_position_info(symbol)
                current_qty = float(broker_pos.get("qty", 0.0))

                if current_qty == 0 and symbol in self.positions:
                    del self.positions[symbol]

                # Establish an immediate trace right on the open grid matrix
                sheet_row_index = self._log(
                    symbol, current_price, "EVAL", 0.5, 0.5, 0.5,
                    current_qty, equity, drawdown, False, "processing", 0.0, "evaluating_features"
                )

                if now_ts < self.cooldowns.get(symbol, 0.0):
                    self._update_log(sheet_row_index, symbol, current_price, "OUT", 0.5, 0.5, 0.5, current_qty, equity, drawdown, False, "cooling", 0.0, "cooldown_active")
                    continue

                d_feat = df.copy()
                d_feat["return_4h"] = d_feat["close"].pct_change(4)
                d_feat["volatility_24h"] = d_feat["return_4h"].rolling(24).std()
                d_feat["ma_20_dist"] = (d_feat["close"] / d_feat["close"].rolling(20).mean()) - 1.0
                d_feat = d_feat.dropna()

                if d_feat.empty:
                    self._update_log(sheet_row_index, symbol, current_price, "OUT", 0.5, 0.5, 0.5, current_qty, equity, drawdown, False, "error", 0.0, "math_wiped_dataframe")
                    continue

                if symbol not in self.ml_models:
                    self.ml_models[symbol] = MLSpecialist()

                model = self.ml_models[symbol]
                model.train_price_model(df)

                ml_prob = float(model.predict(d_feat.iloc[-1]))
                tech_signal = float(technical_bot(df))

                # Persistent local snapshot layer collection pass
                self._save_json_training_snapshot(symbol, d_feat, tech_signal, ml_prob)

                try:
                    decision = self.judge.evaluate(tech_signal, ml_prob, df)
                except TypeError:
                    decision = self.judge.evaluate(tech_signal, ml_prob)

                action, confidence, threshold, regime = decision["action"], decision["confidence"], decision["threshold"], decision["regime"]
                volatility = decision.get("volatility", 0.015)

                if not self.risk.allow_trading(equity) or (daily_loss >= config.MAX_DAILY_LOSS_PCT):
                    self._update_log(sheet_row_index, symbol, current_price, "HOLD", confidence, tech_signal, ml_prob, current_qty, equity, drawdown, False, regime, threshold, "risk_circuit_breaker_active")
                    continue

                if current_qty > 0:
                    self._update_log(sheet_row_index, symbol, current_price, "HOLD", confidence, tech_signal, ml_prob, current_qty, equity, drawdown, False, regime, threshold, "monitored_by_exchange_bracket")
                    continue

                elif action == "BUY" and current_qty == 0:
                    current_portfolio = self.broker.get_all_positions()
                    allocated_capital = sum(float(p.qty) * float(p.current_price) for p in current_portfolio)

                    if allocated_capital >= (equity * config.MAX_TOTAL_EXPOSURE_PCT):
                        self._update_log(sheet_row_index, symbol, current_price, "OUT", confidence, tech_signal, ml_prob, current_qty, equity, drawdown, False, regime, threshold, "skip_max_exposure_limit")
                        continue

                    buy_qty = self.compute_buy_qty(current_price, buying_power, equity, confidence, volatility)
                    if regime == "bear_trend":
                        buy_qty = round(buy_qty * 0.50, 5)

                    if buy_qty > 0:
                        tp_price = current_price * (1.0 + getattr(config, "MIN_PROFIT_TARGET_PCT", 0.04))
                        sl_price = current_price * (1.0 - getattr(config, "MAX_DAILY_LOSS_PCT", 0.02))

                        order = self.broker.submit_order(
                            symbol=symbol, side="buy", qty=buy_qty, take_profit_price=tp_price, stop_loss_price=sl_price
                        )

                        if order:
                            self.positions[symbol] = {
                                "entry_price": current_price,
                                "entry_time": datetime.now(UTC).isoformat(),
                                "qty": buy_qty
                            }
                            self._update_log(sheet_row_index, symbol, current_price, "BUY", confidence, tech_signal, ml_prob, buy_qty, equity, drawdown, True, regime, threshold, f"bracket_entry")
                    else:
                        self._update_log(sheet_row_index, symbol, current_price, "OUT", confidence, tech_signal, ml_prob, current_qty, equity, drawdown, False, regime, threshold, "under_min_notional_floor")
                else:
                    self._update_log(sheet_row_index, symbol, current_price, "OUT", confidence, tech_signal, ml_prob, current_qty, equity, drawdown, False, regime, threshold, "waiting_for_entry")

            except Exception as e:
                print(f"❌ Loop Failure: {e}")
                traceback.print_exc()
                if sheet_row_index is not None:
                    self._update_log(sheet_row_index, symbol, current_price, "CRASH", 0.0, 0.0, 0.0, current_qty, equity, drawdown, False, "fault", 0.0, f"Error: {str(e)[:25]}")
                continue

        self.save_bot_state()

if __name__ == "__main__":
    bot = TradingBot()
    bot.run_cycle()
