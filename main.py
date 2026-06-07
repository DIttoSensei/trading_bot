"""
main.py — integration layer.
Fixes vs previous broken version:
  - Broker.submit_order() called with (symbol, side, qty) only
  - SELL uses broker.close_position() — not submit_order
  - ML returns DataFrame features — no sklearn warning
  - Sheet logging is append-only via sheet_logger.log_row()
  - Per-symbol exceptions never crash the whole loop
  - Position sizing uses risk.position_size()
"""
import os
import json
import traceback
from datetime import datetime, UTC, timedelta

import numpy as np
import pandas as pd

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

import config
from broker import Broker
from ml_layer import MLSpecialist
from layer3_judge import LLMJudge
from risk import RiskManager
from sheet_logger import GoogleSheetLogger


class TradingBot:
    def __init__(self):
        self.symbols = config.TRADE_SYMBOLS

        self.broker = Broker(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=True,  # set False for live
        )
        self.data_client = CryptoHistoricalDataClient()

        self.ml = {s: MLSpecialist(s) for s in self.symbols}
        self.judge = LLMJudge()
        self.risk = RiskManager(config.MAX_DRAWDOWN)
        self.logger = GoogleSheetLogger(
            config.GOOGLE_CREDENTIALS_FILE,
            config.GOOGLE_SHEETS_NAME,
        )

        self.state_path = "bot_state.json"
        self.positions = {}   # symbol → entry_price
        self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
            except Exception:
                pass

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump({"positions": self.positions}, f, indent=2)
        except Exception as e:
            print(f"[State] Save failed: {e}")

    # ------------------------------------------------------------------ data
    def fetch_data(self, symbol: str) -> pd.DataFrame | None:
        try:
            end   = datetime.now(UTC)
            start = end - timedelta(hours=config.LOOKBACK_HOURS)
            req   = CryptoBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Hour,
                start=start,
                end=end,
            )
            df = self.data_client.get_crypto_bars(req).df.reset_index()
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df if len(df) >= 100 else None
        except Exception as e:
            print(f"[Data] fetch failed for {symbol}: {e}")
            return None

    # ------------------------------------------------------------------ log
    def _log(self, symbol, price, action, conf, tech, ml_prob, qty, equity, dd, regime, note="ok"):
        row = [
            datetime.now(UTC).isoformat(),
            symbol, price, action,
            round(conf, 4), round(tech, 4), round(ml_prob, 4),
            qty, round(equity, 2), round(dd, 4),
            regime, config.BASE_THRESHOLD, note,
        ]
        self.logger.log_row(row)

    # ------------------------------------------------------------------ main
    def run_cycle(self):
        print(f"\n{'='*50}")
        print(f"RUN {datetime.now(UTC).isoformat()}")

        acc    = self.broker.get_account()
        equity = float(acc.equity)
        dd     = self.risk.update(equity)

        print(f"Equity: ${equity:,.2f}  |  Drawdown: {dd:.2%}")

        for symbol in self.symbols:
            try:
                self._run_symbol(symbol, equity, dd)
            except Exception as e:
                print(f"[{symbol}] UNHANDLED ERROR: {e}")
                traceback.print_exc()

        self._save_state()

    def _run_symbol(self, symbol: str, equity: float, dd: float):
        # 1. Fetch data
        df = self.fetch_data(symbol)
        if df is None:
            print(f"[{symbol}] Insufficient data, skip.")
            return

        price = float(df["close"].iloc[-1])

        # 2. Train ML (retrain every run — stateless env)
        self.ml[symbol].train(df)

        # 3. Get ML probability
        features = self.ml[symbol].get_latest_features(df)
        ml_prob  = self.ml[symbol].predict(features)

        # 4. Technical signal = RSI-based simple signal
        tech_signal = self._tech_signal(df)

        # 5. Decision
        decision   = self.judge.evaluate(tech_signal, ml_prob, df)
        action     = decision["action"]
        conf       = decision["confidence"]
        regime     = decision["regime"]
        volatility = decision["volatility"]

        print(f"[{symbol}] price={price:.2f}  ml={ml_prob:.3f}  tech={tech_signal:.3f}  conf={conf:.3f}  → {action}")

        # 6. Risk gate
        if not self.risk.allow_trading(equity):
            self._log(symbol, price, "BLOCKED", conf, tech_signal, ml_prob, 0, equity, dd, regime, "max_drawdown")
            return

        qty = 0.0

        # 7. Execute
        if action == "BUY" and symbol not in self.positions:
            qty = self.risk.position_size(equity, price, volatility)
            if qty * price < config.MIN_NOTIONAL_PER_TRADE:
                self._log(symbol, price, "SKIP_BUY", conf, tech_signal, ml_prob, 0, equity, dd, regime, "below_min_notional")
                return
            order = self.broker.submit_order(symbol, "buy", qty)
            if order:
                self.positions[symbol] = price

        elif action == "SELL" and symbol in self.positions:
            order = self.broker.close_position(symbol)
            if order:
                entry = self.positions.pop(symbol)
                pnl_pct = (price - entry) / entry
                print(f"[{symbol}] Closed position. PnL: {pnl_pct:.2%}")

        # 8. Log
        self._log(symbol, price, action, conf, tech_signal, ml_prob, qty, equity, dd, regime)

    # ------------------------------------------------------------------ tech
    def _tech_signal(self, df: pd.DataFrame) -> float:
        """
        Simple RSI-based technical signal normalised to 0–1.
        RSI < 30 → oversold → bullish (signal near 1.0)
        RSI > 70 → overbought → bearish (signal near 0.0)
        """
        try:
            delta = df["close"].diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
            rs    = gain / loss.replace(0, np.nan)
            rsi   = 100 - (100 / (1 + rs))
            rsi_val = float(rsi.iloc[-1])
            if np.isnan(rsi_val):
                return 0.5
            return float(np.clip(1.0 - (rsi_val / 100.0), 0.0, 1.0))
        except Exception:
            return 0.5


if __name__ == "__main__":
    bot = TradingBot()
    bot.run_cycle()