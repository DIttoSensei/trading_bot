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
from layer1_technical import technical_bot # Connected Layer 1
from layer3_judge import LLMJudge
from risk import RiskManager
from sheet_logger import GoogleSheetLogger
from foresight_engine import ForesightEngine # Connected Foresight Math


class TradingBot:
    def __init__(self):
        self.symbols = config.TRADE_SYMBOLS

        self.broker = Broker(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            paper=True,
        )
        self.data_client = CryptoHistoricalDataClient()
        self.ml = {s: MLSpecialist(s) for s in self.symbols}
        self.judge = LLMJudge()
        self.risk = RiskManager(config.MAX_DRAWDOWN)
        
        # Link Foresight Engine using config minimum profit target
        self.foresight = ForesightEngine(min_profit_pct=config.MIN_PROFIT_TARGET_PCT)
        
        self.logger = GoogleSheetLogger(
            config.GOOGLE_CREDENTIALS_FILE,
            config.GOOGLE_SHEETS_NAME,
        )

        self.state_path = "bot_state.json"
        self.positions = {}
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                    self.positions = data.get("positions", {})
                print(f"[State] Loaded positions: {list(self.positions.keys()) or 'none'}")
            except Exception as e:
                print(f"[State] Load failed: {e}")

    def _save_state(self):
        try:
            with open(self.state_path, "w") as f:
                json.dump({"positions": self.positions}, f, indent=2)
        except Exception as e:
            print(f"[State] Save failed: {e}")

    def _sync_positions_from_broker(self):
        try:
            live = self.broker.get_all_positions()
            synced = {}
            for p in live:
                raw = p.symbol.replace("/", "").strip().upper()
                for s in self.symbols:
                    clean_config = s.replace("/", "").strip().upper()
                    if clean_config == raw:
                        synced[s] = float(p.avg_entry_price)
                        break
            self.positions = synced
            print(f"[State] Broker sync: {list(self.positions.keys()) or 'no open positions'}")
        except Exception as e:
            print(f"[State] Broker sync failed, using cached state: {e}")

    def fetch_data(self, symbol: str):
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
            if len(df) < 100:
                print(f"[{symbol}] Only {len(df)} rows, need 100+, skip.")
                return None
            return df
        except Exception as e:
            print(f"[Data] fetch failed {symbol}: {e}")
            return None

    def _log(self, symbol, price, action, conf, tech, ml_prob, qty, equity, dd, regime, note="ok"):
        row = [
            datetime.now(UTC).isoformat(), symbol, round(float(price), 4), action,
            round(float(conf), 4), round(float(tech), 4), round(float(ml_prob), 4),
            round(float(qty), 8), round(float(equity), 2), round(float(dd), 6),
            regime, config.BASE_THRESHOLD, note,
        ]
        self.logger.log_row(row)

    def run_cycle(self):
        print(f"\n{'='*50}")
        print(f"RUN {datetime.now(UTC).isoformat()}")

        acc = self.broker.get_account()
        equity = float(acc.equity)
        dd     = self.risk.update(equity)
        print(f"Equity: ${equity:,.2f}  |  Drawdown: {dd:.2%}")

        self._sync_positions_from_broker()

        for symbol in self.symbols:
            try:
                self._run_symbol(symbol, equity, dd)
            except Exception as e:
                print(f"[{symbol}] UNHANDLED: {e}")
                traceback.print_exc()

        self._save_state()
        print("Cycle complete.")

    def _run_symbol(self, symbol: str, equity: float, dd: float):
        df = self.fetch_data(symbol)
        if df is None:
            return

        price = float(df["close"].iloc[-1])

        # Core Engine Linking
        self.ml[symbol].train(df)
        features    = self.ml[symbol].get_latest_features(df)
        ml_prob     = self.ml[symbol].predict(features)
        
        # Delegate technical analysis to Layer 1 module
        tech_signal = technical_bot(df)

        decision   = self.judge.evaluate(tech_signal, ml_prob, df)
        action     = decision["action"]
        conf       = decision["confidence"]
        regime     = decision["regime"]

        has_position = symbol in self.positions
        final_action = "HOLD"
        qty          = 0.0

        # Dynamic Foresight Exit Rules
        if has_position:
            entry_price = self.positions[symbol]
            targets = self.foresight.get_dynamic_targets(df, entry_price)
            
            if price >= targets["take_profit"]:
                action = "SELL"
                conf = 1.0
                print(f"[{symbol}] FORESIGHT PROFIT HIT: Target ${targets['take_profit']:.2f}. Forcing Sell.")
            
            elif price <= targets["stop_loss"]:
                action = "SELL"
                conf = 1.0
                print(f"[{symbol}] FORESIGHT STOP LOSS HIT: Limit ${targets['stop_loss']:.2f}. Forcing Sell.")

        print(
            f"[{symbol}] ${price:.2f} | ml={ml_prob:.3f} tech={tech_signal:.3f} "
            f"conf={conf:.3f} | signal={action} | holding={'YES' if has_position else 'NO'}"
        )

        if not self.risk.allow_trading(equity):
            self._log(symbol, price, "BLOCKED", conf, tech_signal, ml_prob, 0, equity, dd, regime, "max_drawdown")
            return

        # Execution Blocks with Slash Scrubbing for Alpaca
        if action == "BUY" and not has_position:
            if conf >= config.BASE_THRESHOLD: 
                qty = self.risk.position_size(equity, price)
                if qty * price < config.MIN_NOTIONAL_PER_TRADE:
                    return
                
                broker_symbol = symbol.replace("/", "")
                order = self.broker.submit_order(broker_symbol, "buy", qty)
                if order:
                    self.positions[symbol] = price
                    final_action = "BUY"

        elif action == "SELL" and has_position:
            if conf >= 0.45 or conf == 1.0: # Matches new Judge exhaustion limit
                broker_symbol = symbol.replace("/", "")
                order = self.broker.close_position(broker_symbol)
                if order:
                    entry = self.positions.pop(symbol)
                    pnl_pct = (price - entry) / entry
                    print(f"[{symbol}] Closed. PnL: {pnl_pct:+.2%}")
                    final_action = "SELL"

        elif action == "BUY" and has_position:
            final_action = "HOLD_LONG"

        elif action == "SELL" and not has_position:
            final_action = "HOLD_FLAT"

        self._log(symbol, price, final_action, conf, tech_signal, ml_prob, qty, equity, dd, regime)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run_cycle()