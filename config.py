import os

# Alpaca keys
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
ALPACA_PAPER = True

# Core trading setup
SYMBOL = os.getenv("TRADE_SYMBOL", "BTC/USD")
TRADE_SYMBOLS = ["BTC/USD", "SOL/USD", "DOGE/USD", "PEPE/USD"]
POSITION_FRACTION = float(os.getenv("POSITION_FRACTION", "0.60"))
MIN_EQUITY_FRACTION = float(os.getenv("MIN_EQUITY_FRACTION", "0.15"))
MAX_EQUITY_FRACTION = float(os.getenv("MAX_EQUITY_FRACTION", "0.90"))  # never go 100% - keep buffer
MAX_NOTIONAL_PER_TRADE = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "30000"))
MIN_NOTIONAL_PER_TRADE = float(os.getenv("MIN_NOTIONAL_PER_TRADE", "50"))

# Timing
DATA_REFRESH_MINUTES = int(os.getenv("DATA_REFRESH_MINUTES", "15"))
DECISION_INTERVAL_HOURS = int(os.getenv("DECISION_INTERVAL_HOURS", "1"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "1440"))
MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
BOT_RUN_ONCE = os.getenv("BOT_RUN_ONCE", "false").lower() == "true"

# Risk controls
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.025"))       # slightly wider - avoids getting shaken out
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.10"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.04"))

# Entry thresholds - FIXED: was too tight, almost nothing passed
MIN_BUY_CONFIDENCE = float(os.getenv("MIN_BUY_CONFIDENCE", "0.54"))   # raised from 0.49 - needs real signal
BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.54"))            # raised from 0.51
MAX_THRESHOLD = float(os.getenv("MAX_THRESHOLD", "0.68"))              # raised from 0.65

# Probe entries (small test positions)
ENABLE_PROBE_ENTRY = os.getenv("ENABLE_PROBE_ENTRY", "true").lower() == "true"
PROBE_CONFIDENCE = float(os.getenv("PROBE_CONFIDENCE", "0.52"))
PROBE_SIZE_MULTIPLIER = float(os.getenv("PROBE_SIZE_MULTIPLIER", "0.25"))

# ATR-based trailing stop - NEW
ENABLE_TRAILING_STOP = os.getenv("ENABLE_TRAILING_STOP", "true").lower() == "true"
ATR_WINDOW = int(os.getenv("ATR_WINDOW", "14"))
ATR_STOP_MULTIPLIER = float(os.getenv("ATR_STOP_MULTIPLIER", "2.0"))   # trailing stop = 2x ATR below peak
ATR_TP_MULTIPLIER = float(os.getenv("ATR_TP_MULTIPLIER", "5.0"))

# Backtest gate
ENABLE_BACKTEST_GATE = os.getenv("ENABLE_BACKTEST_GATE", "false").lower() == "true"
ML_TRAIN_MIN_ROWS = int(os.getenv("ML_TRAIN_MIN_ROWS", "500"))         # lowered from 1300 - was blocking training too long
BACKTEST_TEST_WINDOW = int(os.getenv("BACKTEST_TEST_WINDOW", "96"))
BACKTEST_MIN_SIGNALS = int(os.getenv("BACKTEST_MIN_SIGNALS", "6"))
BACKTEST_MIN_WINRATE = float(os.getenv("BACKTEST_MIN_WINRATE", "0.45"))

# Logging
TRADE_LOG_CSV = os.getenv("TRADE_LOG_CSV", "trade_journal.csv")
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")