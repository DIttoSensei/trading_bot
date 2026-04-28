import os

# Alpaca keys: support APCA_* and ALPACA_* names
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", os.getenv("ALPACA_API_KEY", ""))
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
ALPACA_PAPER = True

# Core trading setup
SYMBOL = os.getenv("TRADE_SYMBOL", "BTC/USD")
POSITION_FRACTION = float(os.getenv("POSITION_FRACTION", "0.25"))  # 25% of buying power
MIN_EQUITY_FRACTION = float(os.getenv("MIN_EQUITY_FRACTION", "0.05"))  # at least 5% of equity
MAX_EQUITY_FRACTION = float(os.getenv("MAX_EQUITY_FRACTION", "0.30"))  # cap at 30% of equity
MAX_NOTIONAL_PER_TRADE = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "30000"))
MIN_NOTIONAL_PER_TRADE = float(os.getenv("MIN_NOTIONAL_PER_TRADE", "500"))

# Timing
DATA_REFRESH_MINUTES = int(os.getenv("DATA_REFRESH_MINUTES", "60"))  # scrape hourly
DECISION_INTERVAL_HOURS = int(os.getenv("DECISION_INTERVAL_HOURS", "4"))  # trade every 4h
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "600"))
MODEL_RETRAIN_HOURS = int(os.getenv("MODEL_RETRAIN_HOURS", "24"))
BOT_RUN_ONCE = os.getenv("BOT_RUN_ONCE", "false").lower() == "true"

# Risk controls
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
MIN_BUY_CONFIDENCE = float(os.getenv("MIN_BUY_CONFIDENCE", "0.56"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))

# Backtest gate
ML_TRAIN_MIN_ROWS = int(os.getenv("ML_TRAIN_MIN_ROWS", "240"))
BACKTEST_TEST_WINDOW = int(os.getenv("BACKTEST_TEST_WINDOW", "48"))
BACKTEST_MIN_SIGNALS = int(os.getenv("BACKTEST_MIN_SIGNALS", "8"))
BACKTEST_MIN_WINRATE = float(os.getenv("BACKTEST_MIN_WINRATE", "0.50"))

# Logging
TRADE_LOG_CSV = os.getenv("TRADE_LOG_CSV", "trade_journal.csv")
GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
