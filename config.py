import os

# --- API CREDENTIALS ---
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

# --- ASSETS & FILE PATHS ---
TRADE_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

TRADE_LOG_CSV = "trading_journal.csv"
TRAINING_DATA_FILE = "training_data.json"

GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")

# --- DATA LOOKBACK ---
LOOKBACK_HOURS = 1440  # 60 days

# --- RISK & EXPOSURE LIMITS ---
MAX_DRAWDOWN = 0.05
MAX_DAILY_LOSS_PCT = 0.02
MAX_TOTAL_EXPOSURE_PCT = 0.75

POSITION_FRACTION = 0.15
MIN_EQUITY_FRACTION = 0.05
MAX_EQUITY_FRACTION = 0.25

MIN_NOTIONAL_PER_TRADE = 10.0
MAX_NOTIONAL_PER_TRADE = 5000.0

# --- SIGNAL & CONVICTION THRESHOLDS ---
BASE_THRESHOLD = 0.55       # Minimum baseline confidence for valid signals

SWING_BUY_THRESHOLD = 0.68  # High conviction threshold (lowered from 0.72 to hit valid setups)
SCALP_BUY_THRESHOLD = 0.60  # Balanced threshold (allows execution while blocking 0.50–0.55 noise)

# --- TAKE PROFIT / STOP LOSS TARGETS ---
MIN_PROFIT_TARGET_PCT = 0.04 # Baseline target for swing strategies

SCALP_TP_PCT = 0.020        # Scalp Take Profit (2.0%)
SCALP_SL_PCT = 0.008        # Scalp Stop Loss (0.8% — maintains a 2.5:1 Risk/Reward ratio)import os

ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")

TRADE_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]

TRADE_LOG_CSV = "trading_journal.csv"
TRAINING_DATA_FILE = "training_data.json"

GOOGLE_SHEETS_NAME = os.getenv("GOOGLE_SHEETS_NAME", "Trading Log")
GOOGLE_CREDENTIALS_FILE = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")

LOOKBACK_HOURS = 1440  # 60 days

MAX_DRAWDOWN = 0.05
MAX_DAILY_LOSS_PCT = 0.02
MAX_TOTAL_EXPOSURE_PCT = 0.75

POSITION_FRACTION = 0.15
MIN_EQUITY_FRACTION = 0.05
MAX_EQUITY_FRACTION = 0.25

BASE_THRESHOLD = 0.51

MIN_NOTIONAL_PER_TRADE = 10.0
MAX_NOTIONAL_PER_TRADE = 5000.0

MIN_PROFIT_TARGET_PCT = 0.04

SWING_BUY_THRESHOLD = 0.70  # High conviction threshold
SCALP_BUY_THRESHOLD = 0.58  # Moderate conviction threshold
SCALP_TP_PCT = 0.015        # Scalp Take Profit (1.5%)
SCALP_SL_PCT = 0.0075       # Scalp Stop Loss (0.75%)

