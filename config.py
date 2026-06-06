import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# API KEYS
# ============================================
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

# Alias for compatibility
ALPACA_API_KEY = APCA_API_KEY_ID
ALPACA_SECRET_KEY = APCA_API_SECRET_KEY

# ============================================
# TRADING CONFIGURATION
# ============================================
TRADE_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
LOOKBACK_HOURS = 168  # 7 days
ML_TRAIN_MIN_ROWS = 200

# ============================================
# RISK MANAGEMENT
# ============================================
MAX_POSITION_SIZE_PCT = 0.15  # Max 15% per trade
MAX_TOTAL_EXPOSURE_PCT = 0.40  # Max 40% total deployed
MAX_DAILY_LOSS_PCT = 0.05  # Stop at 5% daily loss
MAX_DRAWDOWN = 0.15  # Max 15% drawdown

# ============================================
# TRADE FILTERS
# ============================================
MIN_CONFIDENCE_THRESHOLD = 0.55  # Lowered for more trades
MIN_PROFIT_TARGET_PCT = 0.015  # 1.5% minimum profit
MAX_SPREAD_PCT = 0.002

# ============================================
# EXIT STRATEGY
# ============================================
TRAILING_STOP_ACTIVATION_PCT = 0.02  # Start trailing after 2% profit
TRAILING_STOP_DISTANCE_PCT = 0.01  # Trail by 1%
MAX_HOLD_HOURS = 48  # Force exit after 48 hours

# ============================================
# THRESHOLDS
# ============================================
BASE_THRESHOLD = 0.55
MAX_THRESHOLD = 0.80

# ============================================
# POSITION SIZING
# ============================================
POSITION_FRACTION = 0.15
MIN_EQUITY_FRACTION = 0.05
MAX_EQUITY_FRACTION = 0.25
MAX_NOTIONAL_PER_TRADE = 5000
MIN_NOTIONAL_PER_TRADE = 10

# ============================================
# LOGGING
# ============================================
TRADE_LOG_CSV = "trade_log.csv"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEETS_NAME = "TradingBot"
