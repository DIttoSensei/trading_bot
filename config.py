import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# API KEYS & CREDENTIALS
# ============================================
APCA_API_KEY_ID = os.getenv("APCA_API_KEY_ID")
APCA_API_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")

ALPACA_API_KEY = APCA_API_KEY_ID
ALPACA_SECRET_KEY = APCA_API_SECRET_KEY

# ============================================
# ASSET MATRIX
# ============================================
TRADE_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
LOOKBACK_HOURS = 168  # 7 days of historical context
ML_TRAIN_MIN_ROWS = 100

# ============================================
# CAPITAL EXPOSURE & RISK LIMITS
# ============================================
MAX_POSITION_SIZE_PCT = 0.20    # Commit up to 20% of equity per sniper trade
MAX_TOTAL_EXPOSURE_PCT = 0.60   # Max aggregate deployment across all pairs
MAX_DAILY_LOSS_PCT = 0.04       # Circuit breaker triggers at 4% daily degradation
MAX_DRAWDOWN = 0.12             # Lifetime structural account stop at 12%

# --- Target allocations used by the math solver ---
POSITION_FRACTION = 0.15        # Base sizing target fraction (15%)
MIN_EQUITY_FRACTION = 0.05      # Minimum allocation floor per trade (5%)
MAX_EQUITY_FRACTION = 0.25      # Maximum allocation ceiling per trade (25%)
MIN_NOTIONAL_PER_TRADE = 10.0   # Minimum dollar size Alpaca allows for crypto
MAX_NOTIONAL_PER_TRADE = 5000.0 # Hard cap threshold per order block

# ============================================
# SIGNAL DEPLOYMENT PARAMETERS
# ============================================
BASE_THRESHOLD = 0.58           # Range trading baseline execution floor
MAX_THRESHOLD = 0.72            # Macro bear market safety hurdle rate
MIN_PROFIT_TARGET_PCT = 0.012   # Range scalp profit threshold (1.2%)

# ============================================
# EXIT & TRAILING PARAMETERS
# ============================================
TRAILING_STOP_ACTIVATION_PCT = 0.015  # Trailing engine wakes up at 1.5% profit
TRAILING_STOP_DISTANCE_PCT = 0.008    # Trails the asset high from a 0.8% distance
MAX_HOLD_HOURS = 168                   # 1-week absolute structural macro hold limit

# ============================================
# PERSISTENCE & OUTPUT LOGGING
# ============================================
TRADE_LOG_CSV = "trade_log.csv"
GOOGLE_CREDENTIALS_FILE = "credentials.json"
GOOGLE_SHEETS_NAME = "TradingBot"