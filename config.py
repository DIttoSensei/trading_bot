import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# API CONFIGURATION
# ============================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# ============================================
# TRADING SYMBOLS (Start with just 1-2)
# ============================================
TRADE_SYMBOLS = ["BTC/USD", "ETH/USD"]  # Start with only these

# ============================================
# RISK MANAGEMENT (CONSERVATIVE FOR PROFITABILITY)
# ============================================
MAX_POSITION_SIZE_PCT = 0.15  # Max 15% of portfolio per trade (reduced from 25%)
MAX_TOTAL_EXPOSURE_PCT = 0.40  # Max 40% total deployed (reduced from unlimited)
MAX_DAILY_LOSS_PCT = 0.05  # Stop trading after 5% daily loss
MAX_DRAWDOWN = 0.15  # Max 15% drawdown before shutdown

# ============================================
# TRADE FILTERS (Quality over quantity)
# ============================================
MIN_CONFIDENCE_THRESHOLD = 0.65  # Increased from 0.30-0.50
MIN_PROFIT_TARGET_PCT = 0.015  # 1.5% minimum expected profit (covers fees + spread)
MAX_SPREAD_PCT = 0.002  # 0.2% max spread (avoid high-spread hours)

# ============================================
# TECHNICAL INDICATORS
# ============================================
LOOKBACK_HOURS = 168  # 7 days (increased from shorter periods)
ML_TRAIN_MIN_ROWS = 200  # Minimum data for ML

# ============================================
# EXIT STRATEGY (Let winners run)
# ============================================
TRAILING_STOP_ACTIVATION_PCT = 0.02  # Start trailing after 2% profit
TRAILING_STOP_DISTANCE_PCT = 0.01  # Trail by 1%
MAX_HOLD_HOURS = 48  # Force exit after 48 hours

# ============================================
# MEMORY & STATE
# ============================================
TRADE_LOG_CSV = "trade_log.csv"
GOOGLE_CREDENTIALS_FILE = "google_credentials.json"
GOOGLE_SHEETS_NAME = "TradingBot"
