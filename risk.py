import numpy as np
import pandas as pd


class RiskManager:
    """
    Manages drawdown limits and ATR-based trailing stops.

    New: TrailingStopTracker
    - Tracks peak price since entry for each symbol
    - Computes ATR-based trailing stop floor
    - Lets winners run while protecting against large reversals
    """

    def __init__(self, max_drawdown: float = 0.10):
        self.max_drawdown = max_drawdown
        self.peak_equity = None
        self.current_drawdown = 0.0

    def update(self, equity: float) -> float:
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        return self.current_drawdown

    def allow_trading(self, equity: float) -> bool:
        self.update(equity)
        return self.current_drawdown < self.max_drawdown


class TrailingStopTracker:
    """
    ATR-based trailing stop per symbol.

    How it works:
    - On entry, record the entry price and reset peak
    - Each tick, update the peak price seen since entry
    - Trailing stop = peak - (ATR * multiplier)
    - If current price drops below trailing stop → exit signal
    """

    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self.peak_prices: dict[str, float] = {}
        self.entry_prices: dict[str, float] = {}

    def on_entry(self, symbol: str, entry_price: float):
        self.peak_prices[symbol] = entry_price
        self.entry_prices[symbol] = entry_price

    def update_peak(self, symbol: str, current_price: float):
        if symbol in self.peak_prices:
            self.peak_prices[symbol] = max(self.peak_prices[symbol], current_price)

    def compute_atr(self, df: pd.DataFrame, window: int = 14) -> float:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else float(close.iloc[-1] * 0.02)

    def should_exit(self, symbol: str, current_price: float, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Returns (should_exit, trailing_stop_price).
        """
        if symbol not in self.peak_prices:
            return False, 0.0

        self.update_peak(symbol, current_price)
        atr = self.compute_atr(df)
        trailing_stop = self.peak_prices[symbol] - (atr * self.atr_multiplier)

        should_exit = current_price < trailing_stop
        return should_exit, trailing_stop

    def on_exit(self, symbol: str):
        self.peak_prices.pop(symbol, None)
        self.entry_prices.pop(symbol, None)