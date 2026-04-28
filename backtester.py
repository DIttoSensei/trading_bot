import numpy as np
import pandas as pd


def walk_forward_gate(df: pd.DataFrame, min_rows: int, test_window: int, min_signals: int, min_winrate: float):
    """
    Walk-forward check using a directional momentum rule as a safety gate.
    This is not the final strategy; it blocks trading when market regime is hostile.
    """
    if df is None or len(df) < min_rows:
        return {"pass": False, "reason": "not_enough_rows", "signals": 0, "winrate": 0.0}

    data = df.copy().reset_index(drop=True)
    data["ret_1"] = data["close"].pct_change()
    data["mom_3"] = data["close"] - data["close"].shift(3)
    data["vol_8"] = data["ret_1"].rolling(8).std()
    data = data.dropna()

    if len(data) < test_window + 10:
        return {"pass": False, "reason": "not_enough_clean_rows", "signals": 0, "winrate": 0.0}

    test = data.tail(test_window).copy()
    test["pred_long"] = (test["mom_3"] > 0) & (test["vol_8"] < test["vol_8"].rolling(20).mean().fillna(test["vol_8"]))
    test["fwd_ret"] = test["close"].shift(-1) / test["close"] - 1.0
    test = test.dropna()

    trades = test[test["pred_long"]]
    signals = int(len(trades))
    if signals < min_signals:
        return {"pass": False, "reason": "not_enough_signals", "signals": signals, "winrate": 0.0}

    wins = int((trades["fwd_ret"] > 0).sum())
    winrate = wins / signals if signals else 0.0

    return {
        "pass": bool(winrate >= min_winrate),
        "reason": "ok" if winrate >= min_winrate else "low_winrate",
        "signals": signals,
        "winrate": float(winrate),
    }
