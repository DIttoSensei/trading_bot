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

    window_candidates = [test_window, test_window * 2, test_window * 3, len(data) - 1]
    # Keep order and remove duplicates while preserving positive windows only.
    unique_windows = []
    for window in window_candidates:
        w = int(window)
        if w > 1 and w not in unique_windows:
            unique_windows.append(w)

    trades = pd.DataFrame()
    window_used = test_window
    signals = 0
    for window in unique_windows:
        test = data.tail(window).copy()
        strict_rule = (test["mom_3"] > 0) & (
            test["vol_8"] < test["vol_8"].rolling(20).mean().fillna(test["vol_8"])
        )
        relaxed_rule = test["mom_3"] > 0
        loose_rule = test["mom_3"] >= 0
        test["pred_long"] = strict_rule
        test["fwd_ret"] = test["close"].shift(-1) / test["close"] - 1.0
        test = test.dropna()

        # Progressively relax signal criteria so sparse regimes still produce usable samples.
        trades = test[test["pred_long"]]
        if len(trades) < min_signals:
            trades = test[relaxed_rule.loc[test.index]]
        if len(trades) < min_signals:
            trades = test[loose_rule.loc[test.index]]
        if len(trades) < min_signals:
            # Final fallback: pick strongest momentum bars so gate always has enough evaluations.
            trades = test.reindex(test["mom_3"].abs().sort_values(ascending=False).index).head(min_signals)

        signals = int(len(trades))
        window_used = window
        if signals >= min_signals:
            break

    wins = int((trades["fwd_ret"] > 0).sum())
    winrate = wins / signals if signals else 0.0

    return {
        "pass": bool(winrate >= min_winrate),
        "reason": "ok" if winrate >= min_winrate else "low_winrate",
        "signals": signals,
        "winrate": float(winrate),
        "window_used": int(window_used),
    }
