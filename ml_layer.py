"""
ml_layer.py
Critical fixes:
  - prepare_latest_features() returns DataFrame (not numpy array)
    → fixes "X does not have valid feature names" sklearn error
  - All division uses .replace(0, np.nan) → no inf
  - Final dropna() on FEATURES columns guaranteed
  - Per-symbol model file
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "return_4h", "return_12h", "return_24h",
    "volatility_24h", "ma20_dist", "ma50_dist",
    "rsi", "vol_chg",
]


class MLSpecialist:
    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        self.model_path = f"model_{symbol.replace('/', '')}.pkl"
        self.model = self._load()

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        df["return_4h"]  = df["close"].pct_change(4)
        df["return_12h"] = df["close"].pct_change(12)
        df["return_24h"] = df["close"].pct_change(24)
        df["volatility_24h"] = df["return_4h"].rolling(24).std()

        ma20 = df["close"].rolling(20).mean()
        ma50 = df["close"].rolling(50).mean()
        df["ma20_dist"] = (df["close"] - ma20) / ma20.replace(0, np.nan)
        df["ma50_dist"] = (df["close"] - ma50) / ma50.replace(0, np.nan)

        delta = df["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["vol_chg"] = df["volume"].pct_change()

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=FEATURES).reset_index(drop=True)
        return df

    def _build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        future = (df["close"].shift(-4) - df["close"]) / df["close"].replace(0, np.nan)
        df["target"] = (future > 0).astype(int)
        return df.dropna(subset=["target"]).reset_index(drop=True)

    def train(self, df: pd.DataFrame) -> bool:
        if len(df) < 200:
            print(f"[ML:{self.symbol}] Too few rows ({len(df)}), skip train.")
            return False

        df = self._build_features(df)
        df = self._build_target(df)

        if len(df) < 50:
            print(f"[ML:{self.symbol}] Too few clean rows, skip train.")
            return False

        X = df[FEATURES]
        y = df["target"]

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self.model.fit(X, y)
        self._save()
        print(f"[ML:{self.symbol}] Trained on {len(X)} samples.")
        return True

    def get_latest_features(self, df: pd.DataFrame):
        """Returns single-row DataFrame with FEATURES columns, or None."""
        df = self._build_features(df)
        if df.empty:
            return None
        return df[FEATURES].iloc[[-1]]  # DataFrame, not numpy — preserves feature names

    def predict(self, X) -> float:
        """Returns float 0.0–1.0. Neutral 0.5 on any failure."""
        if self.model is None or X is None:
            return 0.5
        try:
            return float(np.clip(self.model.predict_proba(X)[0][1], 0.0, 1.0))
        except Exception as e:
            print(f"[ML:{self.symbol}] predict error: {e}")
            return 0.5

    def _save(self):
        try:
            joblib.dump(self.model, self.model_path)
        except Exception as e:
            print(f"[ML:{self.symbol}] save failed: {e}")

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except Exception as e:
                print(f"[ML:{self.symbol}] load failed: {e}")
        return None
