"""
ml_layer.py
Critical fixes:
  - prepare_latest_features() returns DataFrame (not numpy array)
    → fixes "X does not have valid feature names" sklearn error
  - All division uses .replace(0, np.nan) → no inf
  - Final dropna() on FEATURES columns guaranteed
  - Per-symbol model file
  - Separated training-set alignment from live-prediction feature scaling
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
        """Calculates indicators across the raw timeline without dropping missing rows yet."""
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
        return df

    def train(self, df: pd.DataFrame) -> bool:
        """Combines features and targets, drops edge columns, and fits the pipeline."""
        if len(df) < 200:
            print(f"[ML:{self.symbol}] Too few rows ({len(df)}), skip train.")
            return False

        # 1. Build indicators across the whole series
        df_base = self._build_features(df)
        
        # 2. Add the clean target vector looking exactly 4 hours ahead
        future = (df_base["close"].shift(-4) - df_base["close"]) / df_base["close"].replace(0, np.nan)
        df_base["target"] = (future > 0).astype(int)

        # 3. Safely drop empty boundary rows exclusively for the training set
        df_train = df_base.dropna(subset=FEATURES + ["target"]).reset_index(drop=True)

        if len(df_train) < 50:
            print(f"[ML:{self.symbol}] Too few clean training rows, skip train.")
            return False

        X = df_train[FEATURES]
        y = df_train["target"]

        # Re-fitting pipeline with higher regularization parameter C for more variance
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=10.0,
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
        """Extracts the single most recent row with clean features for live execution."""
        df_base = self._build_features(df)
        df_clean = df_base.dropna(subset=FEATURES)
        if df_clean.empty:
            return None
        return df_clean[FEATURES].iloc[[-1]]

    def predict(self, X) -> float:
        """Returns directional probability scalar bounded cleanly from 0.0 to 1.0."""
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
