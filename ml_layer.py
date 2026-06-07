import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"
DATA_PATH = "training_data.json"


class MLSpecialist:
    def __init__(self):
        self.model = self._load_model()

    # -----------------------------
    # Feature Engineering
    # -----------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["return_4h"] = df["close"].pct_change(4)
        df["return_12h"] = df["close"].pct_change(12)
        df["return_24h"] = df["close"].pct_change(24)

        df["volatility_24h"] = df["return_4h"].rolling(24).std()

        df["ma20"] = df["close"].rolling(20).mean()
        df["ma50"] = df["close"].rolling(50).mean()

        df["ma20_dist"] = (df["close"] - df["ma20"]) / df["ma20"]
        df["ma50_dist"] = (df["close"] - df["ma50"]) / df["ma50"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["rsi"] = 100 - (100 / (1 + rs))

        df["volume_change"] = df["volume"].pct_change()

        return df.dropna()

    # -----------------------------
    # Target Creation
    # -----------------------------
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        future_return = (df["close"].shift(-3) - df["close"]) / df["close"]
        df["target"] = (future_return > 0).astype(int)
        return df.dropna()

    # -----------------------------
    # Training
    # -----------------------------
    def train_price_model(self, df: pd.DataFrame):
        if len(df) < 200:
            return

        df = self._build_features(df)
        df = self._create_target(df)

        features = [
            "return_4h",
            "return_12h",
            "return_24h",
            "volatility_24h",
            "ma20_dist",
            "ma50_dist",
            "rsi",
            "volume_change"
        ]

        X = df[features]
        y = df["target"]

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        self.model.fit(X, y)
        self._save_model()

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, feature_row: pd.Series) -> float:
        if self.model is None:
            return 0.5

        try:
            X = feature_row.values.reshape(1, -1)
            return float(self.model.predict_proba(X)[0][1])
        except:
            return 0.5

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save_model(self):
        try:
            joblib.dump(self.model, MODEL_PATH)
        except:
            pass

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                return joblib.load(MODEL_PATH)
            except:
                return None
        return None