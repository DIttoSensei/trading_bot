import os
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.pkl"


class MLSpecialist:
    def __init__(self):
        self.model = self._load_model()

    # -----------------------------
    # SAFE FEATURE ENGINEERING
    # -----------------------------
    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["return_4h"] = df["close"].pct_change(4)
        df["return_12h"] = df["close"].pct_change(12)
        df["return_24h"] = df["close"].pct_change(24)

        df["volatility_24h"] = df["return_4h"].rolling(24).std()

        ma20 = df["close"].rolling(20).mean()
        ma50 = df["close"].rolling(50).mean()

        df["ma20_dist"] = (df["close"] - ma20) / ma20.replace(0, np.nan)
        df["ma50_dist"] = (df["close"] - ma50) / ma50.replace(0, np.nan)

        # RSI SAFE
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)

        df["rsi"] = 100 - (100 / (1 + rs))

        df["volume_change"] = df["volume"].pct_change()

        # HARD CLEAN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        return df

    # -----------------------------
    # TARGET (FUTURE RETURN LABEL)
    # -----------------------------
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        future_return = (df["close"].shift(-3) - df["close"]) / df["close"]
        df["target"] = (future_return > 0).astype(int)
        return df.dropna()

    # -----------------------------
    # TRAIN
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
    # PREDICT (SAFE)
    # -----------------------------
    def predict(self, feature_row: pd.Series) -> float:
        if self.model is None:
            return 0.5

        try:
            X = pd.DataFrame([feature_row.values])
            return float(self.model.predict_proba(X)[0][1])
        except:
            return 0.5

    # -----------------------------
    # PERSISTENCE
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