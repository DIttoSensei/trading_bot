import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MLSpecialist:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

    def _clean(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def train_price_model(self, df: pd.DataFrame):
        df = df.copy()

        # FEATURE ENGINEERING (Expanded for 2-month context)
        df["return"] = df["close"].pct_change()
        
        # Keep short term
        df["momentum_3h"] = df["close"] - df["close"].shift(3)
        
        # Add Long Term (24h momentum)
        df["momentum_24h"] = df["close"] - df["close"].shift(24)
        
        # Expanded Volatility (Last 24h instead of 5h)
        df["volatility_24h"] = df["return"].rolling(24).std()
        
        # Trend indicator (Price relative to 50-hour average)
        df["ma_50_dist"] = df["close"] / df["close"].rolling(50).mean()

        df = df.dropna()

        # Update the features list
        features = ["return", "momentum_3h", "momentum_24h", "volatility_24h", "ma_50_dist"]
        X = df[features].values
        y = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]
        X = X[:-1]

        X = self._clean(X)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

        self.is_trained = True
        print("🧠 ML TRAINED SUCCESSFULLY WITH 2-MONTH CONTEXT")

    def predict(self, row):
        if not self.is_trained:
            return 0.5

        # Match the new features exactly
        X = np.array([[
            row["return"], 
            row["momentum_3h"], 
            row["momentum_24h"], 
            row["volatility_24h"], 
            row["ma_50_dist"]
        ]])
        X = self._clean(X)
        X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0][1]
        return float(prob)