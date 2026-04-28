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

        # FEATURE ENGINEERING (simple but stable)
        df["return"] = df["close"].pct_change()
        df["momentum"] = df["close"] - df["close"].shift(3)
        df["volatility"] = df["return"].rolling(5).std()

        df = df.dropna()

        X = df[["return", "momentum", "volatility"]].values
        y = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]
        X = X[:-1]

        X = self._clean(X)

        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)

        self.is_trained = True
        print("🧠 ML TRAINED SUCCESSFULLY")

    def predict(self, row):
        if not self.is_trained:
            return 0.5

        X = np.array([[row["return"], row["momentum"], row["volatility"]]])
        X = self._clean(X)
        X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0][1]
        return float(prob)