import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class MLSpecialist:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.is_trained = False

    def train_price_model(self, df: pd.DataFrame):
        """
        Trains an ensemble regressor on rolling momentum variables
        to extract shift vectors.
        """
        try:
            if len(df) < 30:
                return

            d = df.copy()
            d["target"] = d["close"].shift(-1) / d["close"] - 1.0
            d = self._build_features(d).dropna()

            if len(d) < 20:
                return

            feature_cols = ["return_1h", "volatility_6h", "ma_20_dist"]
            X = d[feature_cols].values
            y = d["target"].values

            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            print(f"⚠️ Failed training pipeline on Layer 2: {e}")

    def predict(self, feature_row: pd.Series) -> float:
        """
        Maps mathematical predictive probabilities into a standard 
        0.0 to 1.0 entry evaluation score.
        """
        if not self.is_trained:
            return 0.5
        try:
            feat_vector = np.array([[ 
                feature_row["return_1h"], 
                feature_row["volatility_6h"], 
                feature_row["ma_20_dist"] 
            ]])
            pred_return = float(self.model.predict(feat_vector)[0])
            
            # Map returns distribution mathematically into probability boundaries
            prob = 1.0 / (1.0 + np.exp(-pred_return * 150.0))
            return float(np.clip(prob, 0.0, 1.0))
        except Exception:
            return 0.5

    def _build_features(self, d: pd.DataFrame) -> pd.DataFrame:
        d["return_1h"] = d["close"].pct_change(1)
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1.0
        return d