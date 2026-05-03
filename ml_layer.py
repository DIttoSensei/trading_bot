import numpy as np
import pandas as pd
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class MLSpecialist:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.features = [
            "return_1h", "return_4h", "return_24h",
            "momentum_3h", "momentum_12h",
            "volatility_6h", "volatility_24h",
            "ma_20_dist", "ma_50_dist",
            "rsi", "vol_spike", "hour_sin",
        ]
        self._build_model()

    def _build_model(self):
        """
        Optimized for GitHub Actions (2-core CPUs). 
        Reduced estimators and CV folds to prevent 14-minute hangs.
        """
        if XGBOOST_AVAILABLE:
            base = XGBClassifier(
                n_estimators=150,     # Reduced from 300 for speed
                max_depth=3,          # Shallower trees train faster
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=2,             # Use both cores on GitHub runner
                eval_metric="logloss",
                random_state=42
            )
            self.model_type = "XGBoost"
        else:
            base = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=2)
            self.model_type = "RandomForest"
        
        # Reduced cv from 3 to 2. This immediately cuts training time by 33%.
        # Isotonic is kept as per your preference for probability accuracy.
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=2)

    def _clean(self, X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized feature engineering. 
        Calculates indicators for the whole block at once.
        """
        d = df.copy()
        
        # Cyclical Hour Encoding
        if 'timestamp' in d.columns:
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d["hour_sin"] = np.sin(2 * np.pi * d["timestamp"].dt.hour / 24)
        else:
            d["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)

        # Returns & Momentum
        d["return_1h"] = d["close"].pct_change(1)
        d["return_4h"] = d["close"].pct_change(4)
        d["return_24h"] = d["close"].pct_change(24)
        d["momentum_3h"] = d["close"] - d["close"].shift(3)
        d["momentum_12h"] = d["close"] - d["close"].shift(12)
        
        # Volatility & MAs
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["volatility_24h"] = d["return_1h"].rolling(24).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean().replace(0, np.nan)) - 1
        d["ma_50_dist"] = (d["close"] / d["close"].rolling(50).mean().replace(0, np.nan)) - 1

        # RSI
        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi"] = (100 - (100 / (1 + rs.fillna(0)))) / 100

        # Volume Spike
        d["vol_spike"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)
        
        return d.fillna(0)

    def train_price_model(self, df: pd.DataFrame):
        """
        Retrains the model using your Dynamic Labeling logic.
        """
        if self.model is None: return

        # Only use the most recent 3000 rows for training to ensure speed on GitHub.
        # 3000 hours is ~4 months, usually enough for a regime-aware hourly bot.
        df_train = df.tail(3000).copy()
        d = self._engineer_features(df_train)
        
        # Dynamic Labeling: 1.2 Std Dev of 4-hour moves
        threshold = d["return_4h"].std() * 1.2 
        future_return = d["close"].shift(-4) / d["close"] - 1
        
        y = (future_return > threshold).astype(int)
        
        # Shift features to align with future labels
        X = d[self.features].values[:-4]
        y = y.values[:-4]
        
        if len(np.unique(y)) < 2:
            print("⚠️ Market too flat for classification. Using neutral bias.")
            return

        X_scaled = self.scaler.fit_transform(self._clean(X))
        
        try:
            self.model.fit(X_scaled, y)
            self.is_trained = True
            print(f"✅ {self.model_type} trained. Target Move: >{threshold:.2%}")
        except Exception as e:
            print(f"❌ ML training failed: {e}")

    def predict(self, row: pd.Series) -> float:
        """
        Predicts probability of a significant move for a single data row.
        """
        if not self.is_trained: return 0.5

        try:
            # Extract features in the correct order
            vals = [row.get(f, 0.0) for f in self.features]
            X = self.scaler.transform(self._clean(np.array([vals])))
            
            # Calibration makes predict_proba very accurate but slower to train.
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return 0.5