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
        if XGBOOST_AVAILABLE:
            base = XGBClassifier(
                n_estimators=300, 
                max_depth=4,
                learning_rate=0.03, # Lowered for 3-year stability
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
            self.model_type = "XGBoost"
        else:
            base = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            self.model_type = "RandomForest"
        
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)

    def _clean(self, X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        
        # Ensure timestamp is datetime for cyclical encoding
        if 'timestamp' in d.columns:
            d['timestamp'] = pd.to_datetime(d['timestamp'])
            d["hour_sin"] = np.sin(2 * np.pi * d["timestamp"].dt.hour / 24)
        else:
            # If main.py passes index as datetime
            d["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)

        d["return_1h"] = d["close"].pct_change(1)
        d["return_4h"] = d["close"].pct_change(4)
        d["return_24h"] = d["close"].pct_change(24)
        d["momentum_3h"] = d["close"] - d["close"].shift(3)
        d["momentum_12h"] = d["close"] - d["close"].shift(12)
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["volatility_24h"] = d["return_1h"].rolling(24).std()
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1
        d["ma_50_dist"] = (d["close"] / d["close"].rolling(50).mean()) - 1

        # RSI Logic
        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi"] = (100 - (100 / (1 + rs.fillna(0)))) / 100

        d["vol_spike"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)
        
        return d.fillna(0)

    def train_price_model(self, df: pd.DataFrame):
        if self.model is None: return

        d = self._engineer_features(df)
        
        # --- DYNAMIC LABELING (Layer 2 Improvement) ---
        # Instead of fixed 0.5%, we use 1 standard deviation of the 4h move
        # This makes the bot smarter for PEPE vs BTC
        threshold = d["return_4h"].std() * 1.2 
        future_return = d["close"].shift(-4) / d["close"] - 1
        
        y = (future_return > threshold).astype(int)
        
        # Cleanup for training
        X = d[self.features].values[:-4]
        y = y.values[:-4]
        
        if len(np.unique(y)) < 2:
            print("⚠️ Market too flat. Skipping training.")
            return

        X_scaled = self.scaler.fit_transform(self._clean(X))
        
        try:
            self.model.fit(X_scaled, y)
            self.is_trained = True
            print(f"✅ {self.model_type} trained on {len(X)} rows. Target: >{threshold:.2%}")
        except Exception as e:
            print(f"❌ ML training failed: {e}")

    def predict(self, row: pd.Series) -> float:
        if not self.is_trained: return 0.5

        try:
            # Ensure we only use the features the model was trained on
            vals = [row.get(f, 0.0) for f in self.features]
            X = self.scaler.transform(self._clean(np.array([vals])))
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            return 0.5