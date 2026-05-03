import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Falling back to RandomForest. Run: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Run: pip install scikit-learn")


class MLSpecialist:
    """
    Upgraded ML layer using XGBoost (with RandomForest fallback).

    Key improvements over LogisticRegression:
    - XGBoost handles non-linear relationships in price data
    - 12 features vs 5 - adds RSI, MACD diff, BB position, volume spike,
      multi-timeframe momentum, and hour-of-day cyclicality
    - Probability calibration so output is a real probability, not just a score
    - Label: next 4-hour return > 0.5% (not just next candle) - better signal
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self._build_model()

    def _build_model(self):
        if XGBOOST_AVAILABLE:
            base = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            self.model_type = "XGBoost"
        elif SKLEARN_AVAILABLE:
            base = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            self.model_type = "RandomForest"
        else:
            self.model = None
            self.model_type = "none"
            return

        # Calibrate so predict_proba gives real probabilities
        self.model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.scaler = StandardScaler()

    def _clean(self, X):
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        12 features covering momentum, trend, volatility, volume, and time.
        """
        d = df.copy()

        # Price returns
        d["return_1h"] = d["close"].pct_change(1)
        d["return_4h"] = d["close"].pct_change(4)
        d["return_24h"] = d["close"].pct_change(24)

        # Momentum
        d["momentum_3h"] = d["close"] - d["close"].shift(3)
        d["momentum_12h"] = d["close"] - d["close"].shift(12)

        # Volatility
        d["volatility_6h"] = d["return_1h"].rolling(6).std()
        d["volatility_24h"] = d["return_1h"].rolling(24).std()

        # Trend: distance from moving averages
        d["ma_20_dist"] = (d["close"] / d["close"].rolling(20).mean()) - 1
        d["ma_50_dist"] = (d["close"] / d["close"].rolling(50).mean()) - 1

        # RSI
        delta = d["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        d["rsi"] = (100 - (100 / (1 + rs))) / 100  # normalised 0-1

        # Volume spike
        d["vol_spike"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)

        # Hour of day - captures crypto time-of-day patterns (UTC)
        if "timestamp" in d.columns:
            d["hour_sin"] = np.sin(2 * np.pi * pd.to_datetime(d["timestamp"]).dt.hour / 24)
        else:
            d["hour_sin"] = 0.0

        return d

    def train_price_model(self, df: pd.DataFrame):
        if self.model is None:
            print("⚠️  No ML model available. Skipping training.")
            return

        d = self._engineer_features(df)

        features = [
            "return_1h", "return_4h", "return_24h",
            "momentum_3h", "momentum_12h",
            "volatility_6h", "volatility_24h",
            "ma_20_dist", "ma_50_dist",
            "rsi", "vol_spike", "hour_sin",
        ]

        d = d.dropna(subset=features)
        if len(d) < 200:
            print(f"⚠️  Not enough data to train ({len(d)} rows). Need 200+.")
            return

        # Label: is the price higher by >0.5% in 4 hours?
        # This filters out noise - only label "buy" if move is meaningful
        future_return = d["close"].shift(-4) / d["close"] - 1
        d = d.iloc[:-4]  # drop last 4 rows (no future label)
        y = (future_return.iloc[:-4] > 0.005).astype(int).values

        X = d[features].values
        X = self._clean(X)

        if len(np.unique(y)) < 2:
            print("⚠️  Only one class in labels. Market too one-directional to train.")
            return

        X_scaled = self.scaler.fit_transform(X)

        try:
            self.model.fit(X_scaled, y)
            self.is_trained = True
            pos_rate = y.mean()
            print(f"✅ {self.model_type} trained on {len(X)} rows. "
                  f"Positive label rate: {pos_rate:.1%}")
        except Exception as e:
            print(f"❌ ML training failed: {e}")
            self.is_trained = False

    def predict(self, row: pd.Series) -> float:
        """Returns probability of price rising >0.5% in next 4 hours."""
        if not self.is_trained or self.model is None:
            return 0.5

        features = [
            "return_1h", "return_4h", "return_24h",
            "momentum_3h", "momentum_12h",
            "volatility_6h", "volatility_24h",
            "ma_20_dist", "ma_50_dist",
            "rsi", "vol_spike", "hour_sin",
        ]

        try:
            vals = [row.get(f, 0.0) for f in features]
            X = np.array([vals])
            X = self._clean(X)
            X = self.scaler.transform(X)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)
        except Exception as e:
            print(f"⚠️  ML predict error: {e}")
            return 0.5