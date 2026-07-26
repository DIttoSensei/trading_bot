"""
ml_layer.py
Now with:
  - Walk-forward holdout evaluation every training cycle (prints accuracy/AUC/Brier)
  - XGBoost classifier instead of LogisticRegression
  - Expanded feature set (acceleration, volatility regime, wick structure, volume-weighted return)
  - Class balance handling via scale_pos_weight
  - Early stopping on the holdout set to fight overfitting
  - Secondary "magnitude" target logged alongside direction, as a sanity check for
    whether there's ANY learnable structure even if direction alone is too hard
  - Metrics persisted to ml_metrics_history.json so you can track quality over time
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, UTC
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES = [
    "return_4h", "return_12h", "return_24h",
    "volatility_24h", "ma20_dist", "ma50_dist",
    "rsi", "vol_chg",
    "rsi_slope", "vol_ratio", "ema_spread",
    "upper_wick", "lower_wick", "vol_weighted_ret",
]

METRICS_LOG = "ml_metrics_history.json"


class MLSpecialist:
    def __init__(self, symbol: str = "BTC/USD"):
        self.symbol = symbol
        self.model_path = f"model_{symbol.replace('/', '')}.pkl"
        self.scaler_path = f"scaler_{symbol.replace('/', '')}.pkl"
        self.model = None
        self.scaler = None
        self._load()
        self.last_metrics = {}

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

        # --- New features ---
        df["rsi_slope"] = df["rsi"].diff(3)

        df["vol_ratio"] = df["volatility_24h"] / df["volatility_24h"].rolling(48).mean().replace(0, np.nan)

        ema_9 = df["close"].ewm(span=9).mean()
        ema_21 = df["close"].ewm(span=21).mean()
        df["ema_spread"] = (ema_9 - ema_21) / df["close"].rolling(14).std().replace(0, np.nan)

        df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"]
        df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"]

        df["vol_weighted_ret"] = df["return_4h"] / (df["vol_chg"].abs() + 1e-6)

        df = df.replace([np.inf, -np.inf], np.nan)
        return df

    def _fit_one(self, X_train, y_train, X_test, y_test, label: str):
        """Fits one XGB model with early stopping, returns (fitted_scaler, fitted_model, metrics_dict)."""
        pos_rate = y_train.mean()
        scale_pos_weight = (1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=15,
            random_state=42,
        )
        clf.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        test_probs = clf.predict_proba(X_test_scaled)[:, 1]
        test_preds = (test_probs > 0.5).astype(int)

        acc = accuracy_score(y_test, test_preds)
        try:
            auc = roc_auc_score(y_test, test_probs)
        except ValueError:
            auc = float("nan")
        brier = brier_score_loss(y_test, test_probs)

        metrics = {
            "target": label,
            "test_accuracy": round(float(acc), 4),
            "test_auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "test_brier": round(float(brier), 4),
            "pos_rate_train": round(float(pos_rate), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "best_iteration": int(getattr(clf, "best_iteration", clf.n_estimators)),
        }
        print(f"[ML:{self.symbol}] {label.upper()} HOLDOUT — acc={acc:.3f} auc={auc:.3f} "
              f"brier={brier:.3f} pos_rate={pos_rate:.2%} n_train={len(X_train)} n_test={len(X_test)} "
              f"best_iter={metrics['best_iteration']}")

        return scaler, clf, metrics

    def train(self, df: pd.DataFrame) -> bool:
        if len(df) < 250:
            print(f"[ML:{self.symbol}] Too few rows ({len(df)}), skip train.")
            return False

        df_base = self._build_features(df)
        future = (df_base["close"].shift(-4) - df_base["close"]) / df_base["close"].replace(0, np.nan)
        df_base["target_direction"] = (future > 0).astype(int)
        df_base["target_magnitude"] = (future.abs() > 0.01).astype(int)  # >1% move either direction

        df_train_full = df_base.dropna(
            subset=FEATURES + ["target_direction", "target_magnitude"]
        ).reset_index(drop=True)

        if len(df_train_full) < 100:
            print(f"[ML:{self.symbol}] Too few clean rows, skip train.")
            return False

        X = df_train_full[FEATURES]
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

        all_metrics = []

        # --- Directional model (the one actually used live) ---
        y_dir = df_train_full["target_direction"]
        y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
        scaler_dir, clf_dir, metrics_dir = self._fit_one(
            X_train, y_dir_train, X_test, y_dir_test, "direction"
        )
        all_metrics.append(metrics_dir)

        # --- Magnitude model (diagnostic only, tells us if ANY structure exists) ---
        y_mag = df_train_full["target_magnitude"]
        y_mag_train, y_mag_test = y_mag.iloc[:split_idx], y_mag.iloc[split_idx:]
        try:
            _, _, metrics_mag = self._fit_one(
                X_train, y_mag_train, X_test, y_mag_test, "magnitude"
            )
            all_metrics.append(metrics_mag)
        except Exception as e:
            print(f"[ML:{self.symbol}] magnitude model skipped: {e}")

        for m in all_metrics:
            self._append_metrics_log(m)
        self.last_metrics = {m["target"]: m for m in all_metrics}

        # --- Refit directional model on ALL data (train+holdout) for live prediction ---
        scaler_final = StandardScaler()
        X_scaled_full = scaler_final.fit_transform(X)
        clf_final = XGBClassifier(
            n_estimators=metrics_dir["best_iteration"] or 150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=5,
            scale_pos_weight=(1 - y_dir.mean()) / y_dir.mean() if 0 < y_dir.mean() < 1 else 1.0,
            eval_metric="logloss",
            random_state=42,
        )
        clf_final.fit(X_scaled_full, y_dir)

        self.model = clf_final
        self.scaler = scaler_final
        self._save()
        return True

    def _append_metrics_log(self, metrics: dict):
        try:
            history = []
            if os.path.exists(METRICS_LOG):
                with open(METRICS_LOG) as f:
                    history = json.load(f)
            metrics = dict(metrics)
            metrics["symbol"] = self.symbol
            metrics["timestamp"] = datetime.now(UTC).isoformat()
            history.append(metrics)
            history = history[-1000:]
            with open(METRICS_LOG, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"[ML:{self.symbol}] metrics log write failed: {e}")

    def get_latest_features(self, df: pd.DataFrame):
        df_base = self._build_features(df)
        df_clean = df_base.dropna(subset=FEATURES)
        if df_clean.empty:
            return None
        return df_clean[FEATURES].iloc[[-1]]

    def predict(self, X) -> float:
        if self.model is None or self.scaler is None or X is None:
            return 0.5
        try:
            X_scaled = self.scaler.transform(X)
            return float(np.clip(self.model.predict_proba(X_scaled)[0][1], 0.0, 1.0))
        except Exception as e:
            print(f"[ML:{self.symbol}] predict error: {e}")
            return 0.5

    def _save(self):
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
        except Exception as e:
            print(f"[ML:{self.symbol}] save failed: {e}")

    def _load(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            except Exception as e:
                print(f"[ML:{self.symbol}] load failed: {e}")
