"""
LightGBM Quantile Demand Forecaster:
Trains gradient boosted decision trees for probabilistic multi-quantile forecasting
(q0.10, q0.50, q0.90) with point-in-time lag, rolling, and promotional features.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb


def build_forecasting_features(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs leakage-free point-in-time lag, rolling, and calendar features.
    """
    df = df_panel.copy().sort_values(["sku_id", "day_index"])
    
    # 1. Lags of observed POS sales
    for lag in [1, 2, 7, 14, 21]:
        df[f"sales_lag_{lag}"] = df.groupby("sku_id")["pos_sales"].shift(lag)

    # 2. Rolling statistics (shifted by 1 to prevent contemporary leakage)
    for window in [7, 14, 28]:
        grouped = df.groupby("sku_id")["pos_sales"]
        shifted = grouped.shift(1)
        df[f"rolling_mean_{window}"] = shifted.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        df[f"rolling_std_{window}"] = shifted.rolling(window, min_periods=1).std().fillna(0).reset_index(0, drop=True)
        df[f"rolling_zeros_{window}"] = (shifted == 0).rolling(window, min_periods=1).sum().reset_index(0, drop=True)

    # 3. Categorical & Calendar Encoding
    df["category_code"] = df["category"].astype("category").cat.codes
    df["aisle_code"] = df["aisle_id"].astype("category").cat.codes
    df["dow"] = df["day_of_week"]
    df["promo_int"] = df["promo_flag"].astype(int)

    # Fill initial boundary NaNs cleanly
    feature_cols = [c for c in df.columns if "lag_" in c or "rolling_" in c]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    return df


class LightGBMQuantileForecaster:
    """Trains quantile regression models for q in [0.10, 0.50, 0.90]."""

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90], random_state: int = 42):
        self.quantiles = quantiles
        self.random_state = random_state
        self.models: Dict[float, lgb.Booster] = {}
        self.feature_names: List[str] = []

    def fit(self, df_train: pd.DataFrame, target_col: str = "pos_sales", sample_weight: np.ndarray = None):
        """Fits LightGBM boosters for each target quantile."""
        self.feature_names = [
            "sales_lag_1", "sales_lag_2", "sales_lag_7", "sales_lag_14", "sales_lag_21",
            "rolling_mean_7", "rolling_std_7", "rolling_zeros_7",
            "rolling_mean_14", "rolling_std_14",
            "rolling_mean_28", "rolling_std_28",
            "category_code", "aisle_code", "dow", "promo_int", "selling_price"
        ]
        
        X = df_train[self.feature_names].values
        y = df_train[target_col].values

        for q in self.quantiles:
            params = {
                "objective": "quantile",
                "alpha": q,
                "metric": "quantile",
                "boosting_type": "gbdt",
                "n_estimators": 120,
                "learning_rate": 0.08,
                "num_leaves": 31,
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }
            train_data = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
            model = lgb.train(params, train_data, num_boost_round=120)
            self.models[q] = model

    def predict_quantiles(self, df_test: pd.DataFrame) -> Dict[float, np.ndarray]:
        """Generates predictions for all fitted quantiles."""
        X_test = df_test[self.feature_names].values
        preds = {}
        for q, model in self.models.items():
            raw_pred = model.predict(X_test)
            preds[q] = np.maximum(0.0, raw_pred)
            
        # Ensure quantile monotonicity (q0.10 <= q0.50 <= q0.90)
        q_sorted = sorted(self.quantiles)
        for i in range(len(q_sorted) - 1):
            q_low = q_sorted[i]
            q_high = q_sorted[i + 1]
            preds[q_high] = np.maximum(preds[q_high], preds[q_low])
            
        return preds
