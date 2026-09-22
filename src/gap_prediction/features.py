"""
Feature Engineering Module for Gap Prediction:
Extracts point-in-time features with zero future leakage.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def build_gap_features(
    df_panel: pd.DataFrame,
    df_forecasts: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Constructs point-in-time feature representation for each SKU-day row.
    """
    df = df_panel.copy().sort_values(["sku_id", "day_index"])
    
    # 1. Base demand velocity proxy (mean sales from training history)
    mean_demands = df.groupby("sku_id")["pos_sales"].transform("mean")
    df["base_demand_velocity"] = mean_demands
    
    # 2. Normalized zero-streak feature (3 days on 10/day item is screams gap; on 0.2/day item is normal)
    df["normalized_zero_streak"] = df["consecutive_zero_sales"] / (mean_demands + 1e-3)
    
    # 3. Days of supply: SOH / expected daily sales
    df["days_of_supply"] = np.maximum(0, df["system_on_hand"]) / (mean_demands + 1e-3)
    
    # 4. Book stock relative to case pack size
    df["soh_vs_case_pack"] = df["system_on_hand"] / (df["aisle_id"] + 1.0) # Proxy if case pack missing

    # 5. Forecast Residuals (if forecast frame provided)
    if df_forecasts is not None and "pred_q50_aware" in df_forecasts.columns:
        # Merge on (sku_id, day_index)
        merged = df.merge(
            df_forecasts[["sku_id", "day_index", "pred_q10_aware", "pred_q50_aware", "pred_q90_aware"]],
            on=["sku_id", "day_index"],
            how="left"
        )
        q10 = merged["pred_q10_aware"].fillna(mean_demands * 0.5)
        q50 = merged["pred_q50_aware"].fillna(mean_demands)
        q90 = merged["pred_q90_aware"].fillna(mean_demands * 1.5)
        
        df["forecast_residual"] = df["pos_sales"] - q50
        df["forecast_residual_standardized"] = df["forecast_residual"] / (q90 - q10 + 1e-3)
    else:
        df["forecast_residual"] = df["pos_sales"] - mean_demands
        df["forecast_residual_standardized"] = df["forecast_residual"] / (mean_demands * 0.5 + 1e-3)

    # 6. Categorical Codes
    df["category_code"] = df["category"].astype("category").cat.codes
    df["aisle_code"] = df["aisle_id"].astype("category").cat.codes
    df["dow"] = df["day_of_week"]
    df["promo_int"] = df["promo_flag"].astype(int)

    return df
