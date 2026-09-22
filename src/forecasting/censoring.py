"""
Censoring Correction & Bias Quantification Module for Demand Forecasting.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def create_censoring_mask(
    df: pd.DataFrame,
    soh_col: str = "system_on_hand",
    sales_col: str = "pos_sales",
    consecutive_zeros_col: str = "consecutive_zero_sales",
) -> np.ndarray:
    """
    Creates an observable censoring mask for periods where stockouts likely truncated demand.
    Observable proxy: SOH <= 0 OR (consecutive_zeros >= 4 and estimated base velocity > 1.0).
    """
    is_censored = (df[soh_col] <= 0) | (df[consecutive_zeros_col] >= 5)
    return is_censored.values


def quantify_censoring_bias(
    naive_forecasts: np.ndarray,
    censoring_aware_forecasts: np.ndarray,
    ground_truth_demand: np.ndarray = None,
) -> Dict[str, float]:
    """
    Quantifies the systematic downward bias of naive sales forecasting.
    """
    naive_mean = float(np.mean(naive_forecasts))
    aware_mean = float(np.mean(censoring_aware_forecasts))
    
    bias_pct = ((aware_mean - naive_mean) / (aware_mean + 1e-6)) * 100.0
    
    result = {
        "naive_forecast_mean": round(naive_mean, 3),
        "censoring_aware_mean": round(aware_mean, 3),
        "downward_bias_pct": round(bias_pct, 2),
    }

    if ground_truth_demand is not None:
        true_mean = float(np.mean(ground_truth_demand))
        result["true_demand_mean"] = round(true_mean, 3)
        result["naive_error_vs_true_pct"] = round(((naive_mean - true_mean) / true_mean) * 100, 2)
        result["aware_error_vs_true_pct"] = round(((aware_mean - true_mean) / true_mean) * 100, 2)

    return result
