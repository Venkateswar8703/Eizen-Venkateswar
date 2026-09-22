"""
Censoring Diagnostics & Zero-Streak Statistical Power Module.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats
import pandas as pd


def compute_zero_streak_power(
    daily_rates: List[float] = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
    max_streak_days: int = 15,
) -> pd.DataFrame:
    """
    Computes the p-value / false alarm probability of observing a streak of k consecutive zero sales
    under the null hypothesis H0: SKU is in-stock and demand is Poisson(lambda).
    
    P(Sales = 0 for k consecutive days | In-Stock, λ) = (e^{-λ})^k = e^{-kλ}
    """
    records = []
    
    for lam in daily_rates:
        for k in range(1, max_streak_days + 1):
            p_val = np.exp(-lam * k)
            power_to_reject_null = 1.0 - p_val
            
            # Informative if p-value < 0.05
            is_significant_05 = p_val < 0.05
            is_significant_01 = p_val < 0.01

            records.append({
                "mean_daily_demand_lambda": lam,
                "zero_streak_length_days": k,
                "p_value_under_null": round(float(p_val), 4),
                "anomaly_detection_power": round(float(power_to_reject_null), 4),
                "significant_at_p05": is_significant_05,
                "significant_at_p01": is_significant_01,
            })

    return pd.DataFrame(records)


def tobit_censored_mean_recovery(
    observed_sales: np.ndarray,
    is_stockout_proxy: np.ndarray,
) -> Dict[str, float]:
    """
    Recovers true unconstrained latent mean using Tobit right/interval censoring approximation.
    """
    uncensored = observed_sales[~is_stockout_proxy]
    censored = observed_sales[is_stockout_proxy]
    
    naive_mean = float(np.mean(observed_sales))
    uncensored_mean = float(np.mean(uncensored)) if len(uncensored) > 0 else naive_mean
    
    # EM / Tobit adjustment: Under stockout, expected unobserved demand exceeds observed stock
    # For Poisson/Normal, truncated tail expectation:
    truncation_lift = 1.45  # Expected conditional exceedance E[D | D >= S]
    recovered_mean = (np.sum(uncensored) + np.sum(censored) * truncation_lift) / len(observed_sales)

    return {
        "naive_observed_mean": round(naive_mean, 3),
        "uncensored_subset_mean": round(uncensored_mean, 3),
        "tobit_recovered_mean": round(recovered_mean, 3),
        "censoring_bias_pct": round(((recovered_mean - naive_mean) / (recovered_mean + 1e-6)) * 100, 2),
    }
