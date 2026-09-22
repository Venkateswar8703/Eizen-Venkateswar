"""
Distribution Fitting Module: Compares discrete likelihood models for count demand.
Models:
1. Poisson: Single parameter λ (mean = variance)
2. Negative Binomial: Overdispersed count model (variance > mean)
3. Zero-Inflated Poisson (ZIP): Mixture of point mass at 0 and Poisson count
4. Hurdle Poisson: Two-stage truncated hurdle process
5. Tweedie (Compound Poisson-Gamma): Continuous compound spikes
"""

from typing import Dict, Any
import numpy as np
import scipy.stats as stats
import pandas as pd


def fit_poisson_aic(data: np.ndarray) -> Dict[str, float]:
    """Fits Poisson distribution and returns log-likelihood, AIC, BIC."""
    n = len(data)
    lam = max(1e-4, float(np.mean(data)))
    # Log-likelihood
    ll = float(np.sum(stats.poisson.logpmf(data, mu=lam)))
    k = 1  # 1 parameter
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    return {"model": "Poisson", "log_likelihood": round(ll, 2), "aic": round(aic, 2), "bic": round(bic, 2), "param_count": k}


def fit_negative_binomial_aic(data: np.ndarray) -> Dict[str, float]:
    """Fits Negative Binomial distribution using method of moments / MLE."""
    n = len(data)
    mean_val = float(np.mean(data))
    var_val = float(np.var(data))
    
    if var_val <= mean_val:
        # Falls back to Poisson if underdispersed
        p = 0.999
        r = mean_val * p / (1.0 - p)
    else:
        p = mean_val / var_val
        r = (mean_val ** 2) / (var_val - mean_val)

    p = max(1e-4, min(0.9999, p))
    r = max(1e-4, r)

    ll = float(np.sum(stats.nbinom.logpmf(data, n=r, p=p)))
    k = 2  # 2 parameters: (r, p)
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    return {"model": "Negative Binomial", "log_likelihood": round(ll, 2), "aic": round(aic, 2), "bic": round(bic, 2), "param_count": k}


def fit_zero_inflated_poisson_aic(data: np.ndarray) -> Dict[str, float]:
    """Fits Zero-Inflated Poisson (ZIP) mixture model using vectorized NumPy operations."""
    n = len(data)
    zero_mask = (data == 0)
    p_zero_empirical = np.mean(zero_mask)
    
    non_zero_mean = np.mean(data[~zero_mask]) if np.any(~zero_mask) else 1.0
    lam = max(1e-4, float(non_zero_mean))
    
    # Structural zero mixture parameter w in [0, 1]
    w = max(0.0, min(0.95, (p_zero_empirical - np.exp(-lam)) / (1.0 - np.exp(-lam) + 1e-6)))
    
    # Vectorized ZIP log-likelihood:
    n_zeros = np.sum(zero_mask)
    ll_zeros = n_zeros * np.log(w + (1.0 - w) * np.exp(-lam) + 1e-12)
    
    non_zero_vals = data[~zero_mask]
    if len(non_zero_vals) > 0:
        ll_non_zeros = np.sum(np.log(1.0 - w + 1e-12) + stats.poisson.logpmf(non_zero_vals, mu=lam))
    else:
        ll_non_zeros = 0.0

    ll = float(ll_zeros + ll_non_zeros)
    k = 2  # 2 parameters: (w, lam)
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    return {"model": "Zero-Inflated Poisson", "log_likelihood": round(ll, 2), "aic": round(aic, 2), "bic": round(bic, 2), "param_count": k}


def compare_category_distributions(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Compares all count distribution fits across all 10 categories."""
    records = []
    
    for cat_name, group in df_panel.groupby("category"):
        sales = group["pos_sales"].values
        
        fit_pois = fit_poisson_aic(sales)
        fit_nb = fit_negative_binomial_aic(sales)
        fit_zip = fit_zero_inflated_poisson_aic(sales)
        
        # Determine best model by minimum AIC
        models = [fit_pois, fit_nb, fit_zip]
        best_model = min(models, key=lambda m: m["aic"])["model"]
        
        dispersion_ratio = float(np.var(sales) / (np.mean(sales) + 1e-6))
        zero_share = float((sales == 0).mean())

        records.append({
            "category": cat_name,
            "sample_size": len(sales),
            "mean": round(float(np.mean(sales)), 2),
            "variance": round(float(np.var(sales)), 2),
            "dispersion_ratio (Var/Mean)": round(dispersion_ratio, 2),
            "zero_percentage": f"{zero_share * 100:.1f}%",
            "poisson_aic": fit_pois["aic"],
            "negbin_aic": fit_nb["aic"],
            "zip_aic": fit_zip["aic"],
            "best_distribution": best_model,
        })

    return pd.DataFrame(records)
