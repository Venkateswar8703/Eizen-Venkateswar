"""
Statistical Intermittency & Categorization Module.
Implements Syntetos-Boylan ADI (Average Demand Interval) and CV² (Squared Coefficient of Variation)
quadrant categorization for retail time-series.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_adi_cv2(series: np.ndarray) -> Tuple[float, float, str]:
    """
    Computes ADI (Average Demand Interval) and CV² (Squared Coefficient of Variation).
    
    Categories (Syntetos et al., 2005):
    - Smooth:       ADI < 1.32 and CV² < 0.49
    - Erratic:      ADI < 1.32 and CV² >= 0.49
    - Intermittent: ADI >= 1.32 and CV² < 0.49
    - Lumpy:        ADI >= 1.32 and CV² >= 0.49
    """
    non_zero_indices = np.where(series > 0)[0]
    
    if len(non_zero_indices) < 2:
        return 999.0, 999.0, "Lumpy"

    # Average Demand Interval (average inter-arrival time between non-zero sales)
    intervals = np.diff(non_zero_indices)
    adi = float(np.mean(intervals))

    # Squared Coefficient of Variation of non-zero demand quantities
    non_zero_values = series[non_zero_indices]
    mean_val = np.mean(non_zero_values)
    std_val = np.std(non_zero_values)
    cv2 = float((std_val / (mean_val + 1e-6)) ** 2)

    # Classification boundaries (Syntetos, Boylan, Croston standard)
    if adi < 1.32:
        category = "Smooth" if cv2 < 0.49 else "Erratic"
    else:
        category = "Intermittent" if cv2 < 0.49 else "Lumpy"

    return round(adi, 3), round(cv2, 3), category


def categorize_store_skus(df_panel: pd.DataFrame) -> pd.DataFrame:
    """Computes intermittency classification for all SKUs in the store panel."""
    records = []
    
    for sku_id, group in df_panel.groupby("sku_id"):
        sales = group["pos_sales"].values
        category = group["category"].iloc[0]
        aisle = group["aisle_id"].iloc[0]
        
        adi, cv2, quadrant = compute_adi_cv2(sales)
        zero_pct = float((sales == 0).mean() * 100)
        mean_sales = float(np.mean(sales))
        
        records.append({
            "sku_id": sku_id,
            "category": category,
            "aisle_id": aisle,
            "mean_daily_sales": round(mean_sales, 3),
            "zero_sales_pct": round(zero_pct, 1),
            "adi": adi,
            "cv2": cv2,
            "intermittency_quadrant": quadrant,
        })

    return pd.DataFrame(records)
