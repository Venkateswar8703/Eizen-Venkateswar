"""
Target Generation Module for Inventory Gap Prediction:
Defines the primary binary classification target (Material Gap / Phantom Stockout)
and secondary continuous regression target (Gap Magnitude).
"""

from typing import Tuple
import numpy as np
import pandas as pd


def create_gap_targets(
    df_panel: pd.DataFrame,
    gap_threshold: int = 1,
) -> pd.DataFrame:
    """
    Creates leakage-free target variables:
    - target_material_gap: I(|G(i,t)| >= gap_threshold)
    - target_phantom_oos: I(SOH(i,t) > 0 and TrueOnHand(i,t) == 0)
    - target_gap_magnitude: G(i,t) (Signed integer gap)
    """
    df = df_panel.copy()
    
    # Binary classification target: material discrepancy
    df["target_material_gap"] = (df["inventory_gap"].abs() >= gap_threshold).astype(int)
    
    # Critical phantom out-of-stock target (primary business value driver)
    df["target_phantom_oos"] = df["is_phantom_stockout"].astype(int)
    
    # Continuous regression target: signed discrepancy magnitude
    df["target_gap_magnitude"] = df["inventory_gap"].astype(float)
    
    return df
