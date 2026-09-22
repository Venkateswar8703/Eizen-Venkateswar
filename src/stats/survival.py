"""
Survival Analysis & Discrete-Time Hazard Modeling Module for Gap Onset.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_category_hazard_rates(df_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Computes discrete-time daily hazard rate h(t) = P(Gap starts at day t | No gap prior to t).
    """
    records = []
    
    for cat_name, group in df_panel.groupby("category"):
        # Identify gap onsets: transition from gap == 0 to gap > 0
        group = group.sort_values(["sku_id", "day_index"])
        group["prev_gap"] = group.groupby("sku_id")["inventory_gap"].shift(1).fillna(0)
        
        # Risk set: days where previous gap was 0
        risk_set = group[group["prev_gap"] == 0]
        # Gap onset event: gap > 0 today
        events = risk_set[risk_set["inventory_gap"] > 0]
        
        n_at_risk = len(risk_set)
        n_events = len(events)
        
        daily_hazard = n_events / n_at_risk if n_at_risk > 0 else 0.0
        # Expected median time to gap onset T_med = -ln(0.5) / daily_hazard
        median_days_to_gap = round(np.log(2.0) / (daily_hazard + 1e-8), 1)

        records.append({
            "category": cat_name,
            "days_at_risk": n_at_risk,
            "gap_onset_events": n_events,
            "daily_hazard_rate": round(daily_hazard, 4),
            "weekly_hazard_rate": round(1.0 - (1.0 - daily_hazard) ** 7, 4),
            "median_days_to_gap": median_days_to_gap,
            "recommended_audit_cycle_days": min(90, max(7, int(median_days_to_gap * 0.5))),
        })

    return pd.DataFrame(records).sort_values("daily_hazard_rate", ascending=False)
