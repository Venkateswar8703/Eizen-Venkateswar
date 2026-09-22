"""
Expected Economic Value Engine for Cycle Audits (Module 05):
Translates predicted gap probabilities into net dollar value recovered:
EV(i, d) = P(gap) * Rec(i) * tau(i) * [mu_D(i) * (margin + basket_loss - sub_salvage)] - LaborCost - FP_Cost
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


# Category-specific recovery rates (fraction of discrepancy resolved by audit)
CATEGORY_RECOVERY_RATES = {
    "Fresh Produce": 0.90,
    "Dairy & Eggs": 0.88,
    "Meat & Seafood": 0.85,
    "Bakery": 0.90,
    "Frozen": 0.85,
    "Packaged Grocery": 0.85,
    "Beverages": 0.82,
    "Snacks & Confectionery": 0.80,
    "Household & Cleaning": 0.78,
    "Health & Beauty": 0.75, # Lower due to untraceable shoplifting
}


def compute_expected_value_of_count(
    df_predictions: pd.DataFrame,
    hourly_wage: float = 21.0,
    basket_abandonment_prob: float = 0.12,
    avg_basket_margin: float = 18.0,
    substitution_rate: float = 0.45,
    false_alarm_penalty: float = 0.50,
) -> pd.DataFrame:
    """
    Computes expected net dollar value EV(i,d) for every SKU-day observation.
    """
    df = df_predictions.copy()
    
    # 1. Base recovery rate Rec(i)
    rec_rates = df["category"].map(CATEGORY_RECOVERY_RATES).fillna(0.85).values
    
    # 2. Persistence / Remaining days of impact until natural self-correction
    # Fast movers self-correct at next truck in 2-3 days; slow movers persist 20-30 days
    demands = np.maximum(0.05, df["base_demand_velocity"].values)
    soh = np.maximum(0, df["system_on_hand"].values)
    days_to_self_correct = np.clip(soh / demands, 2.0, 30.0)
    
    # 3. Unit margins and economic impact
    selling_price = df["selling_price"].values
    # Margin rate approx 35%
    unit_margin = np.maximum(0.50, selling_price * 0.35)
    
    # Net unit value recovered per day of averted phantom stockout:
    # Direct Margin + Basket Abandonment Loss - Substitution Salvage
    basket_loss = basket_abandonment_prob * avg_basket_margin # $2.16
    substitution_salvage = substitution_rate * unit_margin
    net_daily_unit_loss = unit_margin + basket_loss - substitution_salvage
    
    # Total expected gross dollar loss over persistence horizon
    gross_loss_per_gap = demands * net_daily_unit_loss * days_to_self_correct
    
    # 4. Labor and Audit Cost
    count_duration = df.get("count_duration_min", 1.6).values if "count_duration_min" in df.columns else 1.6
    labor_cost_per_count = (hourly_wage / 60.0) * count_duration
    
    # 5. Expected Net Economic Value EV(i, d)
    gap_prob = np.clip(df["predicted_gap_prob"].values, 0.0, 1.0)
    
    expected_gross_recovery = gap_prob * rec_rates * gross_loss_per_gap
    expected_fp_cost = (1.0 - gap_prob) * false_alarm_penalty
    
    expected_net_value = expected_gross_recovery - labor_cost_per_count - expected_fp_cost
    
    df["recovery_rate"] = np.round(rec_rates, 2)
    df["persistence_days"] = np.round(days_to_self_correct, 1)
    df["unit_margin"] = np.round(unit_margin, 2)
    df["gross_loss_at_risk"] = np.round(gross_loss_per_gap, 2)
    df["count_labor_cost"] = np.round(labor_cost_per_count, 2)
    df["expected_count_value_ev"] = np.round(expected_net_value, 2)
    
    return df
