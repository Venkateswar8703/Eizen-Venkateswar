"""
Sensitivity Analysis Engine for Value of Count (Module 05):

Explores parameter variations across:
- Recovery rate Rec(i)
- Hourly wage
- Gap probability threshold
- Persistence multiplier
- Aisle setup penalty
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from src.valuation.value_model import compute_expected_value_of_count
from src.valuation.submodular import greedy_submodular_count_selection


def run_valuation_sensitivity_sweep(
    df_predictions: pd.DataFrame,
    wage_levels: List[float] = [15.0, 21.0, 28.0, 35.0],
    recovery_multipliers: List[float] = [0.5, 0.75, 1.0, 1.25],
    persistence_multipliers: List[float] = [0.5, 1.0, 1.5, 2.0],
    aisle_setup_times: List[float] = [0.0, 2.0, 4.0, 8.0],
) -> pd.DataFrame:
    """
    Evaluates net expected value and selected SKU counts under parameter sweeps.
    """
    records = []
    
    # 1. Wage Sweep
    for wage in wage_levels:
        df_val = compute_expected_value_of_count(df_predictions, hourly_wage=wage)
        _, summary = greedy_submodular_count_selection(
            df_val, total_time_budget_min=720.0, hourly_wage=wage, aisle_setup_time_min=4.0
        )
        records.append({
            "parameter": "hourly_wage",
            "param_value": wage,
            "unit": "$/hr",
            "num_skus_selected": summary["num_skus_selected"],
            "total_ev_gross": summary["total_expected_economic_value"],
            "net_value": summary["net_value_after_aisle_overhead"],
            "time_utilization_pct": summary["time_utilization_pct"],
        })
        
    # 2. Recovery Rate Sweep
    for rec_mult in recovery_multipliers:
        df_copy = df_predictions.copy()
        df_val = compute_expected_value_of_count(df_copy, hourly_wage=21.0)
        df_val["expected_count_value_ev"] = np.maximum(
            0.0, df_val["expected_count_value_ev"] * rec_mult
        )
        _, summary = greedy_submodular_count_selection(
            df_val, total_time_budget_min=720.0, hourly_wage=21.0, aisle_setup_time_min=4.0
        )
        records.append({
            "parameter": "recovery_rate_multiplier",
            "param_value": rec_mult,
            "unit": "scale",
            "num_skus_selected": summary["num_skus_selected"],
            "total_ev_gross": summary["total_expected_economic_value"],
            "net_value": summary["net_value_after_aisle_overhead"],
            "time_utilization_pct": summary["time_utilization_pct"],
        })

    # 3. Persistence Multiplier Sweep
    for pers_mult in persistence_multipliers:
        df_val = compute_expected_value_of_count(df_predictions, hourly_wage=21.0)
        # Scaled gross loss impact
        df_val["expected_count_value_ev"] = (
            df_val["predicted_gap_prob"] * df_val["recovery_rate"] * (df_val["gross_loss_at_risk"] * pers_mult)
            - df_val["count_labor_cost"]
        )
        _, summary = greedy_submodular_count_selection(
            df_val, total_time_budget_min=720.0, hourly_wage=21.0, aisle_setup_time_min=4.0
        )
        records.append({
            "parameter": "persistence_multiplier",
            "param_value": pers_mult,
            "unit": "scale",
            "num_skus_selected": summary["num_skus_selected"],
            "total_ev_gross": summary["total_expected_economic_value"],
            "net_value": summary["net_value_after_aisle_overhead"],
            "time_utilization_pct": summary["time_utilization_pct"],
        })
        
    # 4. Aisle Setup Penalty Sweep
    for setup_time in aisle_setup_times:
        df_val = compute_expected_value_of_count(df_predictions, hourly_wage=21.0)
        _, summary = greedy_submodular_count_selection(
            df_val, total_time_budget_min=720.0, hourly_wage=21.0, aisle_setup_time_min=setup_time
        )
        records.append({
            "parameter": "aisle_setup_minutes",
            "param_value": setup_time,
            "unit": "minutes",
            "num_skus_selected": summary["num_skus_selected"],
            "total_ev_gross": summary["total_expected_economic_value"],
            "net_value": summary["net_value_after_aisle_overhead"],
            "time_utilization_pct": summary["time_utilization_pct"],
        })
        
    return pd.DataFrame(records)
