"""
Submodular Formulation and Greedy Selection for Cycle Audits (Module 05):

Formulates the SKU selection under budget and aisle routing as a submodular maximization problem:
f(S) = sum_{i in S} EV(i) - sum_{a in A(S)} C_aisle_setup(a)

Greedy algorithm provides a (1 - 1/e) approx under submodular maximization with knapsack constraints.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def greedy_submodular_count_selection(
    df_ev: pd.DataFrame,
    total_time_budget_min: float = 720.0, # e.g. 3 associates * 240 min = 720 min
    item_count_time_min: float = 1.6,
    aisle_setup_time_min: float = 4.0,
    hourly_wage: float = 21.0,
    mandatory_skus: Set[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Greedy selection of SKUs that maximizes net economic value subject to labor time budget
    accounting for shared aisle setup costs (submodular marginal gains).
    
    Parameters:
    - df_ev: DataFrame containing SKU, aisle, expected_count_value_ev, etc.
    - total_time_budget_min: Total labor minutes available.
    - item_count_time_min: Minutes required to scan/count one SKU.
    - aisle_setup_time_min: Minutes setup overhead per distinct aisle entered.
    - hourly_wage: Labor cost $/hour.
    - mandatory_skus: SKUs that MUST be included (e.g. 90-day compliance floor).
    
    Returns:
    - selected_df: Chosen SKUs with audit order and allocated time.
    - summary: Metrics on objective value, utilization, and aisle count.
    """
    df = df_ev.copy()
    if mandatory_skus is None:
        mandatory_skus = set()
        
    selected_indices: List[int] = []
    visited_aisles: Set[str] = set()
    used_time_min: float = 0.0
    
    # Filter to candidate indices with positive EV (or mandatory)
    mand_set = set(mandatory_skus)
    cand_df = df[(df["expected_count_value_ev"] > 0) | (df["sku_id"].isin(mand_set))].copy()
    
    # Pre-select mandatory compliance SKUs first
    if mand_set:
        mand_mask = cand_df["sku_id"].isin(mand_set)
        for idx in cand_df[mand_mask].index:
            sku_aisle = str(cand_df.loc[idx, "aisle_id"])
            marginal_time = item_count_time_min
            if sku_aisle not in visited_aisles:
                marginal_time += aisle_setup_time_min
                visited_aisles.add(sku_aisle)
            selected_indices.append(idx)
            used_time_min += marginal_time

    remaining_cand_df = cand_df.drop(index=selected_indices, errors="ignore")
    
    # Fast vectorized greedy iteration
    while not remaining_cand_df.empty and used_time_min < total_time_budget_min:
        aisles = remaining_cand_df["aisle_id"].astype(str).values
        evs = remaining_cand_df["expected_count_value_ev"].values
        
        # In-aisle check
        is_visited = np.array([a in visited_aisles for a in aisles])
        marginal_times = np.where(is_visited, item_count_time_min, item_count_time_min + aisle_setup_time_min)
        setup_penalties = np.where(is_visited, 0.0, (aisle_setup_time_min / 60.0) * hourly_wage)
        
        net_marginal_gains = evs - setup_penalties
        
        valid_mask = (used_time_min + marginal_times <= total_time_budget_min) & (net_marginal_gains > 0)
        if not np.any(valid_mask):
            break
            
        ratios = np.where(valid_mask, net_marginal_gains / marginal_times, -1e9)
        best_pos = np.argmax(ratios)
        
        best_idx = remaining_cand_df.index[best_pos]
        best_time = marginal_times[best_pos]
        best_aisle = aisles[best_pos]
        
        selected_indices.append(best_idx)
        visited_aisles.add(best_aisle)
        used_time_min += best_time
        remaining_cand_df = remaining_cand_df.drop(index=best_idx)
        
    selected_df = df.loc[selected_indices].copy()
    total_ev = selected_df["expected_count_value_ev"].sum()
    total_setup_overhead_min = len(visited_aisles) * aisle_setup_time_min
    
    summary = {
        "num_skus_selected": len(selected_df),
        "distinct_aisles_visited": len(visited_aisles),
        "total_time_used_min": used_time_min,
        "total_time_budget_min": total_time_budget_min,
        "time_utilization_pct": (used_time_min / total_time_budget_min) * 100.0 if total_time_budget_min > 0 else 0.0,
        "item_count_time_min": len(selected_df) * item_count_time_min,
        "aisle_setup_overhead_min": total_setup_overhead_min,
        "total_expected_economic_value": total_ev,
        "net_value_after_aisle_overhead": total_ev - (total_setup_overhead_min / 60.0) * hourly_wage,
    }
    
    return selected_df, summary
