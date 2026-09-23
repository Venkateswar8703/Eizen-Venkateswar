"""
Greedy Multi-Day Workforce Optimization Scheduler (Module 06):
Fast heuristic baseline that allocates SKUs greedily across associates and days with aisle clustering.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def solve_7day_count_plan_greedy(
    df_skus: pd.DataFrame,
    num_days: int = 7,
    num_associates: int = 3,
    daily_budget_min: float = 240.0,
    item_count_time_min: float = 1.6,
    aisle_setup_time_min: float = 4.0,
    hourly_wage: float = 21.0,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Greedy heuristic scheduler for 7-day multi-associate cycle count planning.

    Parameters
    ----------
    daily_budget_min : float
        Usable labor minutes **per associate per day**. When the total daily store
        budget is 240 minutes across 3 associates, pass 240 / 3 = 80 here.
    """
    df = df_skus.copy().reset_index(drop=True)
    
    days_since_count = df.get("days_since_last_count", pd.Series(0, index=df.index)).values
    is_mandatory = (days_since_count + num_days >= 90)
    
    # Priority sorting: compliance first, then net EV descending
    df["is_mandatory"] = is_mandatory
    # Sort candidates
    df_sorted = df.sort_values(by=["is_mandatory", "expected_count_value_ev"], ascending=[False, False]).copy()
    
    plan_rows = []
    scheduled_skus: Set[str] = set()
    
    # Track time and aisles per associate per day
    # (day, assoc) -> (used_time, set_of_aisles)
    assoc_state = {
        (d, k): {"time": 0.0, "aisles": set()}
        for d in range(1, num_days + 1)
        for k in range(1, num_associates + 1)
    }
    
    for _, row in df_sorted.iterrows():
        sku = row["sku_id"]
        if sku in scheduled_skus:
            continue
        ev = row["expected_count_value_ev"]
        if ev <= 0 and not row["is_mandatory"]:
            continue
            
        aisle = str(row["aisle_id"])
        
        # Best associate-day slot (prefer associate already in same aisle to save setup cost)
        best_slot = None
        best_marginal_cost = 1e9
        
        for (d, k), state in assoc_state.items():
            marginal_time = item_count_time_min if aisle in state["aisles"] else item_count_time_min + aisle_setup_time_min
            if state["time"] + marginal_time <= daily_budget_min:
                # Rank by marginal time (same aisle has 1.6 min vs 5.6 min)
                if marginal_time < best_marginal_cost:
                    best_marginal_cost = marginal_time
                    best_slot = (d, k)
                    
        if best_slot is not None:
            d, k = best_slot
            state = assoc_state[(d, k)]
            state["time"] += best_marginal_cost
            state["aisles"].add(aisle)
            scheduled_skus.add(sku)
            
            plan_rows.append({
                "day_index": d,
                "associate_id": f"Associate_{k:02d}",
                "sku_id": sku,
                "aisle_id": aisle,
                "category": row["category"],
                "item_count_min": item_count_time_min,
                "gap_probability": row.get("predicted_gap_prob", 0.0),
                "expected_count_value_ev": ev,
                "is_compliance_mandate": bool(row["is_mandatory"]),
            })
            
    df_plan = pd.DataFrame(plan_rows)
    if not df_plan.empty:
        df_plan = df_plan.sort_values(by=["day_index", "associate_id", "aisle_id"]).reset_index(drop=True)
        
    total_ev = df_plan["expected_count_value_ev"].sum() if not df_plan.empty else 0.0
    distinct_aisle_trips = 0
    if not df_plan.empty:
        distinct_aisle_trips = df_plan.groupby(["day_index", "associate_id"])["aisle_id"].nunique().sum()
    aisle_overhead_min = distinct_aisle_trips * aisle_setup_time_min
    item_time_min = len(df_plan) * item_count_time_min
    total_time_min = item_time_min + aisle_overhead_min
    total_budget_min = num_days * num_associates * daily_budget_min
    
    metrics = {
        "solver_status": "GreedyHeuristic",
        "total_skus_scheduled": len(df_plan),
        "distinct_aisle_visits": int(distinct_aisle_trips),
        "gross_expected_value": float(total_ev),
        "aisle_setup_overhead_dollars": float((aisle_overhead_min / 60.0) * hourly_wage),
        "net_economic_value": float(total_ev - (aisle_overhead_min / 60.0) * hourly_wage),
        "total_labor_time_used_min": float(total_time_min),
        "total_labor_budget_min": float(total_budget_min),
        "labor_utilization_pct": float((total_time_min / total_budget_min) * 100.0) if total_budget_min > 0 else 0.0,
        "compliance_mandates_satisfied": int(df_plan["is_compliance_mandate"].sum()) if not df_plan.empty else 0,
    }
    
    return df_plan, metrics
