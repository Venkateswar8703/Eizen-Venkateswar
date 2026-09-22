"""
PuLP MILP Multi-Day Workforce Optimization Solver (Module 06):
Alternative formulation using Mixed-Integer Linear Programming (CBC solver) for benchmark comparison.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pulp


def solve_7day_count_plan_milp(
    df_skus: pd.DataFrame,
    num_days: int = 7,
    num_associates: int = 3,
    daily_budget_min: float = 240.0,
    item_count_time_min: float = 1.6,
    aisle_setup_time_min: float = 4.0,
    hourly_wage: float = 21.0,
    max_solve_time_seconds: float = 30.0,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Solves 7-day workforce schedule using PuLP MILP solver.
    """
    df = df_skus.copy().reset_index(drop=True)
    n_skus = len(df)
    
    aisles = sorted(df["aisle_id"].astype(str).unique())
    aisle_to_id = {a: idx for idx, a in enumerate(aisles)}
    sku_aisle_idx = [aisle_to_id[str(a)] for a in df["aisle_id"]]
    
    days_since_count = df.get("days_since_last_count", pd.Series(0, index=df.index)).values
    is_mandatory = (days_since_count + num_days >= 90)
    
    ev_values = df["expected_count_value_ev"].values
    aisle_penalty_dollars = (aisle_setup_time_min / 60.0) * hourly_wage
    
    prob = pulp.LpProblem("CycleCountSchedule", pulp.LpMaximize)
    
    # Candidate filtering for speed
    candidate_indices = [i for i in range(n_skus) if ev_values[i] > 0 or is_mandatory[i]]
    
    x = {}
    for i in candidate_indices:
        for k in range(num_associates):
            for d in range(num_days):
                x[i, k, d] = pulp.LpVariable(f"x_{i}_{k}_{d}", cat=pulp.LpBinary)
                
    y = {}
    for a_idx in range(len(aisles)):
        for k in range(num_associates):
            for d in range(num_days):
                y[a_idx, k, d] = pulp.LpVariable(f"y_{a_idx}_{k}_{d}", cat=pulp.LpBinary)
                
    # 1. Aisle linking constraints
    for (i, k, d), var in x.items():
        a_idx = sku_aisle_idx[i]
        prob += var <= y[a_idx, k, d]
        
    # 2. Daily associate budget
    for k in range(num_associates):
        for d in range(num_days):
            sku_vars = [var for (i, kk, dd), var in x.items() if kk == k and dd == d]
            aisle_vars = [y[a_idx, k, d] for a_idx in range(len(aisles))]
            prob += (
                item_count_time_min * pulp.lpSum(sku_vars) +
                aisle_setup_time_min * pulp.lpSum(aisle_vars) <= daily_budget_min
            )
            
    # 3. Frequency / Compliance constraints
    for i in candidate_indices:
        sku_vars = [var for (ii, k, d), var in x.items() if ii == i]
        if is_mandatory[i]:
            prob += pulp.lpSum(sku_vars) == 1
        else:
            prob += pulp.lpSum(sku_vars) <= 1
            
    # 4. Objective
    obj = (
        pulp.lpSum([var * ev_values[i] for (i, k, d), var in x.items()]) -
        pulp.lpSum([var * aisle_penalty_dollars for var in y.values()])
    )
    prob += obj
    
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=int(max_solve_time_seconds))
    prob.solve(solver)
    
    status_str = pulp.LpStatus[prob.status]
    plan_rows = []
    
    if prob.status in [pulp.constants.LpStatusOptimal, 1]:
        for (i, k, d), var in x.items():
            if var.varValue and var.varValue > 0.5:
                sku_row = df.iloc[i]
                plan_rows.append({
                    "day_index": d + 1,
                    "associate_id": f"Associate_{k+1:02d}",
                    "sku_id": sku_row["sku_id"],
                    "aisle_id": sku_row["aisle_id"],
                    "category": sku_row["category"],
                    "item_count_min": item_count_time_min,
                    "gap_probability": sku_row.get("predicted_gap_prob", 0.0),
                    "expected_count_value_ev": sku_row["expected_count_value_ev"],
                    "is_compliance_mandate": bool(is_mandatory[i]),
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
        "solver_status": status_str,
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
