"""
OR-Tools CP-SAT Multi-Day Workforce Optimization Solver (Module 06):
Schedules cycle audits across 7 days, 3 associates, and 30 aisles.
Enforces 240-min daily associate budget, 4.0-min aisle setup overhead, and 90-day compliance floor.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model


def solve_7day_count_plan_cpsat(
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
    Solves optimal 7-day multi-associate audit schedule using OR-Tools CP-SAT.
    """
    df = df_skus.copy().reset_index(drop=True)
    n_skus = len(df)
    
    # Identify aisles
    aisles = sorted(df["aisle_id"].astype(str).unique())
    aisle_to_id = {a: idx for idx, a in enumerate(aisles)}
    sku_aisle_idx = [aisle_to_id[str(a)] for a in df["aisle_id"]]
    
    # Identify compliance mandates
    days_since_count = df.get("days_since_last_count", pd.Series(0, index=df.index)).values
    # If days_since_last_count + 7 >= 90, it must be counted within this 7-day window
    is_mandatory = (days_since_count + num_days >= 90)
    
    # Integer-scaled parameters for CP-SAT (multiply by 100 to avoid float precision loss)
    SCALE = 100
    item_cost_int = int(round(item_count_time_min * SCALE))
    setup_cost_int = int(round(aisle_setup_time_min * SCALE))
    budget_int = int(round(daily_budget_min * SCALE))
    
    # Setup cost penalty in objective ($ per aisle entry): 4 min / 60 * wage
    aisle_penalty_dollars = (aisle_setup_time_min / 60.0) * hourly_wage
    
    # Scale EV to integer
    ev_values = df["expected_count_value_ev"].values
    ev_int = [int(round(max(0.0, val) * SCALE)) for val in ev_values]
    setup_penalty_int = int(round(aisle_penalty_dollars * SCALE))
    
    model = cp_model.CpModel()
    
    # Decision Variables:
    # x[i, k, d]: binary SKU i assigned to associate k on day d
    x = {}
    for i in range(n_skus):
        # Only create variables for positive EV or mandatory SKUs to keep solver fast
        if ev_values[i] > 0 or is_mandatory[i]:
            for k in range(num_associates):
                for d in range(num_days):
                    x[i, k, d] = model.NewBoolVar(f"x_{i}_{k}_{d}")
                    
    # y[a, k, d]: binary aisle a visited by associate k on day d
    y = {}
    for a_idx in range(len(aisles)):
        for k in range(num_associates):
            for d in range(num_days):
                y[a_idx, k, d] = model.NewBoolVar(f"y_{a_idx}_{k}_{d}")
                
    # 1. Aisle Linking Constraints: x[i, k, d] <= y[aisle(i), k, d]
    for (i, k, d), var in x.items():
        a_idx = sku_aisle_idx[i]
        model.Add(var <= y[a_idx, k, d])
        
    # 2. Daily Labor Capacity per Associate:
    for k in range(num_associates):
        for d in range(num_days):
            sku_vars = [var for (i, kk, dd), var in x.items() if kk == k and dd == d]
            aisle_vars = [y[a_idx, k, d] for a_idx in range(len(aisles))]
            
            total_time_expr = (
                sum(var * item_cost_int for var in sku_vars) +
                sum(avar * setup_cost_int for avar in aisle_vars)
            )
            model.Add(total_time_expr <= budget_int)
            
    # 3. At Most One Count per SKU over 7 days (or Exactly 1 if mandatory):
    for i in range(n_skus):
        sku_vars = [var for (ii, k, d), var in x.items() if ii == i]
        if not sku_vars:
            continue
        if is_mandatory[i]:
            model.Add(sum(sku_vars) == 1)
        else:
            model.Add(sum(sku_vars) <= 1)
            
    # 4. Objective Function: Maximize Total Net EV
    obj_terms = []
    for (i, k, d), var in x.items():
        obj_terms.append(var * ev_int[i])
    for (a_idx, k, d), var in y.items():
        obj_terms.append(var * (-setup_penalty_int))
        
    model.Maximize(sum(obj_terms))
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_solve_time_seconds
    solver.parameters.num_workers = 4
    status = solver.Solve(model)
    
    # Extract Solution
    plan_rows = []
    status_str = solver.StatusName(status)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (i, k, d), var in x.items():
            if solver.Value(var) == 1:
                sku_row = df.iloc[i]
                a_idx = sku_aisle_idx[i]
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
        
    # Calculate operational metrics
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
        "solver_runtime_sec": solver.WallTime(),
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
