"""
Optimization Policy Benchmark Engine (Module 06):
Compares Random, ABC, Top-K Prob, Top-K EV, Greedy Heuristic, MILP, and CP-SAT policies.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.optimization.cpsat_solver import solve_7day_count_plan_cpsat
from src.optimization.milp_solver import solve_7day_count_plan_milp
from src.optimization.greedy import solve_7day_count_plan_greedy


def evaluate_policy_schedule(
    df_plan: pd.DataFrame,
    hourly_wage: float = 21.0,
    item_count_time_min: float = 1.6,
    aisle_setup_time_min: float = 4.0,
    total_budget_min: float = 7 * 3 * 80.0,  # 240 total/day ÷ 3 associates = 80/associate × 3 × 7 days
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics on a generated schedule.
    """
    if df_plan.empty:
        return {
            "skus_counted": 0,
            "distinct_aisles": 0,
            "gross_ev": 0.0,
            "aisle_penalty": 0.0,
            "net_ev": 0.0,
            "time_used_min": 0.0,
            "utilization_pct": 0.0,
            "ev_per_hour": 0.0,
        }
        
    distinct_trips = df_plan.groupby(["day_index", "associate_id"])["aisle_id"].nunique().sum()
    gross_ev = df_plan["expected_count_value_ev"].sum()
    aisle_penalty = (distinct_trips * aisle_setup_time_min / 60.0) * hourly_wage
    net_ev = gross_ev - aisle_penalty
    total_time = len(df_plan) * item_count_time_min + distinct_trips * aisle_setup_time_min
    util_pct = (total_time / total_budget_min) * 100.0 if total_budget_min > 0 else 0.0
    ev_per_hour = (net_ev / (total_time / 60.0)) if total_time > 0 else 0.0
    
    return {
        "skus_counted": len(df_plan),
        "distinct_aisle_trips": int(distinct_trips),
        "gross_ev": round(gross_ev, 2),
        "aisle_penalty_dollars": round(aisle_penalty, 2),
        "net_ev": round(net_ev, 2),
        "time_used_min": round(total_time, 1),
        "labor_utilization_pct": round(util_pct, 1),
        "value_per_labor_hour": round(ev_per_hour, 2),
    }


def run_all_optimization_benchmarks(
    df_skus: pd.DataFrame,
    num_days: int = 7,
    num_associates: int = 3,
    daily_budget_min: float = 240.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs full benchmark suite across all policies.
    """
    df = df_skus.copy()
    plans = {}
    benchmark_records = []
    
    # 1. Random Policy
    df_rand = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    plan_rand, _ = solve_7day_count_plan_greedy(df_rand, num_days, num_associates, daily_budget_min)
    plans["Random"] = plan_rand
    metrics_rand = evaluate_policy_schedule(plan_rand)
    metrics_rand["policy"] = "1. Random Selection"
    benchmark_records.append(metrics_rand)
    
    # 2. ABC Classification (Ranked by velocity * price)
    df_abc = df.copy()
    df_abc["revenue_proxy"] = df_abc["base_demand_velocity"] * df_abc["selling_price"]
    df_abc = df_abc.sort_values(by="revenue_proxy", ascending=False).reset_index(drop=True)
    plan_abc, _ = solve_7day_count_plan_greedy(df_abc, num_days, num_associates, daily_budget_min)
    plans["ABC"] = plan_abc
    metrics_abc = evaluate_policy_schedule(plan_abc)
    metrics_abc["policy"] = "2. ABC / Velocity"
    benchmark_records.append(metrics_abc)
    
    # 3. Top-K Gap Probability (Pure ML prob, ignoring margin/cost)
    df_prob = df.sort_values(by="predicted_gap_prob", ascending=False).reset_index(drop=True)
    plan_prob, _ = solve_7day_count_plan_greedy(df_prob, num_days, num_associates, daily_budget_min)
    plans["TopK_Prob"] = plan_prob
    metrics_prob = evaluate_policy_schedule(plan_prob)
    metrics_prob["policy"] = "3. Top-K Gap Probability"
    benchmark_records.append(metrics_prob)
    
    # 4. Greedy Heuristic (With Aisle Clustering)
    plan_greedy, _ = solve_7day_count_plan_greedy(df, num_days, num_associates, daily_budget_min)
    plans["Greedy"] = plan_greedy
    metrics_greedy = evaluate_policy_schedule(plan_greedy)
    metrics_greedy["policy"] = "4. Greedy Submodular"
    benchmark_records.append(metrics_greedy)
    
    # 5. PuLP MILP Solver
    try:
        plan_milp, m_milp = solve_7day_count_plan_milp(df, num_days, num_associates, daily_budget_min, max_solve_time_seconds=15.0)
        plans["MILP"] = plan_milp
        metrics_milp = evaluate_policy_schedule(plan_milp)
        metrics_milp["policy"] = "5. PuLP MILP"
        benchmark_records.append(metrics_milp)
    except Exception as e:
        print(f"MILP solve note: {e}")
        
    # 6. OR-Tools CP-SAT (Primary)
    plan_cpsat, m_cpsat = solve_7day_count_plan_cpsat(df, num_days, num_associates, daily_budget_min, max_solve_time_seconds=20.0)
    plans["CP_SAT"] = plan_cpsat
    metrics_cpsat = evaluate_policy_schedule(plan_cpsat)
    metrics_cpsat["policy"] = "6. OR-Tools CP-SAT (Primary)"
    benchmark_records.append(metrics_cpsat)
    
    df_bench = pd.DataFrame(benchmark_records)
    # Put policy column first
    cols = ["policy"] + [c for c in df_bench.columns if c != "policy"]
    df_bench = df_bench[cols]
    
    return df_bench, plans
