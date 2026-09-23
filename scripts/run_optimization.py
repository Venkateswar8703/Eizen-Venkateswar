"""
Runner Script for Module 06: Workforce Optimization.
Produces 7-day count schedule using OR-Tools CP-SAT and runs policy benchmarks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.optimization.cpsat_solver import solve_7day_count_plan_cpsat
from src.optimization.benchmarks import run_all_optimization_benchmarks


def main():
    print("=" * 60)
    print("MODULE 06: WORKFORCE OPTIMIZATION EXECUTION")
    print("=" * 60)
    
    results_dir = Path("data/results")
    processed_dir = Path("data/processed")
    results_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    ev_path = results_dir / "valuation_ev_scored.parquet"
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing valuation results at {ev_path}. Run Module 05 first.")
        
    df_ev = pd.read_parquet(ev_path)
    day_col = "day_index" if "day_index" in df_ev.columns else "day"
    max_day = df_ev[day_col].max()
    df_latest = df_ev[df_ev[day_col] == max_day].copy().reset_index(drop=True)
    print(f"Loaded {len(df_latest)} SKUs scored with EV for Day {max_day}.")
    
    # NOTE: The assignment specifies 240 usable labor minutes TOTAL per day across 3 associates.
    # The CP-SAT solver enforces the budget PER ASSOCIATE, so per-associate budget = 240 / 3 = 80 min.
    TOTAL_DAILY_MINUTES = 240.0
    NUM_ASSOCIATES = 3
    PER_ASSOCIATE_BUDGET = TOTAL_DAILY_MINUTES / NUM_ASSOCIATES  # = 80 min / associate
    print(f"\nSolving 7-day schedule with OR-Tools CP-SAT ({NUM_ASSOCIATES} associates, {TOTAL_DAILY_MINUTES:.0f} min/day TOTAL = {PER_ASSOCIATE_BUDGET:.0f} min/associate, 4.0 min aisle setup)...")
    plan_cpsat, metrics_cpsat = solve_7day_count_plan_cpsat(
        df_latest,
        num_days=7,
        num_associates=NUM_ASSOCIATES,
        daily_budget_min=PER_ASSOCIATE_BUDGET,
        item_count_time_min=1.6,
        aisle_setup_time_min=4.0,
        hourly_wage=21.0,
        max_solve_time_seconds=20.0,
    )
    
    # Export 7-day plan
    plan_path = processed_dir / "count_plan_7day.parquet"
    plan_cpsat.to_parquet(plan_path, index=False)
    print(f"Exported 7-day count schedule to {plan_path} with {len(plan_cpsat)} scheduled counts.")
    
    print("\nCP-SAT Schedule Summary:")
    for k, v in metrics_cpsat.items():
        print(f"  {k}: {v}")
        
    print("\nSample Daily Associate Allocations (Day 1):")
    day1 = plan_cpsat[plan_cpsat["day_index"] == 1]
    for assoc in day1["associate_id"].unique():
        assoc_rows = day1[day1["associate_id"] == assoc]
        aisles = assoc_rows["aisle_id"].unique()
        print(f"  {assoc}: {len(assoc_rows)} SKUs across {len(aisles)} aisles ({', '.join(str(a) for a in aisles[:4])}...) | EV=${assoc_rows['expected_count_value_ev'].sum():.2f}")
        
    # 2. Run Policy Benchmarks
    print("\nRunning Optimization Policy Benchmarks (Random vs ABC vs Top-K Prob vs Greedy vs MILP vs CP-SAT)...")
    df_bench, plans = run_all_optimization_benchmarks(
        df_latest,
        num_days=7,
        num_associates=NUM_ASSOCIATES,
        daily_budget_min=PER_ASSOCIATE_BUDGET,
    )
    
    bench_path = results_dir / "optimization_benchmarks.parquet"
    df_bench.to_parquet(bench_path, index=False)
    print(f"Saved benchmark results to {bench_path}.")
    
    print("\n" + "=" * 80)
    print("POLICY BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(df_bench.to_string(index=False))
    print("=" * 80)
    
    print("\nMODULE 06 COMPLETE")


if __name__ == "__main__":
    main()
