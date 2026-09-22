"""
Unit and Integration Tests for Module 06: Workforce Optimization.
"""

import pytest
import numpy as np
import pandas as pd
from src.optimization.cpsat_solver import solve_7day_count_plan_cpsat
from src.optimization.greedy import solve_7day_count_plan_greedy
from src.optimization.benchmarks import run_all_optimization_benchmarks, evaluate_policy_schedule


@pytest.fixture
def mock_skus_for_opt():
    np.random.seed(42)
    categories = ["Dairy & Eggs", "Beverages", "Health & Beauty", "Packaged Grocery"]
    data = []
    for i in range(120):
        # assign 6 aisles
        aisle = f"AISLE_{i % 6:02d}"
        days_since = 95 if i < 5 else np.random.randint(10, 80) # first 5 are mandatory compliance
        data.append({
            "sku_id": f"SKU_{i:04d}",
            "aisle_id": aisle,
            "category": categories[i % len(categories)],
            "selling_price": np.random.uniform(5.0, 40.0),
            "base_demand_velocity": np.random.uniform(0.5, 5.0),
            "predicted_gap_prob": np.random.uniform(0.1, 0.95),
            "expected_count_value_ev": np.random.uniform(-1.0, 50.0),
            "days_since_last_count": days_since,
        })
    return pd.DataFrame(data)


def test_cpsat_solver_constraints(mock_skus_for_opt):
    num_days = 3
    num_associates = 2
    daily_budget = 120.0
    item_time = 1.6
    setup_time = 4.0
    
    plan, metrics = solve_7day_count_plan_cpsat(
        mock_skus_for_opt,
        num_days=num_days,
        num_associates=num_associates,
        daily_budget_min=daily_budget,
        item_count_time_min=item_time,
        aisle_setup_time_min=setup_time,
        max_solve_time_seconds=10.0,
    )
    
    assert not plan.empty
    assert metrics["solver_status"] in ["OPTIMAL", "FEASIBLE"]
    
    # Check no SKU is counted more than once
    assert plan["sku_id"].nunique() == len(plan)
    
    # Check each associate on each day does not exceed daily budget
    for (d, k), group in plan.groupby(["day_index", "associate_id"]):
        n_items = len(group)
        n_aisles = group["aisle_id"].nunique()
        total_time = n_items * item_time + n_aisles * setup_time
        assert total_time <= daily_budget + 1e-3, f"Budget exceeded on day {d} for {k}: {total_time} > {daily_budget}"


def test_cpsat_compliance_mandates(mock_skus_for_opt):
    plan, metrics = solve_7day_count_plan_cpsat(
        mock_skus_for_opt,
        num_days=3,
        num_associates=2,
        daily_budget_min=120.0,
        max_solve_time_seconds=10.0,
    )
    
    # SKUs 0 to 4 have days_since_last_count = 95 >= 90, so they must be included
    mand_skus = {f"SKU_{i:04d}" for i in range(5)}
    scheduled_skus = set(plan["sku_id"].unique())
    assert mand_skus.issubset(scheduled_skus)
    assert metrics["compliance_mandates_satisfied"] >= 5


def test_greedy_scheduler(mock_skus_for_opt):
    plan, metrics = solve_7day_count_plan_greedy(
        mock_skus_for_opt,
        num_days=3,
        num_associates=2,
        daily_budget_min=120.0,
    )
    assert not plan.empty
    assert plan["sku_id"].nunique() == len(plan)
    assert metrics["labor_utilization_pct"] <= 100.0


def test_benchmark_suite(mock_skus_for_opt):
    df_bench, plans = run_all_optimization_benchmarks(
        mock_skus_for_opt,
        num_days=2,
        num_associates=2,
        daily_budget_min=60.0,
    )
    assert len(df_bench) >= 5
    assert "policy" in df_bench.columns
    assert "net_ev" in df_bench.columns
    assert "value_per_labor_hour" in df_bench.columns
