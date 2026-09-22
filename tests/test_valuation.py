"""
Unit and Integration Tests for Module 05: Valuation and Submodular Selection.
"""

import pytest
import numpy as np
import pandas as pd
from src.valuation.value_model import compute_expected_value_of_count, CATEGORY_RECOVERY_RATES
from src.valuation.submodular import greedy_submodular_count_selection
from src.valuation.sensitivity import run_valuation_sensitivity_sweep


@pytest.fixture
def mock_predictions_df():
    np.random.seed(42)
    categories = list(CATEGORY_RECOVERY_RATES.keys())
    data = []
    for i in range(100):
        data.append({
            "sku_id": f"SKU_{i:04d}",
            "aisle_id": f"AISLE_{i % 10:02d}",
            "category": np.random.choice(categories),
            "selling_price": np.random.uniform(3.0, 50.0),
            "base_demand_velocity": np.random.uniform(0.5, 8.0),
            "system_on_hand": np.random.uniform(0, 40),
            "predicted_gap_prob": np.random.uniform(0.05, 0.95),
            "count_duration_min": 1.6,
            "day": 730,
        })
    return pd.DataFrame(data)


def test_ev_computation(mock_predictions_df):
    df_ev = compute_expected_value_of_count(mock_predictions_df, hourly_wage=21.0)
    
    assert "expected_count_value_ev" in df_ev.columns
    assert "gross_loss_at_risk" in df_ev.columns
    assert "count_labor_cost" in df_ev.columns
    
    # Labor cost for 1.6 min at $21/hr = 21 * (1.6/60) = $0.56
    np.testing.assert_allclose(df_ev["count_labor_cost"].iloc[0], 0.56, atol=0.01)
    
    # Higher gap probability must strictly yield higher expected recovery for same SKU
    row = mock_predictions_df.iloc[[0]].copy()
    row_low = row.copy()
    row_low["predicted_gap_prob"] = 0.1
    row_high = row.copy()
    row_high["predicted_gap_prob"] = 0.9
    
    ev_low = compute_expected_value_of_count(row_low)["expected_count_value_ev"].iloc[0]
    ev_high = compute_expected_value_of_count(row_high)["expected_count_value_ev"].iloc[0]
    assert ev_high > ev_low


def test_submodular_selection_budget(mock_predictions_df):
    df_ev = compute_expected_value_of_count(mock_predictions_df)
    
    budget = 120.0 # 2 hours
    selected, summary = greedy_submodular_count_selection(
        df_ev,
        total_time_budget_min=budget,
        item_count_time_min=1.6,
        aisle_setup_time_min=4.0,
    )
    
    # Check that budget is strictly respected
    assert summary["total_time_used_min"] <= budget
    assert len(selected) > 0
    assert summary["num_skus_selected"] == len(selected)
    assert summary["time_utilization_pct"] <= 100.0


def test_submodular_diminishing_returns(mock_predictions_df):
    df_ev = compute_expected_value_of_count(mock_predictions_df)
    
    _, sum_small = greedy_submodular_count_selection(df_ev, total_time_budget_min=60.0)
    _, sum_large = greedy_submodular_count_selection(df_ev, total_time_budget_min=240.0)
    
    # Marginal value per minute should be higher or equal in smaller budget (greedy diminishing returns)
    val_rate_small = sum_small["net_value_after_aisle_overhead"] / sum_small["total_time_used_min"]
    val_rate_large = sum_large["net_value_after_aisle_overhead"] / sum_large["total_time_used_min"]
    
    assert val_rate_small >= val_rate_large - 1e-3


def test_sensitivity_sweep(mock_predictions_df):
    sweep_df = run_valuation_sensitivity_sweep(
        mock_predictions_df,
        wage_levels=[15.0, 30.0],
        recovery_multipliers=[0.5, 1.0],
        persistence_multipliers=[1.0],
        aisle_setup_times=[2.0, 4.0],
    )
    
    assert len(sweep_df) == 7
    assert set(sweep_df["parameter"].unique()) == {
        "hourly_wage", "recovery_rate_multiplier", "persistence_multiplier", "aisle_setup_minutes"
    }
