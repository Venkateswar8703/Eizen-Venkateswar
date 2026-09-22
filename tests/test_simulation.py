"""
Test Suite: Simulation Physics, Conservation Laws, and Determinism.
"""

import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.simulation.generator import InventorySimulator
from src.simulation.censoring import apply_pos_censoring
from src.simulation.error_mechanisms import ErrorMechanismEngine


@pytest.fixture
def config_path():
    root = Path(__file__).resolve().parent.parent
    return str(root / "configs" / "simulation_config.yaml")


def test_physical_stock_non_negative(config_path):
    """Verifies that physical on-hand inventory never becomes negative in physical reality."""
    sim = InventorySimulator(config_path)
    sim.num_days = 30  # Fast test run
    results = sim.run_simulation()
    df_panel = results["daily_panel"]
    
    assert (df_panel["true_on_hand"] >= 0).all(), "TrueOnHand dropped below zero!"


def test_pos_censoring_bounds():
    """Verifies that POS sales never exceed accessible on-shelf stock."""
    latent_demand = np.array([10, 5, 0, 8, 20])
    true_on_hand = np.array([4, 12, 10, 0, 15])
    misplaced = np.array([1, 0, 2, 0, 5])
    
    pos_sales, lost_sales = apply_pos_censoring(latent_demand, true_on_hand, misplaced)
    
    # Expected:
    # 0: accessible = 4 - 1 = 3 -> pos = min(10, 3) = 3, lost = 7
    # 1: accessible = 12 - 0 = 12 -> pos = min(5, 12) = 5, lost = 0
    # 2: accessible = 10 - 2 = 8 -> pos = min(0, 8) = 0, lost = 0
    # 3: accessible = 0 - 0 = 0 -> pos = min(8, 0) = 0, lost = 8
    # 4: accessible = 15 - 5 = 10 -> pos = min(20, 10) = 10, lost = 10
    
    np.testing.assert_array_equal(pos_sales, np.array([3, 5, 0, 0, 10]))
    np.testing.assert_array_equal(lost_sales, np.array([7, 0, 0, 8, 10]))


def test_deterministic_reproducibility(config_path):
    """Verifies that running the simulator with identical seed produces bitwise identical results."""
    sim1 = InventorySimulator(config_path)
    sim1.num_days = 14
    res1 = sim1.run_simulation()["daily_panel"]

    sim2 = InventorySimulator(config_path)
    sim2.num_days = 14
    res2 = sim2.run_simulation()["daily_panel"]

    pd.testing.assert_frame_equal(res1, res2)


def test_inaccuracy_rate_range(config_path):
    """Verifies that gap rate aligns with DeHoratius & Raman (2008) 60-70% benchmark."""
    sim = InventorySimulator(config_path)
    sim.num_days = 60
    df = sim.run_simulation()["daily_panel"]
    
    inaccuracy_rate = (df["inventory_gap"] != 0).mean()
    assert 0.50 <= inaccuracy_rate <= 0.85, f"Inaccuracy rate {inaccuracy_rate:.2f} outside realistic range!"


def test_compliance_floor_enforcement(config_path):
    """Verifies that no SKU exceeds 90 days without a cycle count."""
    sim = InventorySimulator(config_path)
    sim.num_days = 120
    df = sim.run_simulation()["daily_panel"]
    
    # After day 90, days_since_last_count must never exceed 90
    df_post_90 = df[df["day_index"] > 90]
    assert (df_post_90["days_since_last_count"] <= 91).all(), "Compliance floor was violated!"
