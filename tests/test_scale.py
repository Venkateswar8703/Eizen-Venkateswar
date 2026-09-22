"""
Unit and Integration Tests for Module 07: Closed-Loop Exploration and Scale.
"""

import pytest
import numpy as np
import pandas as pd
from src.scale.closed_loop_bandit import simulate_closed_loop_retraining


def test_closed_loop_simulation_runs():
    df_res = simulate_closed_loop_retraining(
        n_skus=200,
        n_cycles=4,
        audits_per_cycle=30,
        epsilon_values=[0.0, 0.10],
        seed=42,
    )
    
    assert not df_res.empty
    assert len(df_res) == 8 # 2 eps * 4 cycles
    assert "exploration_rate_eps" in df_res.columns
    assert "selection_bias_velocity_shift_pct" in df_res.columns
    assert "undetected_gaps_remaining" in df_res.columns
    
    # Check that non-zero exploration has non-negative detections
    assert (df_res["gaps_detected"] >= 0).all()


def test_exploration_reduces_undetected_gaps():
    df_res = simulate_closed_loop_retraining(
        n_skus=400,
        n_cycles=6,
        audits_per_cycle=50,
        epsilon_values=[0.0, 0.15],
        seed=42,
    )
    
    # Compare cycle 6
    c6_pure_greedy = df_res[(df_res["cycle"] == 6) & (df_res["exploration_rate_eps"] == 0.0)].iloc[0]
    c6_explore = df_res[(df_res["cycle"] == 6) & (df_res["exploration_rate_eps"] == 0.15)].iloc[0]
    
    # Pure greedy accumulates more undetected phantom stockouts in dark corners
    assert c6_explore["gap_capture_recall"] >= c6_pure_greedy["gap_capture_recall"]
