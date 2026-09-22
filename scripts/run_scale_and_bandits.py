"""
Runner Script for Module 07: Closed-Loop Feedback Bias & Bandit Exploration Simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.scale.closed_loop_bandit import simulate_closed_loop_retraining


def main():
    print("=" * 60)
    print("MODULE 07: CLOSED-LOOP FEEDBACK BIAS & BANDIT SIMULATION")
    print("=" * 60)
    
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Simulating 8 retraining cycles under exploration rates epsilon in [0.0, 0.05, 0.15]...")
    df_bandit = simulate_closed_loop_retraining(
        n_skus=1600,
        n_cycles=8,
        audits_per_cycle=150,
        epsilon_values=[0.0, 0.05, 0.15],
        seed=42,
    )
    
    out_path = results_dir / "closed_loop_bias_metrics.parquet"
    df_bandit.to_parquet(out_path, index=False)
    print(f"Saved bandit simulation results to {out_path}.")
    
    print("\nSummary at Cycle 8 (Final Horizon):")
    final_cycle = df_bandit[df_bandit["cycle"] == 8]
    print(final_cycle[["exploration_rate_eps", "gaps_detected", "undetected_gaps_remaining", "audit_precision", "gap_capture_recall", "selection_bias_velocity_shift_pct"]].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("MODULE 07 SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
