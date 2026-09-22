"""
Master End-to-End Pipeline Runner for the Perpetual Inventory Challenge.
Executes all modules sequentially from Data Simulation to Final Optimization:

1. Data & Hidden Simulation (Module 01)
2. Statistical Analysis & Censoring Diagnostics (Module 02)
3. Demand Forecasting under Censoring (Module 03)
4. Inventory Gap Prediction & ML (Module 04)
5. Economic Valuation of Counts (Module 05)
6. Workforce Optimization & 7-Day Plan (Module 06)
7. Scale & Closed-Loop Bandit Exploration (Module 07)
"""

import os
import sys
import time
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXEC = sys.executable


def run_stage(stage_num: str, stage_name: str, script_rel_path: str):
    print("\n" + "=" * 75)
    print(f"STAGE {stage_num}: {stage_name.upper()}")
    print(f"Executing: {script_rel_path}")
    print("=" * 75)
    
    script_full_path = PROJECT_ROOT / script_rel_path
    if not script_full_path.exists():
        raise FileNotFoundError(f"Script not found at {script_full_path}")
        
    start_time = time.time()
    result = subprocess.run([PYTHON_EXEC, str(script_full_path)], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n[FAILED] Stage {stage_num} failed with return code {result.returncode} after {elapsed:.1f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n[SUCCESS] Stage {stage_num} completed in {elapsed:.1f}s")


def main():
    total_start = time.time()
    print("*" * 75)
    print("EIZEN AI — PERPETUAL INVENTORY CHALLENGE: END-TO-END PIPELINE")
    print("*" * 75)
    
    # 1. Data Simulation
    run_stage("01", "Hidden Inventory Simulation", "scripts/generate_data.py")
    
    # 2. Statistical Analysis
    run_stage("02", "Statistical Diagnostics & Censoring", "scripts/run_statistical_analysis.py")
    
    # 3. Demand Forecasting
    run_stage("03", "Demand Forecasting Under Censoring", "scripts/train_forecasting.py")
    
    # 4. Gap Prediction
    run_stage("04", "Inventory Gap Prediction & ML", "scripts/train_gap_prediction.py")
    
    # 5. Economic Valuation
    run_stage("05", "Economic Value Engine", "scripts/run_valuation.py")
    
    # 6. Workforce Optimization
    run_stage("06", "Workforce Optimization & 7-Day Schedule", "scripts/run_optimization.py")
    
    # 7. Scale & Bandit Exploration
    run_stage("07", "Scale to 2,000 Stores & Closed-Loop Bandit Exploration", "scripts/run_scale_and_bandits.py")
    
    total_elapsed = time.time() - total_start
    print("\n" + "*" * 75)
    print(f"ALL MODULES EXECUTED SUCCESSFULLY IN {total_elapsed/60.0:.2f} MINUTES")
    print("*" * 75)
    print("\nTangible Deliverables Generated:")
    print("  - data/simulated/store_daily_panel.parquet")
    print("  - data/processed/product_master.parquet")
    print("  - data/results/forecasting_metrics.parquet")
    print("  - data/results/gap_leaderboard.parquet")
    print("  - data/results/gap_predictions.parquet")
    print("  - data/results/valuation_ev_scored.parquet")
    print("  - data/results/valuation_sensitivity.parquet")
    print("  - data/processed/count_plan_7day.parquet")
    print("  - data/results/optimization_benchmarks.parquet")
    print("  - data/results/closed_loop_bias_metrics.parquet")


if __name__ == "__main__":
    main()
