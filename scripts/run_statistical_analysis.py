"""
Statistical Analysis Runner: Computes empirical metrics across all 16 questions in Module 02.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stats.intermittency import categorize_store_skus
from src.stats.distribution_fitting import compare_category_distributions
from src.stats.censoring_diagnostics import compute_zero_streak_power, tobit_censored_mean_recovery
from src.stats.change_point import simulate_cusum_arl
from src.stats.survival import compute_category_hazard_rates
from src.stats.calibration_analysis import compute_calibration_curve, compute_conformal_interval_residuals


def main():
    print("================================================================")
    print("   PERPETUAL INVENTORY CHALLENGE — MODULE 02 STATISTICAL ENGINE  ")
    print("================================================================")
    
    data_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.parquet"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.csv.gz"

    print(f"Loading store panel from: {data_path}")
    df_panel = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)

    # 1. Intermittency ADI / CV² Quadrant Breakdown
    print("\n--- 1. Intermittency Breakdown (Syntetos-Boylan ADI / CV²) ---")
    df_intermittency = categorize_store_skus(df_panel)
    quad_counts = df_intermittency["intermittency_quadrant"].value_counts()
    for quad, count in quad_counts.items():
        print(f"  • {quad:<15}: {count:4d} SKUs ({count / len(df_intermittency) * 100:.1f}%)")

    # 2. Count Distribution Comparisons across 10 Categories
    print("\n--- 2. Discrete Count Distribution Fits & AIC/BIC Comparison ---")
    df_dist = compare_category_distributions(df_panel)
    print(df_dist[["category", "dispersion_ratio (Var/Mean)", "zero_percentage", "best_distribution"]].to_string(index=False))

    # 3. Censoring Tobit Recovery & Bias
    print("\n--- 3. Censored Demand vs. Naive POS Sales ---")
    stockout_mask = (df_panel["true_on_hand"] == 0).values
    tobit_res = tobit_censored_mean_recovery(df_panel["pos_sales"].values, stockout_mask)
    print(f"  • Naive Observed POS Mean:     {tobit_res['naive_observed_mean']} units/day")
    print(f"  • Tobit Recovered Latent Mean: {tobit_res['tobit_recovered_mean']} units/day")
    print(f"  • Downward Censoring Bias:     {tobit_res['censoring_bias_pct']}% underestimation")

    # 4. Zero-Streak Anomaly Power
    print("\n--- 4. Statistical Power of Zero-Streak Test (λ = 0.3/day) ---")
    df_streak = compute_zero_streak_power(daily_rates=[0.3, 1.0, 5.0], max_streak_days=10)
    streak_03 = df_streak[df_streak["mean_daily_demand_lambda"] == 0.3]
    for _, row in streak_03.iterrows():
        k = int(row["zero_streak_length_days"])
        pval = row["p_value_under_null"]
        sig = "✓ SIGNIFICANT (p < 0.05)" if row["significant_at_p05"] else "Not significant"
        print(f"  • Streak of {k:2d} zero-sales days: p-value = {pval:.4f} | {sig}")

    # 5. CUSUM Change-Point Run Length & Wasted Counts
    print("\n--- 5. Sequential CUSUM Change-Point Detector ---")
    cusum_stats = simulate_cusum_arl(in_control_lambda=2.5, threshold_h=4.0, allowance_delta=0.4)
    print(f"  • In-Control In-Stock Mean λ:   {cusum_stats['in_control_lambda']}")
    print(f"  • Average Run Length (ARL_0):   {cusum_stats['arl_0_days_to_false_alarm']} days to false alarm")
    print(f"  • Daily False Alarms (1600 SKUs): {cusum_stats['expected_false_alarms_per_day_1600_skus']} false alarms/day")
    print(f"  • Wasted Audit Labor / Week:    {cusum_stats['expected_wasted_counts_per_week']} audits/week")

    # 6. Discrete-Time Survival Hazard by Category
    print("\n--- 6. Survival Hazard Rates & Median Days to Gap Onset ---")
    df_hazard = compute_category_hazard_rates(df_panel)
    print(df_hazard[["category", "daily_hazard_rate", "median_days_to_gap", "recommended_audit_cycle_days"]].to_string(index=False))

    print("\n================================================================")
    print("Module 02 Statistical Diagnostic Engine Execution Complete!")


if __name__ == "__main__":
    main()
