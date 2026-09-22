"""
Runner Script for Module 05: Economic Valuation of Cycle Counts.
Consumes calibrated gap predictions from Module 04, computes EV(i, d),
runs submodular selection, executes sensitivity sweeps, and saves results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.valuation.value_model import compute_expected_value_of_count
from src.valuation.submodular import greedy_submodular_count_selection
from src.valuation.sensitivity import run_valuation_sensitivity_sweep


def main():
    print("=" * 60)
    print("MODULE 05: ECONOMIC VALUE ENGINE EXECUTION")
    print("=" * 60)
    
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pred_path = results_dir / "gap_predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing gap predictions at {pred_path}. Run Module 04 first.")
        
    df_preds = pd.read_parquet(pred_path)
    print(f"Loaded {len(df_preds):,} predictions from Module 04.")
    
    # 1. Compute Expected Economic Value EV(i, d)
    df_ev = compute_expected_value_of_count(
        df_preds,
        hourly_wage=21.0,
        basket_abandonment_prob=0.12,
        avg_basket_margin=18.0,
        substitution_rate=0.45,
        false_alarm_penalty=0.50,
    )
    
    output_ev_path = results_dir / "valuation_ev_scored.parquet"
    df_ev.to_parquet(output_ev_path, index=False)
    print(f"Computed EV for all test rows. Saved to {output_ev_path}.")
    
    # Summary of latest day (day 730) for optimization handoff
    day_col = "day_index" if "day_index" in df_ev.columns else "day"
    max_day = df_ev[day_col].max()
    df_latest = df_ev[df_ev[day_col] == max_day].copy()
    print(f"\nEvaluating Latest Day (Day {max_day}) with {len(df_latest)} SKUs:")
    
    mean_ev = df_latest["expected_count_value_ev"].mean()
    pos_ev_count = (df_latest["expected_count_value_ev"] > 0).sum()
    top10_val = df_latest.nlargest(10, "expected_count_value_ev")
    
    print(f"  Mean EV across all SKUs: ${mean_ev:.2f}")
    print(f"  SKUs with Net Positive EV: {pos_ev_count} / {len(df_latest)} ({pos_ev_count/len(df_latest):.1%})")
    print("\nTop 5 Highest Value SKUs to Audit:")
    for _, r in top10_val.head(5).iterrows():
        print(f"    SKU {r['sku_id']} ({r['category']}): GapProb={r['predicted_gap_prob']:.2f}, SOH={r['system_on_hand']:.0f}, EV=${r['expected_count_value_ev']:.2f}")
        
    # 2. Run Greedy Submodular Selection for single-day budget (720 min = 3 associates * 240 min)
    selected_df, summary = greedy_submodular_count_selection(
        df_latest,
        total_time_budget_min=720.0,
        item_count_time_min=1.6,
        aisle_setup_time_min=4.0,
        hourly_wage=21.0,
    )
    print(f"\nSingle-Day Greedy Submodular Audit Plan (720 min budget):")
    print(f"  SKUs Selected: {summary['num_skus_selected']}")
    print(f"  Aisles Visited: {summary['distinct_aisles_visited']}")
    print(f"  Time Used: {summary['total_time_used_min']:.1f} min ({summary['time_utilization_pct']:.1f}%)")
    print(f"  Gross EV Recovered: ${summary['total_expected_economic_value']:.2f}")
    print(f"  Net Value after Aisle Overhead: ${summary['net_value_after_aisle_overhead']:.2f}")
    
    # 3. Sensitivity Sweeps
    print("\nExecuting Parameter Sensitivity Sweeps...")
    df_sens = run_valuation_sensitivity_sweep(df_latest)
    sens_path = results_dir / "valuation_sensitivity.parquet"
    df_sens.to_parquet(sens_path, index=False)
    print(f"Saved sensitivity matrix to {sens_path}.")
    print("\nSensitivity Highlights:")
    print(df_sens[["parameter", "param_value", "unit", "num_skus_selected", "net_value"]].to_string(index=False))
    
    print("\n" + "=" * 60)
    print("MODULE 05 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
