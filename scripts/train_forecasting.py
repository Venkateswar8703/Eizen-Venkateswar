"""
Runner Script for Module 03 Demand Forecasting:
Executes the temporal backtest across all 1,600 SKUs and saves the benchmark metrics
and quantile forecasts to data/results/.
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.backtesting import run_temporal_forecasting_benchmark


def main():
    print("================================================================")
    print("     PERPETUAL INVENTORY CHALLENGE — MODULE 03 FORECASTING      ")
    print("================================================================")
    
    data_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.parquet"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.csv.gz"

    print(f"Loading store panel from: {data_path}")
    df_panel = pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)

    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    benchmark_results = run_temporal_forecasting_benchmark(df_panel)
    elapsed = time.time() - t0

    df_metrics = benchmark_results["metrics_table"]
    bias_stats = benchmark_results["censoring_bias"]
    df_forecasts = benchmark_results["test_forecasts"]

    print("\n----------------------------------------------------------------")
    print("DEMAND FORECASTING BENCHMARK LEADERBOARD (Test Period: Days 641-730):")
    print("----------------------------------------------------------------")
    print(df_metrics.to_string(index=False))
    print("----------------------------------------------------------------")
    
    print("\nCENSORING BIAS QUANTIFICATION:")
    for k, v in bias_stats.items():
        print(f"  • {k:<30}: {v}")

    # Save Results
    metrics_path = results_dir / "forecasting_metrics.parquet"
    forecasts_path = results_dir / "demand_forecasts.parquet"
    
    df_metrics.to_parquet(metrics_path, index=False)
    # Save test forecasts subset with key columns
    cols_to_save = [
        "date", "day_index", "sku_id", "category", "aisle_id", "pos_sales", "latent_demand",
        "system_on_hand", "true_on_hand", "is_phantom_stockout",
        "pred_q10_aware", "pred_q50_aware", "pred_q90_aware", "pred_q50_naive"
    ]
    df_forecasts[cols_to_save].to_parquet(forecasts_path, index=False)

    print(f"\n✓ Saved metrics to: {metrics_path}")
    print(f"✓ Saved forecasts to: {forecasts_path} ({len(df_forecasts):,} test rows)")
    print(f"Completed forecasting pipeline in {elapsed:.2f} seconds!")
    print("================================================================")


if __name__ == "__main__":
    main()
