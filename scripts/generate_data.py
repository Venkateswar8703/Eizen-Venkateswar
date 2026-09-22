"""
Dataset Generation Script: Runs the 730-day simulation and saves the parquet datasets.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.generator import InventorySimulator


def main():
    config_path = PROJECT_ROOT / "configs" / "simulation_config.yaml"
    data_dir = PROJECT_ROOT / "data"
    sim_dir = data_dir / "simulated"
    proc_dir = data_dir / "processed"

    sim_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("      PERPETUAL INVENTORY CHALLENGE — SIMULATION ENGINE         ")
    print("================================================================")
    print(f"Loading configuration from: {config_path}")
    
    t0 = time.time()
    simulator = InventorySimulator(str(config_path))
    
    print(f"Initializing {len(simulator.skus)} SKUs across 10 categories...")
    print(f"Simulating {simulator.num_days} days (104 weeks) for store {simulator.store_id}...")
    
    results = simulator.run_simulation()
    
    df_panel = results["daily_panel"]
    df_audits = results["audit_logs"]
    df_master = results["product_master"]
    
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.2f} seconds!")
    print("----------------------------------------------------------------")
    print(f"Daily Panel Records:  {len(df_panel):,} rows")
    print(f"Historical Audits:    {len(df_audits):,} count records")
    print(f"Product Master:       {len(df_master):,} SKUs")
    print("----------------------------------------------------------------")

    # Realism & Sanity Statistics
    inaccurate_pct = (df_panel["inventory_gap"] != 0).mean() * 100
    phantom_pct = (df_panel["is_phantom_stockout"]).mean() * 100
    total_sales = df_panel["pos_sales"].sum()
    total_lost = df_panel["true_lost_sales"].sum()
    
    print("Validation & Realism Checks:")
    print(f"  • Inventory Inaccuracy Rate: {inaccurate_pct:.1f}% (Benchmark: 60-70% in DeHoratius & Raman)")
    print(f"  • Phantom Stockout Rate:     {phantom_pct:.2f}% of SKU-days")
    print(f"  • Total POS Units Sold:      {total_sales:,}")
    print(f"  • Total Lost Sales Units:    {total_lost:,} ({total_lost / (total_sales + total_lost) * 100:.2f}% unserved)")
    print("----------------------------------------------------------------")

    # Save datasets (Parquet preferred, with CSV.gz fallback)
    panel_file = sim_dir / "store_daily_panel.parquet"
    audit_file = proc_dir / "historical_audit_log.parquet"
    master_file = proc_dir / "product_master.parquet"

    print("Saving datasets...")
    try:
        df_panel.to_parquet(panel_file, index=False)
        df_audits.to_parquet(audit_file, index=False)
        df_master.to_parquet(master_file, index=False)
        print(f"  ✓ Saved: {panel_file} ({os.path.getsize(panel_file) / (1024 * 1024):.1f} MB)")
        print(f"  ✓ Saved: {audit_file} ({os.path.getsize(audit_file) / (1024 * 1024):.2f} MB)")
        print(f"  ✓ Saved: {master_file} ({os.path.getsize(master_file) / 1024:.1f} KB)")
    except Exception as e:
        print(f"  Notice: Parquet engine ({e}) - saving as compressed CSV...")
        panel_csv = sim_dir / "store_daily_panel.csv.gz"
        audit_csv = proc_dir / "historical_audit_log.csv.gz"
        master_csv = proc_dir / "product_master.csv.gz"
        df_panel.to_csv(panel_csv, index=False, compression="gzip")
        df_audits.to_csv(audit_csv, index=False, compression="gzip")
        df_master.to_csv(master_csv, index=False, compression="gzip")
        print(f"  ✓ Saved: {panel_csv} ({os.path.getsize(panel_csv) / (1024 * 1024):.1f} MB)")
        print(f"  ✓ Saved: {audit_csv} ({os.path.getsize(audit_csv) / (1024 * 1024):.2f} MB)")
        print(f"  ✓ Saved: {master_csv} ({os.path.getsize(master_csv) / 1024:.1f} KB)")
    print("================================================================")
    print("Module 01 Dataset Generation Complete!")


if __name__ == "__main__":
    main()
