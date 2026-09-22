"""
Runner Script for Module 04 Inventory Gap Prediction:
Trains and benchmarks all gap prediction model tiers on the 730-day store dataset.
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gap_prediction.targets import create_gap_targets
from src.gap_prediction.features import build_gap_features
from src.gap_prediction.calibration import ModelCalibrator, compute_ece
from src.gap_prediction.evaluate import evaluate_gap_model
from src.gap_prediction.models import (
    ZeroStreakHeuristic,
    HighActivityIndexHeuristic,
    LowRecordedInventoryHeuristic,
    LogisticRegressionGapModel,
    LightGBMGapClassifier,
    DLinearGapBaseline,
    AisleGraphRelationalModel,
)


def main():
    print("================================================================")
    print("     PERPETUAL INVENTORY CHALLENGE — MODULE 04 GAP PREDICTION   ")
    print("================================================================")
    
    panel_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.parquet"
    if not panel_path.exists():
        panel_path = PROJECT_ROOT / "data" / "simulated" / "store_daily_panel.csv.gz"

    forecast_path = PROJECT_ROOT / "data" / "results" / "demand_forecasts.parquet"
    df_forecasts = pd.read_parquet(forecast_path) if forecast_path.exists() else None

    print(f"Loading store panel from: {panel_path}")
    df_panel = pd.read_parquet(panel_path) if str(panel_path).endswith(".parquet") else pd.read_csv(panel_path)

    # 1. Target and Feature Generation
    print("Generating primary targets and point-in-time features...")
    df_targeted = create_gap_targets(df_panel, gap_threshold=1)
    df_feat = build_gap_features(df_targeted, df_forecasts=df_forecasts)

    # 2. Temporal Split (Zero future leakage)
    train_mask = df_feat["day_index"] <= 550
    val_mask = (df_feat["day_index"] > 550) & (df_feat["day_index"] <= 640)
    test_mask = df_feat["day_index"] > 640

    df_train = df_feat[train_mask].copy()
    df_val = df_feat[val_mask].copy()
    df_test = df_feat[test_mask].copy()

    y_test = df_test["target_phantom_oos"].values
    
    # Dollars at risk for top-k valuation: margin * demand * persistence proxy
    dollars_at_risk = (df_test["selling_price"] * 0.40 * df_test["base_demand_velocity"] * 5.0).values

    print(f"Dataset Split: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,} rows")
    print("Base Phantom Stockout Rate in Test Set:", f"{np.mean(y_test) * 100:.2f}%")

    t0 = time.time()
    leaderboard_records = []

    # --------------------------------------------------------------------------
    # Tier 0: Heuristics
    # --------------------------------------------------------------------------
    print("\n[Tier 0] Evaluating Classical Heuristics...")
    heur_zs = ZeroStreakHeuristic()
    p_zs = heur_zs.predict_proba(df_test)
    ece_zs, _ = compute_ece(y_test, p_zs)
    res_zs = evaluate_gap_model("Tier 0: Zero-Streak Rule", y_test, p_zs, dollars_at_risk, k=250)
    res_zs["ECE (Calibration)"] = ece_zs
    leaderboard_records.append(res_zs)

    heur_act = HighActivityIndexHeuristic()
    p_act = heur_act.predict_proba(df_test)
    ece_act, _ = compute_ece(y_test, p_act)
    res_act = evaluate_gap_model("Tier 0: High-Activity Index", y_test, p_act, dollars_at_risk, k=250)
    res_act["ECE (Calibration)"] = ece_act
    leaderboard_records.append(res_act)

    heur_low = LowRecordedInventoryHeuristic()
    p_low = heur_low.predict_proba(df_test)
    ece_low, _ = compute_ece(y_test, p_low)
    res_low = evaluate_gap_model("Tier 0: Low-Recorded-Inventory Index", y_test, p_low, dollars_at_risk, k=250)
    res_low["ECE (Calibration)"] = ece_low
    leaderboard_records.append(res_low)

    # --------------------------------------------------------------------------
    # Tier 1: Classical Linear / Logistic
    # --------------------------------------------------------------------------
    print("[Tier 1] Fitting Logistic Regression Baseline...")
    log_model = LogisticRegressionGapModel()
    log_model.fit(df_train, target_col="target_phantom_oos")
    p_log = log_model.predict_proba(df_test)
    ece_log, _ = compute_ece(y_test, p_log)
    res_log = evaluate_gap_model("Tier 1: Logistic Regression", y_test, p_log, dollars_at_risk, k=250)
    res_log["ECE (Calibration)"] = ece_log
    leaderboard_records.append(res_log)

    # --------------------------------------------------------------------------
    # Tier 3: DLinear Time-Series Baseline (Zeng et al., AAAI 2023)
    # --------------------------------------------------------------------------
    print("[Tier 3] Fitting DLinear Time-Series Counter-Baseline...")
    dlinear_model = DLinearGapBaseline()
    dlinear_model.fit(df_train, target_col="target_phantom_oos")
    p_dlinear = dlinear_model.predict_proba(df_test)
    ece_dlin, _ = compute_ece(y_test, p_dlinear)
    res_dlin = evaluate_gap_model("Tier 3: DLinear Baseline (AAAI 2023)", y_test, p_dlinear, dollars_at_risk, k=250)
    res_dlin["ECE (Calibration)"] = ece_dlin
    leaderboard_records.append(res_dlin)

    # --------------------------------------------------------------------------
    # Tier 1/2: LightGBM Gradient Boosted Decision Tree
    # --------------------------------------------------------------------------
    print("[Tier 1/2] Fitting LightGBM GBDT...")
    lgbm_model = LightGBMGapClassifier()
    lgbm_model.fit(df_train, target_col="target_phantom_oos")
    p_lgb_raw = lgbm_model.predict_proba(df_test)
    ece_lgb_raw, _ = compute_ece(y_test, p_lgb_raw)
    res_lgb = evaluate_gap_model("Tier 1: LightGBM (Raw)", y_test, p_lgb_raw, dollars_at_risk, k=250)
    res_lgb["ECE (Calibration)"] = ece_lgb_raw
    leaderboard_records.append(res_lgb)

    # Calibrate LightGBM with Isotonic Regression on Validation Set
    print("[Calibration] Fitting Isotonic Calibrator on Validation Set...")
    p_val_raw = lgbm_model.predict_proba(df_val)
    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(p_val_raw, df_val["target_phantom_oos"].values)
    p_lgb_cal = calibrator.calibrate(p_lgb_raw)
    ece_lgb_cal, _ = compute_ece(y_test, p_lgb_cal)
    res_lgb_cal = evaluate_gap_model("Tier 1: LightGBM (Isotonic Calibrated)", y_test, p_lgb_cal, dollars_at_risk, k=250)
    res_lgb_cal["ECE (Calibration)"] = ece_lgb_cal
    leaderboard_records.append(res_lgb_cal)

    # --------------------------------------------------------------------------
    # Research Extension: Aisle Graph Relational Model (GNN)
    # --------------------------------------------------------------------------
    print("[Research Extension] Evaluating Aisle Graph Relational Model...")
    gnn_model = AisleGraphRelationalModel(base_classifier=lgbm_model)
    p_gnn = gnn_model.predict_proba(df_test)
    ece_gnn, _ = compute_ece(y_test, p_gnn)
    res_gnn = evaluate_gap_model("Research Ext: Aisle Graph Relational (GNN)", y_test, p_gnn, dollars_at_risk, k=250)
    res_gnn["ECE (Calibration)"] = ece_gnn
    leaderboard_records.append(res_gnn)

    elapsed = time.time() - t0

    df_leaderboard = pd.DataFrame(leaderboard_records).sort_values("PR-AUC", ascending=False)

    print("\n----------------------------------------------------------------")
    print("GAP PREDICTION MODEL LEADERBOARD (Test Period: Days 641-730):")
    print("----------------------------------------------------------------")
    print(df_leaderboard[["Model", "PR-AUC", "Precision@250", "ECE (Calibration)", "Lift over Random", "Dollar Capture Share (%)"]].to_string(index=False))
    print("----------------------------------------------------------------")

    # Feature Importance
    df_importance = lgbm_model.get_feature_importances()
    print("\nTOP FEATURE IMPORTANCES (LightGBM Gain Share):")
    for _, row in df_importance.head(7).iterrows():
        print(f"  • {row['Feature']:<35}: {row['Relative_Share']*100:.2f}%")

    # Save Results
    results_dir = PROJECT_ROOT / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    leaderboard_path = results_dir / "gap_leaderboard.parquet"
    predictions_path = results_dir / "gap_predictions.parquet"

    df_leaderboard.to_parquet(leaderboard_path, index=False)
    
    # Save predictions panel for Module 05 Valuation and Module 06 Optimization
    df_test["predicted_gap_prob"] = p_lgb_cal
    df_test["predicted_gap_prob_raw"] = p_lgb_raw
    
    cols_to_export = [
        "date", "day_index", "sku_id", "category", "aisle_id", "selling_price",
        "system_on_hand", "true_on_hand", "inventory_gap", "is_phantom_stockout",
        "base_demand_velocity", "predicted_gap_prob", "days_since_last_count"
    ]
    df_test[cols_to_export].to_parquet(predictions_path, index=False)

    print(f"\n✓ Saved leaderboard to: {leaderboard_path}")
    print(f"✓ Saved gap predictions to: {predictions_path} ({len(df_test):,} rows)")
    print(f"Completed gap prediction pipeline in {elapsed:.2f} seconds!")
    print("================================================================")


if __name__ == "__main__":
    main()
