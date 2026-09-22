"""
Temporal Backtesting & Benchmark Engine:
Executes rolling-origin backtesting across all 1,600 SKUs comparing classical baselines
against LightGBM (Naive vs. Censoring-Aware).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

from .baselines import SeasonalNaiveForecaster, SimpleETSForecaster, CrostonForecaster, SBAForecaster, TSBForecaster
from .lgbm_forecaster import build_forecasting_features, LightGBMQuantileForecaster
from .evaluation import compute_mase, compute_rmsse, compute_pinball_loss
from .censoring import create_censoring_mask, quantify_censoring_bias


def run_temporal_forecasting_benchmark(df_panel: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes a complete temporal backtest on the store daily panel:
    - Train: Days 1 to 550
    - Test: Days 641 to 730 (last 90 days evaluation window)
    """
    print("Building point-in-time time series lag & rolling features...")
    df_feat = build_forecasting_features(df_panel)
    
    # Train / Test Temporal Split (Zero lookahead leakage)
    train_mask = df_feat["day_index"] <= 550
    test_mask = df_feat["day_index"] > 640
    
    df_train = df_feat[train_mask].copy()
    df_test = df_feat[test_mask].copy()
    
    y_test_true = df_test["pos_sales"].values
    y_test_latent = df_test["latent_demand"].values
    
    # 1. Train LightGBM Naive (Trained on raw POS sales)
    print("Fitting LightGBM Quantile Forecaster (Naive Mode)...")
    lgbm_naive = LightGBMQuantileForecaster(quantiles=[0.10, 0.50, 0.90])
    lgbm_naive.fit(df_train, target_col="pos_sales")
    preds_naive = lgbm_naive.predict_quantiles(df_test)
    
    # 2. Train LightGBM Censoring-Aware (Downweighted / masked stockouts)
    print("Fitting LightGBM Quantile Forecaster (Censoring-Aware Mode)...")
    censored_weights = np.where(df_train["system_on_hand"] <= 0, 0.10, 1.0)
    lgbm_aware = LightGBMQuantileForecaster(quantiles=[0.10, 0.50, 0.90])
    lgbm_aware.fit(df_train, target_col="pos_sales", sample_weight=censored_weights)
    preds_aware = lgbm_aware.predict_quantiles(df_test)

    # 3. Evaluate Classical Baselines across all SKUs
    print("Evaluating Classical Baselines (Seasonal Naive, ETS, Croston, SBA, TSB)...")
    baseline_models = {
        "Seasonal Naive (7d)": SeasonalNaiveForecaster(season_length=7),
        "Simple ETS": SimpleETSForecaster(alpha=0.2),
        "Croston (1972)": CrostonForecaster(alpha=0.1),
        "SBA (2005)": SBAForecaster(alpha=0.1),
        "TSB (2011)": TSBForecaster(alpha=0.15, beta=0.10),
    }

    baseline_preds = {name: [] for name in baseline_models}
    
    # Pre-group training and testing slices into dictionaries for O(1) retrieval
    train_groups = {sku: grp["pos_sales"].values for sku, grp in df_train.groupby("sku_id")}
    test_counts = {sku: len(grp) for sku, grp in df_test.groupby("sku_id")}
    
    for sku in df_panel["sku_id"].unique():
        if sku not in test_counts or test_counts[sku] == 0:
            continue
        sku_train = train_groups.get(sku, np.array([]))
        sku_test_len = test_counts[sku]
        
        for name, model in baseline_models.items():
            fc = model.predict(sku_train, horizon=sku_test_len)
            baseline_preds[name].extend(fc)

    for name in baseline_preds:
        baseline_preds[name] = np.array(baseline_preds[name])

    # 4. Compute Benchmark Metrics
    y_train_full = df_train["pos_sales"].values
    
    model_evaluations = []
    
    # Evaluate Baselines
    for name, p in baseline_preds.items():
        mase_val = compute_mase(y_test_true, p, y_train_full)
        rmsse_val = compute_rmsse(y_test_true, p, y_train_full)
        p50_loss = compute_pinball_loss(y_test_true, p, 0.50)
        model_evaluations.append({
            "Model": name,
            "MASE": mase_val,
            "RMSSE": rmsse_val,
            "Pinball Loss (q50)": p50_loss,
            "Pinball Loss (q90)": np.nan,
            "Type": "Classical Baseline"
        })

    # Evaluate LightGBM Naive
    mase_lgb_naive = compute_mase(y_test_true, preds_naive[0.50], y_train_full)
    rmsse_lgb_naive = compute_rmsse(y_test_true, preds_naive[0.50], y_train_full)
    p10_naive = compute_pinball_loss(y_test_true, preds_naive[0.10], 0.10)
    p50_naive = compute_pinball_loss(y_test_true, preds_naive[0.50], 0.50)
    p90_naive = compute_pinball_loss(y_test_true, preds_naive[0.90], 0.90)
    model_evaluations.append({
        "Model": "LightGBM (Naive Sales)",
        "MASE": mase_lgb_naive,
        "RMSSE": rmsse_lgb_naive,
        "Pinball Loss (q50)": p50_naive,
        "Pinball Loss (q90)": p90_naive,
        "Type": "Machine Learning"
    })

    # Evaluate LightGBM Censoring-Aware
    mase_lgb_aware = compute_mase(y_test_true, preds_aware[0.50], y_train_full)
    rmsse_lgb_aware = compute_rmsse(y_test_true, preds_aware[0.50], y_train_full)
    p10_aware = compute_pinball_loss(y_test_true, preds_aware[0.10], 0.10)
    p50_aware = compute_pinball_loss(y_test_true, preds_aware[0.50], 0.50)
    p90_aware = compute_pinball_loss(y_test_true, preds_aware[0.90], 0.90)
    model_evaluations.append({
        "Model": "LightGBM (Censoring-Aware)",
        "MASE": mase_lgb_aware,
        "RMSSE": rmsse_lgb_aware,
        "Pinball Loss (q50)": p50_aware,
        "Pinball Loss (q90)": p90_aware,
        "Type": "Machine Learning (Censored Loss)"
    })

    df_metrics = pd.DataFrame(model_evaluations).sort_values("MASE")
    
    # 5. Quantify Censoring Bias
    bias_stats = quantify_censoring_bias(preds_naive[0.50], preds_aware[0.50], y_test_latent)
    
    # 6. Save Forecast Panel
    df_test["pred_q10_aware"] = preds_aware[0.10]
    df_test["pred_q50_aware"] = preds_aware[0.50]
    df_test["pred_q90_aware"] = preds_aware[0.90]
    df_test["pred_q50_naive"] = preds_naive[0.50]

    return {
        "metrics_table": df_metrics,
        "censoring_bias": bias_stats,
        "test_forecasts": df_test,
    }
