"""
Test Suite: Gap Prediction Targets, Features, Models, and Calibration (Module 04).
"""

import numpy as np
import pandas as pd
import pytest

from src.gap_prediction.targets import create_gap_targets
from src.gap_prediction.features import build_gap_features
from src.gap_prediction.calibration import ModelCalibrator, compute_ece
from src.gap_prediction.evaluate import compute_pr_auc, compute_precision_at_k
from src.gap_prediction.models import ZeroStreakHeuristic, LogisticRegressionGapModel, LightGBMGapClassifier


@pytest.fixture
def dummy_panel():
    return pd.DataFrame({
        "sku_id": ["SKU_1"] * 30 + ["SKU_2"] * 30,
        "day_index": list(range(1, 31)) * 2,
        "category": ["Health & Beauty"] * 30 + ["Produce"] * 30,
        "aisle_id": [1] * 30 + [2] * 30,
        "day_of_week": [i % 7 for i in range(30)] * 2,
        "is_weekend": [False] * 60,
        "is_holiday": [False] * 60,
        "selling_price": [12.99] * 60,
        "promo_flag": [False] * 60,
        "pos_sales": [0, 0, 0, 0, 1] * 12,
        "system_on_hand": [5] * 60,
        "true_on_hand": [0] * 30 + [5] * 30,
        "inventory_gap": [5] * 30 + [0] * 30,
        "is_phantom_stockout": [True] * 30 + [False] * 30,
        "consecutive_zero_sales": list(range(1, 31)) * 2,
        "days_since_last_count": [10] * 60,
        "days_since_last_receipt": [5] * 60,
    })


def test_target_generation(dummy_panel):
    df_targeted = create_gap_targets(dummy_panel, gap_threshold=1)
    assert "target_material_gap" in df_targeted.columns
    assert "target_phantom_oos" in df_targeted.columns
    assert df_targeted["target_phantom_oos"].sum() == 30


def test_feature_engineering_no_leakage(dummy_panel):
    df_feat = build_gap_features(dummy_panel)
    assert "normalized_zero_streak" in df_feat.columns
    assert "days_of_supply" in df_feat.columns
    assert not df_feat["normalized_zero_streak"].isna().any()


def test_calibration_reduces_error():
    """Isotonic regression must produce valid calibrated probabilities in [0, 1]."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_uncalibrated = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    
    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(y_uncalibrated, y_true)
    y_cal = calibrator.calibrate(y_uncalibrated)
    
    assert (y_cal >= 0.0).all() and (y_cal <= 1.0).all()


def test_pr_auc_and_precision_at_k():
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    
    pr_auc = compute_pr_auc(y_true, y_prob)
    prec_2 = compute_precision_at_k(y_true, y_prob, k=2)
    
    assert pr_auc == 1.0
    assert prec_2 == 1.0
