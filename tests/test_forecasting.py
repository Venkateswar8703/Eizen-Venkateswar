"""
Test Suite: Demand Forecasting Baselines, LightGBM Quantiles, and Metrics (Module 03).
"""

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import SeasonalNaiveForecaster, CrostonForecaster, SBAForecaster, TSBForecaster
from src.forecasting.lgbm_forecaster import build_forecasting_features, LightGBMQuantileForecaster
from src.forecasting.evaluation import compute_mase, compute_rmsse, compute_pinball_loss


def test_seasonal_naive_periodicity():
    """Seasonal Naive must reproduce the 7-day pattern."""
    history = np.array([1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70])
    sn = SeasonalNaiveForecaster(season_length=7)
    fc = sn.predict(history, horizon=7)
    np.testing.assert_array_equal(fc, np.array([10, 20, 30, 40, 50, 60, 70]))


def test_sba_strictly_corrects_croston_bias():
    """SBA forecast must be strictly (1 - alpha/2) times Croston forecast."""
    history = np.array([0, 0, 4, 0, 0, 0, 5, 0, 0, 6])
    alpha = 0.1
    croston = CrostonForecaster(alpha=alpha).predict(history, horizon=5)
    sba = SBAForecaster(alpha=alpha).predict(history, horizon=5)
    np.testing.assert_allclose(sba, (1.0 - alpha / 2.0) * croston)


def test_mase_and_pinball_metrics():
    """MASE and Pinball loss should be non-negative and well-behaved."""
    y_true = np.array([5.0, 0.0, 3.0, 8.0])
    y_pred = np.array([4.0, 1.0, 3.0, 7.0])
    y_train = np.array([2.0, 4.0, 5.0, 3.0, 6.0, 5.0, 4.0, 5.0, 6.0, 4.0])
    
    mase = compute_mase(y_true, y_pred, y_train, seasonality=7)
    pinball = compute_pinball_loss(y_true, y_pred, quantile=0.5)
    
    assert mase >= 0.0
    assert pinball >= 0.0


def test_lgbm_quantile_monotonicity():
    """LightGBM quantile predictions must strictly satisfy q10 <= q50 <= q90."""
    df_dummy = pd.DataFrame({
        "sku_id": ["SKU_1"] * 50 + ["SKU_2"] * 50,
        "day_index": list(range(1, 51)) * 2,
        "category": ["Bakery"] * 50 + ["Dairy"] * 50,
        "aisle_id": [1] * 50 + [2] * 50,
        "day_of_week": [i % 7 for i in range(50)] * 2,
        "is_weekend": [False] * 100,
        "is_holiday": [False] * 100,
        "selling_price": [2.99] * 100,
        "promo_flag": [False] * 100,
        "pos_sales": np.random.poisson(lam=4.0, size=100),
    })
    df_feat = build_forecasting_features(df_dummy)
    
    forecaster = LightGBMQuantileForecaster(quantiles=[0.10, 0.50, 0.90])
    forecaster.fit(df_feat.iloc[:80])
    preds = forecaster.predict_quantiles(df_feat.iloc[80:])
    
    q10 = preds[0.10]
    q50 = preds[0.50]
    q90 = preds[0.90]
    
    assert (q10 <= q50).all(), "Monotonicity violated between q10 and q50"
    assert (q50 <= q90).all(), "Monotonicity violated between q50 and q90"
