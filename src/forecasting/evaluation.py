"""
Forecasting Evaluation Metrics:
Implements decision-relevant time-series metrics:
- MASE (Mean Absolute Scaled Error, Hyndman & Koehler 2006)
- RMSSE (Root Mean Squared Scaled Error, M5 standard)
- Pinball / Quantile Loss for probabilistic forecasts
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 7) -> float:
    """Computes Mean Absolute Scaled Error relative to seasonal in-sample naive."""
    if len(y_train) <= seasonality:
        scale = float(np.mean(np.abs(np.diff(y_train)))) if len(y_train) > 1 else 1.0
    else:
        # Seasonal naive in-sample scale denominator
        scale = float(np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality])))
    
    scale = max(1e-4, scale)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return round(mae / scale, 4)


def compute_rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, seasonality: int = 7) -> float:
    """Computes Root Mean Squared Scaled Error (M5 Competition standard)."""
    if len(y_train) <= seasonality:
        scale = float(np.mean((np.diff(y_train)) ** 2)) if len(y_train) > 1 else 1.0
    else:
        scale = float(np.mean((y_train[seasonality:] - y_train[:-seasonality]) ** 2))
        
    scale = max(1e-4, scale)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return round(float(np.sqrt(mse / scale)), 4)


def compute_pinball_loss(y_true: np.ndarray, y_pred_quantile: np.ndarray, quantile: float) -> float:
    """Computes Pinball / Quantile Loss for a specific quantile q in (0, 1)."""
    error = y_true - y_pred_quantile
    loss = np.maximum(quantile * error, (quantile - 1.0) * error)
    return round(float(np.mean(loss)), 4)
