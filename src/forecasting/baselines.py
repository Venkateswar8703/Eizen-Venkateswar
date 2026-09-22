"""
Demand Forecasting Baselines:
Implements classical statistical and intermittent demand forecasting models:
1. Seasonal Naive (7-day retail seasonality)
2. Simple Exponential Smoothing (ETS)
3. Croston's Method (1972)
4. Syntetos-Boylan Approximation (SBA, 2005)
5. Teunter-Syntetos-Babai (TSB, 2011)
"""

from typing import Dict, Tuple, Optional
import numpy as np


class SeasonalNaiveForecaster:
    """Predicts demand using the value from exactly one seasonal cycle ago (e.g. 7 days)."""

    def __init__(self, season_length: int = 7):
        self.season_length = season_length

    def predict(self, history: np.ndarray, horizon: int = 14) -> np.ndarray:
        if len(history) < self.season_length:
            return np.full(horizon, float(np.mean(history)) if len(history) > 0 else 0.0)
        recent_cycle = history[-self.season_length:]
        # Tile the recent seasonal cycle across the horizon
        reps = int(np.ceil(horizon / self.season_length))
        return np.tile(recent_cycle, reps)[:horizon].astype(float)


class SimpleETSForecaster:
    """Simple Exponential Smoothing with adaptive alpha parameter."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def predict(self, history: np.ndarray, horizon: int = 14) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(horizon)
        level = float(history[0])
        for val in history[1:]:
            level = self.alpha * val + (1.0 - self.alpha) * level
        return np.full(horizon, level)


class CrostonForecaster:
    """
    Croston's Method (1972) for intermittent demand.
    Separates demand magnitude z_t from inter-arrival interval p_t:
        y_hat = z_hat / p_hat
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def predict(self, history: np.ndarray, horizon: int = 14) -> np.ndarray:
        non_zero_indices = np.where(history > 0)[0]
        if len(non_zero_indices) == 0:
            return np.zeros(horizon)
        if len(non_zero_indices) == 1:
            return np.full(horizon, float(history[non_zero_indices[0]]) / len(history))

        # Initial values
        z = float(history[non_zero_indices[0]])
        p = float(non_zero_indices[0] + 1) if non_zero_indices[0] > 0 else 1.0
        last_non_zero_idx = non_zero_indices[0]

        for t in range(non_zero_indices[0] + 1, len(history)):
            if history[t] > 0:
                interval = t - last_non_zero_idx
                z = self.alpha * history[t] + (1.0 - self.alpha) * z
                p = self.alpha * interval + (1.0 - self.alpha) * p
                last_non_zero_idx = t

        forecast_rate = max(0.0, z / (p + 1e-6))
        return np.full(horizon, forecast_rate)


class SBAForecaster:
    """
    Syntetos-Boylan Approximation (SBA, 2005).
    Corrects Croston's positive bias by multiplying by (1 - alpha / 2):
        y_hat_SBA = (1 - alpha / 2) * (z_hat / p_hat)
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.croston = CrostonForecaster(alpha=alpha)

    def predict(self, history: np.ndarray, horizon: int = 14) -> np.ndarray:
        croston_fc = self.croston.predict(history, horizon)
        return (1.0 - self.alpha / 2.0) * croston_fc


class TSBForecaster:
    """
    Teunter-Syntetos-Babai (TSB, 2011).
    Updates demand probability at every period, preventing obsolescence lag:
        p_t = beta * I(y_t > 0) + (1 - beta) * p_{t-1}
        z_t = alpha * y_t + (1 - alpha) * z_{t-1} (if y_t > 0)
        y_hat = p_t * z_t
    """

    def __init__(self, alpha: float = 0.15, beta: float = 0.10):
        self.alpha = alpha
        self.beta = beta

    def predict(self, history: np.ndarray, horizon: int = 14) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(horizon)
        non_zero = history[history > 0]
        if len(non_zero) == 0:
            return np.zeros(horizon)

        z = float(np.mean(non_zero))
        p = float(len(non_zero) / len(history))

        for val in history:
            if val > 0:
                z = self.alpha * val + (1.0 - self.alpha) * z
                p = self.beta * 1.0 + (1.0 - self.beta) * p
            else:
                p = (1.0 - self.beta) * p

        rate = max(0.0, p * z)
        return np.full(horizon, rate)
