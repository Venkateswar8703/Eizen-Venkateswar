"""
Probability Calibration Module for Gap Prediction:
Applies Isotonic Regression and Platt Scaling to align predicted probabilities
with empirical frequencies, minimizing Expected Calibration Error (ECE).
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ModelCalibrator:
    """Calibrates raw model probability outputs using Isotonic Regression or Platt Scaling."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None

    def fit(self, y_raw_prob: np.ndarray, y_true: np.ndarray):
        """Fits calibration curve on validation holdout set."""
        y_raw_prob = np.clip(y_raw_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_raw_prob, y_true)
        elif self.method == "platt":
            # Logistic regression on log-odds
            log_odds = np.log(y_raw_prob / (1.0 - y_raw_prob)).reshape(-1, 1)
            self.calibrator = LogisticRegression(C=1.0)
            self.calibrator.fit(log_odds, y_true)

    def calibrate(self, y_raw_prob: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated probabilities to calibrated probabilities."""
        if self.calibrator is None:
            return y_raw_prob
            
        y_raw_prob = np.clip(y_raw_prob, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(y_raw_prob)
        elif self.method == "platt":
            log_odds = np.log(y_raw_prob / (1.0 - y_raw_prob)).reshape(-1, 1)
            calibrated = self.calibrator.predict_proba(log_odds)[:, 1]
            
        return np.clip(calibrated, 0.0, 1.0)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """
    Computes Expected Calibration Error (ECE) and Brier Score.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = np.sum(mask)
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            weight = count / n
            ece += weight * abs(avg_acc - avg_conf)

    brier_score = float(np.mean((y_prob - y_true) ** 2))
    return round(float(ece), 4), round(brier_score, 4)
