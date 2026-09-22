"""
Probability Calibration, Reliability Diagrams, and Conformal Uncertainty Module.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes binned reliability curve and Expected Calibration Error (ECE).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    bin_confs = []
    bin_accs = []
    bin_weights = []
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = (bin_indices == b)
        count = np.sum(mask)
        if count > 0:
            avg_conf = float(np.mean(y_prob[mask]))
            avg_acc = float(np.mean(y_true[mask]))
            weight = count / n
            bin_confs.append(round(avg_conf, 3))
            bin_accs.append(round(avg_acc, 3))
            bin_weights.append(weight)
            ece += weight * abs(avg_acc - avg_conf)
        else:
            bin_confs.append(round((bins[b] + bins[b+1]) / 2, 3))
            bin_accs.append(0.0)
            bin_weights.append(0.0)

    # Brier Score = mean((y_prob - y_true)^2)
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    return {
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_weights": bin_weights,
        "expected_calibration_error_ece": round(ece, 4),
        "brier_score": round(brier_score, 4),
    }


def compute_conformal_interval_residuals(
    residuals: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Computes (1 - alpha) conformal quantile bound using split conformal / EnbPI approach.
    """
    abs_residuals = np.abs(residuals)
    n = len(abs_residuals)
    # 1 - alpha quantile with finite sample correction (n+1)(1-alpha)/n
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q_bound = float(np.quantile(abs_residuals, q_level))
    return round(q_bound, 3)
