"""
Evaluation Metrics Module for Gap Prediction:
Implements PR-AUC, Precision@k (k=250 daily count capacity), Lift over Random/Heuristic,
and Dollars-at-Risk Captured @k.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (PR-AUC)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return round(float(auc(recall, precision)), 4)


def compute_precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 250) -> float:
    """
    Computes Precision@k: fraction of true discrepancies among top-k ranked items.
    """
    if len(y_true) <= k:
        return round(float(np.mean(y_true)), 4)
    
    top_k_indices = np.argsort(-y_prob)[:k]
    prec_k = float(np.mean(y_true[top_k_indices]))
    return round(prec_k, 4)


def evaluate_gap_model(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    dollars_at_risk: np.ndarray = None,
    k: int = 250,
) -> Dict[str, Any]:
    """
    Computes comprehensive decision and ML metrics for a gap prediction model.
    """
    pr_auc_val = compute_pr_auc(y_true, y_prob)
    prec_k_val = compute_precision_at_k(y_true, y_prob, k=k)
    random_base_rate = float(np.mean(y_true))
    lift_over_random = prec_k_val / (random_base_rate + 1e-6)
    
    # Dollars at risk captured in top-k
    if dollars_at_risk is not None and len(dollars_at_risk) == len(y_true):
        top_k_indices = np.argsort(-y_prob)[:k]
        top_k_dollars = float(np.sum(dollars_at_risk[top_k_indices]))
        total_dollars = float(np.sum(dollars_at_risk))
        dollar_capture_pct = (top_k_dollars / (total_dollars + 1e-6)) * 100.0
    else:
        top_k_dollars = 0.0
        dollar_capture_pct = 0.0

    return {
        "Model": model_name,
        "PR-AUC": pr_auc_val,
        "Precision@250": prec_k_val,
        "Random Baseline Rate": round(random_base_rate, 4),
        "Lift over Random": round(lift_over_random, 2),
        "Dollars Captured @250 ($)": round(top_k_dollars, 2),
        "Dollar Capture Share (%)": round(dollar_capture_pct, 2),
    }
