"""
Gap Prediction Package (Module 04):
Target Generation, Point-in-Time Features, Multi-Tier ML Models,
Probability Calibration, and Precision@k Evaluation.
"""

from .targets import create_gap_targets
from .features import build_gap_features
from .calibration import ModelCalibrator, compute_ece
from .evaluate import compute_pr_auc, compute_precision_at_k, evaluate_gap_model
from .models import (
    ZeroStreakHeuristic,
    HighActivityIndexHeuristic,
    LowRecordedInventoryHeuristic,
    LogisticRegressionGapModel,
    LightGBMGapClassifier,
    DLinearGapBaseline,
    AisleGraphRelationalModel,
)

__all__ = [
    "create_gap_targets",
    "build_gap_features",
    "ModelCalibrator",
    "compute_ece",
    "compute_pr_auc",
    "compute_precision_at_k",
    "evaluate_gap_model",
    "ZeroStreakHeuristic",
    "HighActivityIndexHeuristic",
    "LowRecordedInventoryHeuristic",
    "LogisticRegressionGapModel",
    "LightGBMGapClassifier",
    "DLinearGapBaseline",
    "AisleGraphRelationalModel",
]
