"""
Statistical Diagnostics & Analysis Package (Module 02).
"""

from .intermittency import compute_adi_cv2, categorize_store_skus
from .distribution_fitting import compare_category_distributions
from .censoring_diagnostics import compute_zero_streak_power, tobit_censored_mean_recovery
from .change_point import PageHinkleyCUSUM, simulate_cusum_arl
from .survival import compute_category_hazard_rates
from .calibration_analysis import compute_calibration_curve, compute_conformal_interval_residuals

__all__ = [
    "compute_adi_cv2",
    "categorize_store_skus",
    "compare_category_distributions",
    "compute_zero_streak_power",
    "tobit_censored_mean_recovery",
    "PageHinkleyCUSUM",
    "simulate_cusum_arl",
    "compute_category_hazard_rates",
    "compute_calibration_curve",
    "compute_conformal_interval_residuals",
]
