"""
Test Suite: Statistical Diagnostic & Intermittency Functions (Module 02).
"""

import numpy as np
import pandas as pd
import pytest

from src.stats.intermittency import compute_adi_cv2
from src.stats.distribution_fitting import fit_poisson_aic, fit_negative_binomial_aic, fit_zero_inflated_poisson_aic
from src.stats.censoring_diagnostics import compute_zero_streak_power, tobit_censored_mean_recovery
from src.stats.change_point import PageHinkleyCUSUM
from src.stats.calibration_analysis import compute_calibration_curve


def test_intermittency_adi_cv2_smooth():
    """Smooth demand: frequent non-zero with low variance."""
    sales = np.array([5, 4, 6, 5, 5, 4, 6, 5, 5, 4])
    adi, cv2, quad = compute_adi_cv2(sales)
    assert quad == "Smooth"
    assert adi == 1.0


def test_intermittency_adi_cv2_intermittent():
    """Intermittent demand: spaced out non-zero sales."""
    sales = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    adi, cv2, quad = compute_adi_cv2(sales)
    assert quad == "Intermittent"
    assert adi == 3.0


def test_distribution_fitting_aic():
    """Negative binomial should achieve superior log-likelihood on overdispersed data."""
    rng = np.random.default_rng(42)
    # Generate overdispersed counts
    data = rng.negative_binomial(n=2.0, p=0.3, size=500)
    
    fit_p = fit_poisson_aic(data)
    fit_nb = fit_negative_binomial_aic(data)
    
    assert fit_nb["aic"] < fit_p["aic"], "Negative Binomial should beat Poisson on overdispersed data"


def test_zero_streak_power():
    """A streak of 10 zeros on a 0.3 rate item should have p < 0.05."""
    df_power = compute_zero_streak_power(daily_rates=[0.3], max_streak_days=10)
    row_10 = df_power[df_power["zero_streak_length_days"] == 10].iloc[0]
    assert row_10["significant_at_p05"] is True or row_10["significant_at_p05"] == 1


def test_cusum_alarm_trigger():
    """CUSUM should trigger an alarm when sales drop to zero on a high-velocity item."""
    cusum = PageHinkleyCUSUM(target_mean=5.0, allowance_delta=0.5, threshold_h=6.0)
    alarm_fired = False
    for _ in range(4):
        alarm, _ = cusum.step(0.0)
        if alarm:
            alarm_fired = True
            break
    assert alarm_fired is True
