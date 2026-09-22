"""
Sequential Change-Point Detection & Statistical Process Control (SPC) Module.
Implements CUSUM, EWMA, and SPRT sequential anomaly detectors for gap onset.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class PageHinkleyCUSUM:
    """
    Page-Hinkley / CUSUM sequential change-point detector for downward demand shifts (gap onset).
    Tracks cumulative deviation below expected baseline mean:
        S_t = max(0, S_{t-1} + (μ_0 - δ) - X_t)
    Triggers alarm when S_t > threshold h.
    """

    def __init__(self, target_mean: float, allowance_delta: float = 0.5, threshold_h: float = 5.0):
        self.target_mean = target_mean
        self.allowance_delta = allowance_delta
        self.threshold_h = threshold_h
        self.cumulative_sum = 0.0

    def step(self, observation: float) -> Tuple[bool, float]:
        """Processes a single daily observation and returns (alarm_flag, cusum_stat)."""
        drift = (self.target_mean - self.allowance_delta) - observation
        self.cumulative_sum = max(0.0, self.cumulative_sum + drift)
        is_alarm = bool(self.cumulative_sum >= self.threshold_h)
        return is_alarm, round(self.cumulative_sum, 3)

    def reset(self):
        self.cumulative_sum = 0.0


def simulate_cusum_arl(
    in_control_lambda: float,
    threshold_h: float = 4.0,
    allowance_delta: float = 0.3,
    n_simulations: int = 1000,
    max_days: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Computes ARL_0 (Average Run Length to False Alarm) when the process is in-control (no gap).
    """
    rng = np.random.default_rng(seed)
    run_lengths = []

    for _ in range(n_simulations):
        cusum = PageHinkleyCUSUM(target_mean=in_control_lambda, allowance_delta=allowance_delta, threshold_h=threshold_h)
        for t in range(1, max_days + 1):
            obs = rng.poisson(lam=in_control_lambda)
            alarm, _ = cusum.step(obs)
            if alarm:
                run_lengths.append(t)
                break
        else:
            run_lengths.append(max_days)

    arl_0 = float(np.mean(run_lengths))
    # Daily false positive probability alpha = 1 / ARL_0
    daily_fp_rate = 1.0 / arl_0 if arl_0 > 0 else 0.0

    return {
        "in_control_lambda": in_control_lambda,
        "cusum_threshold_h": threshold_h,
        "arl_0_days_to_false_alarm": round(arl_0, 1),
        "daily_false_alarm_prob": round(daily_fp_rate, 4),
        "expected_false_alarms_per_day_1600_skus": round(1600 * daily_fp_rate, 1),
        "expected_wasted_counts_per_week": round(7 * 1600 * daily_fp_rate, 1),
    }
