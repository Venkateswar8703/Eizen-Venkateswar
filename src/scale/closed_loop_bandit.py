"""
Closed-Loop Feedback Bias & Bandit Exploration Simulator (Module 07):

Simulates the feedback loop problem in perpetual inventory audit policies:
1. Model predicts gap probability P(G).
2. Optimization policy allocates counts only to high-P(G) SKUs (selection bias).
3. Only audited SKUs produce true labels for retraining.
4. Uncounted SKUs with false-negative errors never get detected or retrained (feedback trap).

Evaluates epsilon-greedy exploration (epsilon in [0.0, 0.05, 0.15]) to maintain unbiased training data.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, average_precision_score


def simulate_closed_loop_retraining(
    n_skus: int = 1600,
    n_cycles: int = 8,
    audits_per_cycle: int = 150,
    epsilon_values: List[float] = [0.0, 0.05, 0.15],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulates consecutive retraining cycles under different exploration rates.
    """
    np.random.seed(seed)
    
    # Ground truth SKU population
    base_theft_rate = np.random.beta(1.5, 20.0, size=n_skus) # true propensity
    sku_velocities = np.random.gamma(2.0, 2.0, size=n_skus)
    
    results = []
    
    for eps in epsilon_values:
        # Initial beliefs / model scores
        estimated_risk = base_theft_rate + np.random.normal(0, 0.05, size=n_skus)
        estimated_risk = np.clip(estimated_risk, 0.01, 0.99)
        
        # True hidden gaps accumulate over time
        hidden_gaps = np.zeros(n_skus, dtype=bool)
        
        for cycle in range(1, n_cycles + 1):
            # 1. Generate new ground truth gaps (e.g. shrinkage event)
            new_gaps = np.random.binomial(1, base_theft_rate).astype(bool)
            hidden_gaps = hidden_gaps | new_gaps
            
            # 2. Select SKUs to audit using epsilon-greedy
            n_explore = int(round(audits_per_cycle * eps))
            n_exploit = audits_per_cycle - n_explore
            
            # Exploit top risk
            exploit_candidates = np.argsort(estimated_risk)[::-1]
            selected_exploit = exploit_candidates[:n_exploit]
            
            # Explore remaining
            remaining = np.setdiff1d(np.arange(n_skus), selected_exploit)
            selected_explore = np.random.choice(remaining, size=n_explore, replace=False) if n_explore > 0 else np.array([], dtype=int)
            
            audited_indices = np.concatenate([selected_exploit, selected_explore])
            
            # 3. Collect Labels
            observed_labels = hidden_gaps[audited_indices]
            # Discrepancies found are corrected in store
            gaps_found = observed_labels.sum()
            hidden_gaps[audited_indices] = False
            
            # Remaining undetected phantom stockouts in store
            undetected_gaps = hidden_gaps.sum()
            
            # 4. Selection Bias Metric:
            # Difference in mean velocity between audited sample and full population
            pop_mean_vel = np.mean(sku_velocities)
            audited_mean_vel = np.mean(sku_velocities[audited_indices])
            velocity_bias = abs(audited_mean_vel - pop_mean_vel) / pop_mean_vel
            
            # 5. Retrain / Update Model beliefs on audited observations
            # Without exploration, un-audited items slowly get zero weight or stale scores
            decay_mask = np.ones(n_skus, dtype=bool)
            decay_mask[audited_indices] = False
            
            # Updated beliefs
            estimated_risk[audited_indices] = (
                0.7 * estimated_risk[audited_indices] + 0.3 * observed_labels.astype(float)
            )
            # Drift / decay on unobserved items (feedback loop penalty if never counted)
            if eps == 0.0:
                estimated_risk[decay_mask] *= 0.95 # false sense of security
            else:
                estimated_risk[decay_mask] = estimated_risk[decay_mask] * 0.98 + 0.02 * 0.05
                
            # Precision and Coverage
            precision = gaps_found / len(audited_indices) if len(audited_indices) > 0 else 0.0
            total_true_gaps = gaps_found + undetected_gaps
            gap_capture_recall = gaps_found / total_true_gaps if total_true_gaps > 0 else 1.0
            
            results.append({
                "exploration_rate_eps": eps,
                "cycle": cycle,
                "audits_performed": len(audited_indices),
                "gaps_detected": int(gaps_found),
                "undetected_gaps_remaining": int(undetected_gaps),
                "audit_precision": round(precision, 4),
                "gap_capture_recall": round(gap_capture_recall, 4),
                "selection_bias_velocity_shift_pct": round(velocity_bias * 100.0, 2),
            })
            
    return pd.DataFrame(results)
