"""
Audit Policy Module: Simulates historical cycle-counting practices.

Generates realistic Missing-Not-At-Random (MNAR) audit labels by simulating
the store's pre-existing Legacy ABC rotation policy + 90-day SOX compliance floor.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class LegacyABCAuditPolicy:
    """
    Implements standard retail legacy cycle-counting heuristic:
    1. Prioritizes items reaching the 90-day compliance floor.
    2. Prioritizes high-velocity Class A items and items with high SOH.
    3. Respects daily store labor constraint (240 minutes).
    """

    def __init__(
        self,
        daily_budget_min: float = 240.0,
        aisle_setup_min: float = 4.0,
        compliance_max_days: int = 90,
    ):
        self.daily_budget_min = daily_budget_min
        self.aisle_setup_min = aisle_setup_min
        self.compliance_max_days = compliance_max_days

    def select_skus_to_count(
        self,
        day: int,
        skus: list,
        system_on_hand: np.ndarray,
        days_since_count: np.ndarray,
        rng: np.random.Generator,
    ) -> List[int]:
        """
        Selects SKU indices to audit today using legacy heuristic while obeying labor budget.
        """
        n_skus = len(skus)
        
        # 1. Check Compliance Floor (Hard priority)
        must_count = [i for i in range(n_skus) if days_since_count[i] >= self.compliance_max_days]
        
        # 2. Heuristic Priority Score (Legacy ABC + High SOH)
        scores = np.zeros(n_skus)
        for i, sku in enumerate(skus):
            # Class A velocity boost + High SOH bias
            velocity_weight = 3.0 if sku.abc_class == "A" else (1.5 if sku.abc_class == "B" else 1.0)
            soh_weight = np.log1p(max(0, system_on_hand[i]))
            recency_weight = days_since_count[i] / 30.0
            scores[i] = velocity_weight * (1.0 + soh_weight) * (1.0 + recency_weight)

        # Sort remaining SKUs by legacy score descending
        candidate_indices = np.argsort(-scores).tolist()
        
        # Combine must_count first, then candidates
        ordered_indices = []
        seen = set()
        for idx in must_count:
            ordered_indices.append(idx)
            seen.add(idx)
        for idx in candidate_indices:
            if idx not in seen:
                ordered_indices.append(idx)
                seen.add(idx)

        # 3. Knapsack / Labor Budget Simulation with Aisle Setup overhead
        selected_skus = []
        visited_aisles = set()
        consumed_minutes = 0.0

        for idx in ordered_indices:
            sku = skus[idx]
            aisle = sku.aisle_id
            
            # Additional setup cost if entering a new aisle
            incremental_setup = self.aisle_setup_min if aisle not in visited_aisles else 0.0
            item_duration = sku.count_duration_min
            total_incremental = item_duration + incremental_setup

            if consumed_minutes + total_incremental <= self.daily_budget_min:
                selected_skus.append(idx)
                visited_aisles.add(aisle)
                consumed_minutes += total_incremental
            else:
                # Labor budget exhausted
                break

        return selected_skus
