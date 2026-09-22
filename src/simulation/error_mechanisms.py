"""
Error Mechanisms Module: Simulates all physical discrepancy processes that drive
the divergence between physical reality (TrueOnHand) and the ledger (SOH).

Mechanisms:
1. Shoplifting / Shrink: Bursty compound process (not single Poisson).
2. Cashier Mis-Scans / Wrong PLU: Substituted scan codes and cashier omissions.
3. Perishability & Spoilage: Unrecorded damaged/expired discards.
4. Receiving Errors: Case vs. unit miscounts at loading dock.
5. Misplaced Stock: Stock in backroom/wrong aisle, functionally absent.
6. Vendor DSD Off-System Drops: Unrecorded supplier deliveries (G < 0).
"""

from typing import Dict, Any, Tuple
import numpy as np


class ErrorMechanismEngine:
    """Generates physical discrepancies across all 10 product categories."""

    def __init__(self, config: Dict[str, Any], rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.regime_shifts = config.get("regime_shifts", [])

    def compute_daily_discrepancies(
        self,
        day: int,
        skus: list,
        true_on_hand: np.ndarray,
        pos_sales: np.ndarray,
        system_receipts: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates all error mechanisms for the given day across all SKUs.
        """
        n_skus = len(skus)
        
        theft_units = np.zeros(n_skus, dtype=int)
        mis_scan_units = np.zeros(n_skus, dtype=int)
        spoilage_units = np.zeros(n_skus, dtype=int)
        receiving_errors = np.zeros(n_skus, dtype=int)
        misplaced_units = np.zeros(n_skus, dtype=int)
        dsd_unrecorded = np.zeros(n_skus, dtype=int)

        for idx, sku in enumerate(skus):
            cat_cfg = self.config["categories"][sku.category]
            
            # 1. Apply Regime Multipliers if applicable
            theft_mult = 1.0
            rcv_mult = 1.0
            for shift in self.regime_shifts:
                if day >= shift["day"]:
                    if shift["category"] in (sku.category, "All"):
                        if shift["parameter"] == "theft_prob":
                            theft_mult *= shift["multiplier"]
                        elif shift["parameter"] == "receiving_error_prob":
                            rcv_mult *= shift["multiplier"]

            # 2. Bursty Compound Theft (Zero-inflated Geometric / Negative Binomial)
            # Physical reality: Shoplifter sweeps multiple units of desirable SKUs
            theft_prob = cat_cfg["theft_prob"] * theft_mult
            if true_on_hand[idx] > 0 and self.rng.random() < theft_prob:
                burst_mean = cat_cfg["theft_burst_mean"]
                # Geometric burst size (minimum 1 unit)
                burst_size = 1 + self.rng.geometric(p=1.0 / (burst_mean + 1e-6))
                # Can only steal what physically exists
                theft_units[idx] = min(burst_size, true_on_hand[idx])

            # 3. Cashier Mis-Scans / Wrong PLU
            # When an item is purchased, cashier might scan wrong item or miss a scan
            if pos_sales[idx] > 0:
                mis_scan_prob = cat_cfg["mis_scan_prob"]
                # Binomial trials on scanned units
                n_misses = self.rng.binomial(n=pos_sales[idx], p=mis_scan_prob)
                if n_misses > 0:
                    # In 70% of mis-scans, physical unit left store without ledger deduction (G > 0)
                    # In 30%, extra scan occurred (G < 0)
                    direction = 1 if self.rng.random() < 0.70 else -1
                    mis_scan_units[idx] = direction * n_misses

            # 4. Spoilage / Damage (Perishables)
            spoil_rate = cat_cfg.get("spoilage_rate", 0.0)
            if spoil_rate > 0 and true_on_hand[idx] > 0:
                # Spoilage occurs proportional to stock
                spoiled = self.rng.binomial(n=true_on_hand[idx], p=spoil_rate)
                spoilage_units[idx] = spoiled

            # 5. Receiving Discrepancies (Case vs. Unit / Dock short-ships)
            if system_receipts[idx] > 0:
                rcv_prob = cat_cfg["receiving_error_prob"] * rcv_mult
                if self.rng.random() < rcv_prob:
                    case_size = sku.case_pack_size
                    # Common errors: shorted 1 case (-case_size) or received 1 extra case (+case_size)
                    error_cases = self.rng.choice([-1, 1], p=[0.60, 0.40])
                    receiving_errors[idx] = error_cases * case_size

            # 6. Misplaced Stock (Stock in store but unavailable on shelf)
            misplace_prob = cat_cfg.get("misplacement_prob", 0.01)
            if true_on_hand[idx] > 2 and self.rng.random() < misplace_prob:
                max_misplace = max(1, int(true_on_hand[idx] * 0.40))
                misplaced_units[idx] = self.rng.integers(1, max_misplace + 1)

            # 7. Vendor DSD Off-System Drops (Beverages, Bread, Snacks)
            # Vendor merchandiser restocks shelf directly without store invoice entry (G < 0)
            if cat_cfg.get("dsd_share", 0.0) > 0:
                if self.rng.random() < 0.03 * cat_cfg["dsd_share"]:
                    extra_drop = self.rng.integers(3, 12)
                    dsd_unrecorded[idx] = extra_drop

        return {
            "theft_units": theft_units,
            "mis_scan_units": mis_scan_units,
            "spoilage_units": spoilage_units,
            "receiving_errors": receiving_errors,
            "misplaced_units": misplaced_units,
            "dsd_unrecorded": dsd_unrecorded,
        }
