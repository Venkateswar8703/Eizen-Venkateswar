"""
Core Inventory State-Space Simulator for Store 0001.

Simulates 104 weeks (730 days) of daily physical and perpetual ledger dynamics
across 1,600 SKUs and 10 categories, capturing the full generative physics of
phantom inventory, censoring, bursty shrinkage, and cycle counts.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import yaml

from .error_mechanisms import ErrorMechanismEngine
from .censoring import apply_pos_censoring
from .audit_policy import LegacyABCAuditPolicy


@dataclass
class SKUAttributes:
    sku_id: str
    category: str
    aisle_id: int
    unit_price: float
    unit_cost: float
    unit_margin: float
    case_pack_size: int
    target_shelf_capacity: int
    reorder_point: int
    order_up_to: int
    count_duration_min: float
    is_locked_case: bool
    is_dsd: bool
    is_perishable: bool
    abc_class: str
    mean_daily_demand: float


class InventorySimulator:
    """Generates the full ground-truth and observable inventory datasets."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.seed = self.config.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        self.num_days = self.config.get("num_days", 730)
        self.store_id = self.config.get("store_id", "STORE_0001")
        
        self.skus: List[SKUAttributes] = []
        self._initialize_sku_master()
        
        self.error_engine = ErrorMechanismEngine(self.config, self.rng)
        self.audit_policy = LegacyABCAuditPolicy(
            daily_budget_min=self.config.get("daily_labor_minutes", 240.0),
            aisle_setup_min=self.config.get("aisle_setup_time_min", 4.0),
            compliance_max_days=self.config.get("compliance_max_days", 90),
        )

    def _initialize_sku_master(self):
        """Initializes 1,600 realistic retail SKUs across 10 categories."""
        sku_counter = 1
        aisle_map = {
            "Fresh Produce": [1, 2, 3],
            "Bakery": [4, 5],
            "Meat & Seafood": [6, 7, 8],
            "Dairy & Eggs": [9, 10, 11],
            "Frozen": [12, 13, 14, 15],
            "Packaged Grocery": [16, 17, 18, 19, 20, 21, 22, 23],
            "Beverages": [24, 25, 26],
            "Snacks & Confectionery": [27, 28],
            "Household & Cleaning": [29, 30],
            "Health & Beauty": [31, 32],
        }

        for cat_name, cat_cfg in self.config["categories"].items():
            sku_count = cat_cfg["sku_count"]
            aisles = aisle_map.get(cat_name, [1])
            
            for _ in range(sku_count):
                sku_id = f"SKU_{sku_counter:04d}"
                aisle = int(self.rng.choice(aisles))
                
                p_min, p_max = cat_cfg["unit_price_range"]
                price = round(float(self.rng.uniform(p_min, p_max)), 2)
                
                m_min, m_max = cat_cfg["margin_rate_range"]
                margin_rate = float(self.rng.uniform(m_min, m_max))
                cost = round(price * (1.0 - margin_rate), 2)
                margin = round(price - cost, 2)
                
                d_min, d_max = cat_cfg["mean_demand_range"]
                mean_demand = float(self.rng.uniform(d_min, d_max))
                
                # Case pack and inventory replenishment levels
                case_size = int(self.rng.choice([4, 6, 8, 12, 24]))
                capacity = max(case_size * 2, int(mean_demand * 5 + case_size))
                reorder_pt = max(2, int(mean_demand * 2.5))
                order_up = max(reorder_pt + case_size, capacity)
                
                # Duration and flags
                count_time = cat_cfg["count_time_min"] + self.rng.uniform(-0.2, 0.3)
                is_locked = (cat_name == "Health & Beauty" and price > 18.0)
                is_dsd = (self.rng.random() < cat_cfg.get("dsd_share", 0.0))
                is_perishable = (cat_name in ["Fresh Produce", "Dairy & Eggs", "Meat & Seafood", "Bakery"])
                
                # ABC velocity class
                if mean_demand >= 5.0:
                    abc = "A"
                elif mean_demand >= 1.5:
                    abc = "B"
                else:
                    abc = "C"

                sku = SKUAttributes(
                    sku_id=sku_id,
                    category=cat_name,
                    aisle_id=aisle,
                    unit_price=price,
                    unit_cost=cost,
                    unit_margin=margin,
                    case_pack_size=case_size,
                    target_shelf_capacity=capacity,
                    reorder_point=reorder_pt,
                    order_up_to=order_up,
                    count_duration_min=round(count_time, 2),
                    is_locked_case=is_locked,
                    is_dsd=is_dsd,
                    is_perishable=is_perishable,
                    abc_class=abc,
                    mean_daily_demand=round(mean_demand, 3),
                )
                self.skus.append(sku)
                sku_counter += 1

    def run_simulation(self) -> Dict[str, pd.DataFrame]:
        """
        Executes the 730-day simulation loop and returns:
        - daily_panel: Full panel dataframe (1.17M rows)
        - audit_logs: Table of historical cycle-count events
        - product_master: Table of static SKU metadata
        """
        n_skus = len(self.skus)
        
        # State Arrays
        true_on_hand = np.zeros(n_skus, dtype=int)
        system_on_hand = np.zeros(n_skus, dtype=int)
        days_since_count = np.zeros(n_skus, dtype=int)
        days_since_receipt = np.zeros(n_skus, dtype=int)
        consecutive_zeros = np.zeros(n_skus, dtype=int)
        
        # Outstanding pipeline orders: list of (arrival_day, sku_idx, units)
        pipeline_orders: List[Dict[str, Any]] = []

        # Warm start initial inventory
        for i, sku in enumerate(self.skus):
            initial_stock = int(sku.order_up_to * self.rng.uniform(0.6, 1.0))
            true_on_hand[i] = initial_stock
            system_on_hand[i] = initial_stock
            days_since_count[i] = int(self.rng.integers(1, 45))
            days_since_receipt[i] = int(self.rng.integers(1, 5))

        daily_records = []
        audit_records = []
        start_date = pd.Timestamp("2024-01-01")

        for day in range(1, self.num_days + 1):
            current_date = start_date + pd.Timedelta(days=day - 1)
            dow = current_date.dayofweek
            is_weekend = (dow in [5, 6])
            
            # Weekend & Holiday Demand Multipliers
            dow_multiplier = 1.35 if is_weekend else (1.15 if dow == 4 else 0.90)
            is_holiday = (current_date.month in [11, 12] and current_date.day in [24, 25, 31])
            holiday_mult = 1.60 if is_holiday else 1.0

            # 1. Pipeline Receipts Arrival (Lead time = 2 days)
            system_receipts = np.zeros(n_skus, dtype=int)
            remaining_orders = []
            for order in pipeline_orders:
                if order["arrival_day"] <= day:
                    idx = order["sku_idx"]
                    units = order["units"]
                    system_receipts[idx] += units
                else:
                    remaining_orders.append(order)
            pipeline_orders = remaining_orders

            # Physical receipts into TrueOnHand (incorporates receiving errors)
            true_receipts = system_receipts.copy()

            # 2. Demand Generation & Price Promotions
            promo_flags = np.zeros(n_skus, dtype=bool)
            selling_prices = np.zeros(n_skus, dtype=float)
            latent_demands = np.zeros(n_skus, dtype=int)

            for i, sku in enumerate(self.skus):
                # 5% chance of promo circular on any given week
                is_promo = (self.rng.random() < 0.05 and day % 7 == 0) or False
                promo_flags[i] = is_promo
                
                promo_discount = 0.80 if is_promo else 1.0
                selling_prices[i] = round(sku.unit_price * promo_discount, 2)
                
                promo_lift = 2.0 if is_promo else 1.0
                effective_lambda = sku.mean_daily_demand * dow_multiplier * holiday_mult * promo_lift
                
                # Draw demand: NegBin for bursty categories, Poisson for standard
                if sku.category in ["Health & Beauty", "Snacks & Confectionery"]:
                    # Overdispersed Negative Binomial
                    p_nb = 0.60
                    r_nb = max(1.0, effective_lambda * p_nb / (1.0 - p_nb))
                    latent_demands[i] = int(self.rng.negative_binomial(n=r_nb, p=p_nb))
                else:
                    latent_demands[i] = int(self.rng.poisson(lam=effective_lambda))

            # 3. Simulate Physical Error Mechanisms (Theft, Spoilage, Mis-scans)
            discrepancies = self.error_engine.compute_daily_discrepancies(
                day=day,
                skus=self.skus,
                true_on_hand=true_on_hand + true_receipts,
                pos_sales=latent_demands,
                system_receipts=system_receipts,
            )

            # Update TrueOnHand with physical deliveries, DSD drops, and physical receiving errors
            true_on_hand += true_receipts + discrepancies["receiving_errors"] + discrepancies["dsd_unrecorded"]
            true_on_hand = np.maximum(0, true_on_hand)

            # Deduct physical spoilage & theft before POS trading begins
            true_on_hand = np.maximum(0, true_on_hand - discrepancies["theft_units"] - discrepancies["spoilage_units"])

            # 4. Point-of-Sale Trading with Physical Censoring
            pos_sales, true_lost_sales = apply_pos_censoring(
                latent_demand=latent_demands,
                true_on_hand=true_on_hand,
                misplaced_units=discrepancies["misplaced_units"],
            )

            # Physical inventory is consumed by sales and cashier mis-scans
            true_on_hand = np.maximum(0, true_on_hand - pos_sales)

            # 5. Cycle Counts Execution (Legacy ABC Policy)
            counted_indices = self.audit_policy.select_skus_to_count(
                day=day,
                skus=self.skus,
                system_on_hand=system_on_hand,
                days_since_count=days_since_count,
                rng=self.rng,
            )

            audit_flags = np.zeros(n_skus, dtype=bool)
            audit_counts = np.full(n_skus, np.nan)
            system_adjustments = np.zeros(n_skus, dtype=int)

            for idx in counted_indices:
                sku = self.skus[idx]
                audit_flags[idx] = True
                
                # Associate counts physical stock (and locates misplaced stock)
                physical_found = int(true_on_hand[idx])
                audit_counts[idx] = physical_found
                
                # Adjustment posted = physical_found - SOH
                pre_soh = system_on_hand[idx]
                adjustment = physical_found - pre_soh
                system_adjustments[idx] = adjustment
                
                # Reset elapsed counters
                days_since_count[idx] = 0

                # Log audit record
                audit_records.append({
                    "audit_id": f"AUD_{day:04d}_{sku.sku_id}",
                    "date": current_date,
                    "sku_id": sku.sku_id,
                    "category": sku.category,
                    "aisle_id": sku.aisle_id,
                    "associate_id": f"ASSOC_{(idx % 3) + 1}",
                    "duration_minutes": sku.count_duration_min,
                    "pre_count_soh": pre_soh,
                    "physical_count": physical_found,
                    "posted_discrepancy": adjustment,
                    "policy_type": "COMPLIANCE_90D" if days_since_count[idx] >= 90 else "LEGACY_ABC",
                })

            # 6. Perpetual Inventory (SOH) Book Update
            # SOH(t) = SOH(t-1) + Receipts_sys - (POS + MisScans) + Adjustments
            effective_pos_deduction = np.maximum(0, pos_sales - discrepancies["mis_scan_units"])
            system_on_hand = system_on_hand + system_receipts - effective_pos_deduction + system_adjustments

            # 7. Automated Replenishment Orders (Triggered strictly by SOH)
            for i, sku in enumerate(self.skus):
                # Count current pipeline orders for this SKU
                on_order_units = sum(o["units"] for o in pipeline_orders if o["sku_idx"] == i)
                inventory_position = system_on_hand[i] + on_order_units
                
                if inventory_position <= sku.reorder_point:
                    order_qty = ((sku.order_up_to - inventory_position) // sku.case_pack_size) * sku.case_pack_size
                    if order_qty <= 0:
                        order_qty = sku.case_pack_size
                    
                    pipeline_orders.append({
                        "arrival_day": day + 2,  # 2-day lead time
                        "sku_idx": i,
                        "units": int(order_qty),
                    })

            # Update day counters
            days_since_count += 1
            days_since_receipt = np.where(system_receipts > 0, 0, days_since_receipt + 1)
            consecutive_zeros = np.where(pos_sales == 0, consecutive_zeros + 1, 0)

            # Record daily state for all SKUs
            for i, sku in enumerate(self.skus):
                daily_records.append({
                    "date": current_date,
                    "day_index": day,
                    "sku_id": sku.sku_id,
                    "category": sku.category,
                    "aisle_id": sku.aisle_id,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "selling_price": selling_prices[i],
                    "promo_flag": promo_flags[i],
                    # Observable Features
                    "pos_sales": pos_sales[i],
                    "system_on_hand": system_on_hand[i],
                    "system_receipts": system_receipts[i],
                    "audit_event": audit_flags[i],
                    "audit_recorded_on_hand": audit_counts[i],
                    "system_adjustment": system_adjustments[i],
                    "days_since_last_count": days_since_count[i],
                    "days_since_last_receipt": days_since_receipt[i],
                    "consecutive_zero_sales": consecutive_zeros[i],
                    # Ground Truth (Hidden Layer)
                    "latent_demand": latent_demands[i],
                    "true_on_hand": true_on_hand[i],
                    "inventory_gap": system_on_hand[i] - true_on_hand[i],
                    "is_phantom_stockout": bool(system_on_hand[i] > 0 and true_on_hand[i] == 0),
                    "true_lost_sales": true_lost_sales[i],
                    "theft_units": discrepancies["theft_units"][i],
                    "mis_scan_units": discrepancies["mis_scan_units"][i],
                    "spoilage_units": discrepancies["spoilage_units"][i],
                    "receiving_discrepancy": discrepancies["receiving_errors"][i],
                    "misplaced_units": discrepancies["misplaced_units"][i],
                })

        # Assemble Output DataFrames
        df_panel = pd.DataFrame(daily_records)
        df_audits = pd.DataFrame(audit_records)
        df_product_master = pd.DataFrame([s.__dict__ for s in self.skus])

        return {
            "daily_panel": df_panel,
            "audit_logs": df_audits,
            "product_master": df_product_master,
        }
