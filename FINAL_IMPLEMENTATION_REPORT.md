# Final Implementation Report: Perpetual Inventory Challenge

**Author:** Senior Applied Data Scientist  
**Assignment Reference:** Eizen AI — Predict-Then-Optimise Perpetual Inventory & Cycle-Count Scheduling  
**Repository Status:** Implemented & Tested · 28/28 Unit Tests Passing

> **Note on Metrics:** This report describes the implementation architecture and approach.
> Specific numeric results (PR-AUC, MASE, net EV, etc.) are computed dynamically by the pipeline.
> Run `python scripts/run_pipeline.py` and inspect the generated parquet artifacts for current values.
> Hard-coded performance numbers are not presented here to avoid misleading claims.

---

## 1. Executive Summary

This report documents the end-to-end implementation of the **Perpetual Inventory Challenge** across
all modules. The project addresses the core retail operations problem: **unrecorded physical stock loss
(phantom stockouts) that silences automated replenishment, causing silent sales loss, while store labor
is strictly limited to 240 minutes per day across 3 associates.**

Using a **Predict-Then-Optimise** paradigm, the system:

1. **Simulates** a realistic retail store panel (1,600 SKUs, 10 categories, 730 days, ~1.17M rows)
   incorporating bursty compound theft, perishable staling, dock receiving errors, DSD drops,
   misplaced stock, and structural regime shifts — with deterministic seed 42.

2. **Diagnoses** demand distributions using ADI/CV² intermittency classification, distribution fitting
   (Negative Binomial, ZIP), zero-streak significance testing, and CUSUM change-point detection.

3. **Forecasts** demand using Seasonal Naive, Croston/SBA/TSB, and LightGBM multi-quantile regression,
   with censoring-aware training that downweights phantom-stockout observations.

4. **Predicts** out-of-time phantom stockout probabilities using a progression of models:
   simple heuristics → Logistic Regression → Calibrated LightGBM (champion) →
   DLinear baseline → Aisle Graph Relational Model (research extension).
   The GNN does **not** outperform calibrated LightGBM (negative result, documented honestly).

5. **Values** each potential count using an economic formula combining gap probability,
   recovery rate, persistence days, demand velocity, and unit margin.

6. **Optimises** a constrained rolling 7-day associate audit schedule using OR-Tools CP-SAT,
   with a greedy heuristic as a fallback baseline.

7. **Documents** a proposed enterprise architecture for 2,000 stores and simulates ε-greedy
   closed-loop exploration to prevent feedback-loop label bias.

8. **Connects** all generated data artifacts to an interactive 9-tab Streamlit dashboard
   that displays only real computed metrics (no hard-coded numbers).

---

## 2. Module-by-Module Implementation

### Module 01 — Generative Simulation & Hidden State Panel

- **Config:** [`configs/simulation_config.yaml`](configs/simulation_config.yaml)
- **Code:** [`src/simulation/`](src/simulation/)
- **Output:** `data/simulated/store_daily_panel.parquet` (~1.17M rows)

**Physical error mechanisms:**
- Compound Poisson/Geometric theft bursts (configurable per category)
- Perishable exponential decay (Fresh Produce, Dairy, Meat, Bakery)
- POS mis-scans (wrong PLU / barcode substitution)
- Dock receiving errors (case-pack count discrepancies)
- Off-system DSD vendor drops (unrecorded inventory inflows)
- Backroom top-stock misplacement
- Regime shifts: Day 400 theft surge, Day 550 dock scanner improvement

**Tests:** 5 passing in [`tests/test_simulation.py`](tests/test_simulation.py)

---

### Module 02 — Statistical Diagnostics & Censoring Analysis

- **Doc:** [`STATISTICAL_ANALYSIS.md`](STATISTICAL_ANALYSIS.md)
- **Code:** [`src/stats/`](src/stats/)

**Key analyses:**
- ADI/CV² intermittency quadrant classification (Syntetos-Boylan framework)
- Negative Binomial and ZIP/Hurdle distribution fitting vs. Poisson
- Zero-streak Poisson significance testing (p-value = e^{-λk})
- CUSUM/EWMA change-point detection on inventory gap time series
- Censoring diagnostics: phantom stockout downward bias quantification
- Survival/hazard analysis on days-between-count distributions

**Tests:** 5 passing in [`tests/test_stats.py`](tests/test_stats.py)

---

### Module 03 — Demand Forecasting under Censoring

- **Code:** [`src/forecasting/`](src/forecasting/)
- **Script:** `scripts/train_forecasting.py`
- **Output:** `data/results/forecasting_metrics.parquet`, `data/results/demand_forecasts.parquet`

**Temporal split (no random shuffling):**
- Train: Days 1–550
- Validation: Days 551–640
- Test: Days 641–730

**Models compared:** Seasonal Naive, Simple ETS, Croston (1972), SBA (2005), TSB (2011),
LightGBM Naive (trained on raw POS), LightGBM Censoring-Aware (downweights stockout periods)

**Metrics:** MASE, RMSSE, Pinball Loss. MAPE not used (inappropriate for intermittent demand).

> **Censoring correction note:** The censoring-aware model downweights observations where
> `system_on_hand ≤ 0` by 10×. This is an approximation, not a full survival model correction.

**Tests:** 4 passing in [`tests/test_forecasting.py`](tests/test_forecasting.py)

---

### Module 04 — Inventory Gap Prediction & Deep Learning Research

- **Code:** [`src/gap_prediction/`](src/gap_prediction/)
- **Script:** `scripts/train_gap_prediction.py`
- **Output:** `data/results/gap_leaderboard.parquet`, `data/results/gap_predictions.parquet`

**Primary target:** `P(phantom_stockout)` = P(SOH > 0 AND TrueOnHand = 0)

**Feature engineering (zero future leakage):**
- Normalized zero-sales streak (streak length / mean demand)
- Days of supply (SOH / expected demand)
- Standardized forecast residual
- Days since last count / receipt
- Base demand velocity (SKU-level mean)
- Category code, aisle code, day-of-week, promotion flag
- System on-hand, consecutive zero sales

**Model progression:**
| Tier | Model | Notes |
|------|-------|-------|
| 0 | Zero-Streak Heuristic | Rule-based |
| 0 | High-Activity Index | Heuristic composite |
| 0 | Low-Recorded-Inventory | Heuristic |
| 1 | Logistic Regression | Linear baseline |
| 1/2 | LightGBM (Raw) | GBDT |
| 1/2 | LightGBM (Isotonic Calibrated) | **Champion** |
| Research | DLinear (AAAI 2023) | Linear trend+seasonal |
| Research | Aisle Graph Relational | Aisle-neighbour aggregation |

**Evaluation metrics:** PR-AUC, Precision@K, Brier score, ECE, Lift over random, Dollar capture share  
**Actual metric values:** see `data/results/gap_leaderboard.parquet` after running the pipeline.

**Research finding:** GNN does NOT outperform calibrated LightGBM. Negative result documented.

**Tests:** 4 passing in [`tests/test_gap_prediction.py`](tests/test_gap_prediction.py)

---

### Module 05 — Economic Value Engine

- **Code:** [`src/valuation/`](src/valuation/)
- **Script:** `scripts/run_valuation.py`
- **Output:** `data/results/valuation_ev_scored.parquet`, `data/results/valuation_sensitivity.parquet`

**EV formula:**
```
EV(i,d) = P(gap) × RecoveryRate × PersistenceDays × DailyDemand
        × (UnitMargin + BasketAbandonmentLoss - SubstitutionSalvage)
        - LaborCost - FalsePositivePenalty
```

All value-tree coefficients are **calibration assumptions** (see `LIMITATIONS.md`).

**Sensitivity analysis** varies: wage ($15–$35/hr), recovery rate (0.5–1.25×),
persistence (0.5–2.0×), aisle setup time (0–8 min).

**Tests:** 4 passing in [`tests/test_valuation.py`](tests/test_valuation.py)

---

### Module 06 — Workforce Optimization & 7-Day Plan

- **Code:** [`src/optimization/`](src/optimization/)
- **Script:** `scripts/run_optimization.py`
- **Output:** `data/processed/count_plan_7day.parquet`, `data/results/optimization_benchmarks.parquet`

**Constraints enforced:**
- 3 associates, **240 usable minutes/day TOTAL** across all associates (= 80 min/associate)
- 4.0-minute setup overhead per distinct aisle per associate per day
- At most 1 count per SKU over 7 days
- 90-day mandatory compliance floor (hard constraint)

**Policies benchmarked:** Random, ABC velocity, Top-K gap probability,
Greedy submodular, PuLP MILP, OR-Tools CP-SAT (primary)

**Actual benchmark values:** see `data/results/optimization_benchmarks.parquet` after running pipeline.

**Count plan schema:** `day_index`, `associate_id`, `sku_id`, `aisle_id`, `category`,
`item_count_min`, `gap_probability`, `expected_count_value_ev`, `is_compliance_mandate`

**Tests:** 4 passing in [`tests/test_optimization.py`](tests/test_optimization.py)

---

### Module 07 — Scale Architecture & Closed-Loop Exploration

- **Doc:** [`SCALE_DESIGN.md`](SCALE_DESIGN.md) — **Proposed architecture** (not deployed)
- **Code:** [`src/scale/closed_loop_bandit.py`](src/scale/closed_loop_bandit.py)
- **Output:** `data/results/closed_loop_bias_metrics.parquet`

**Scale scope:** 2,000 stores × 40,000 SKUs = 80M store-SKU rows/day

The scale design documents a proposed distributed architecture. Cost and timing figures are
**estimates** based on cloud compute pricing, not measured benchmarks.

**Closed-loop bandit simulation:** Simulates ε-greedy exploration to mitigate label bias.
Higher ε = more random audits = better recall of previously uncounted SKUs.

**Tests:** 2 passing in [`tests/test_scale.py`](tests/test_scale.py)

---

### Module 08 — Streamlit Dashboard

- **Code:** [`app/main.py`](app/main.py)

**9 tabs, all using real artifacts:**
1. Problem & Value Tree (mathematical formulation)
2. Data & Simulation (inaccuracy rates, gap distributions, error breakdown)
3. Statistical Diagnostics (ADI/CV² computed from panel, zero-streak analysis, censoring)
4. Demand Forecasting (metrics from parquet, censoring fan chart)
5. Gap Prediction (leaderboard from parquet, probability distribution)
6. Count-Plan Optimisation (real 7-day schedule, aisle clustering)
7. Live What-If Levers (parameter exploration using real EV data)
8. Scale Architecture (proposed design, bandit simulation results)
9. Results & Benchmarks (policy comparison from parquet)

No hard-coded performance numbers are displayed in the app.

---

## 3. Test Suite

```bash
pytest -q
# Expected: 28 passed
```

| File | Scope | Tests |
|------|-------|-------|
| `test_simulation.py` | Physics, determinism, compliance | 5 |
| `test_stats.py` | Distribution fitting, intermittency | 5 |
| `test_forecasting.py` | Temporal split, baselines, metrics | 4 |
| `test_gap_prediction.py` | Features, leakage guards, calibration | 4 |
| `test_valuation.py` | EV formula, sensitivity, submodular | 4 |
| `test_optimization.py` | CP-SAT constraints, compliance, greedy | 4 |
| `test_scale.py` | Bandit simulation, recall metrics | 2 |
| **Total** | | **28** |

---

## 4. End-to-End Pipeline

```bash
# Complete pipeline (all modules)
python scripts/run_pipeline.py

# Or run modules individually:
python scripts/generate_data.py
python scripts/run_statistical_analysis.py
python scripts/train_forecasting.py
python scripts/train_gap_prediction.py
python scripts/run_valuation.py
python scripts/run_optimization.py
python scripts/run_scale_and_bandits.py
```

Pipeline timing varies by hardware (typical: 2–5 minutes on a laptop CPU).

---

## 5. Scientific Integrity Notes

1. **Synthetic data:** All results are from a simulator, not real retail stores.
2. **Calibration parameters:** All economic coefficients (wages, margins, recovery rates) are
   assumptions, not empirically measured facts.
3. **GNN negative result:** Documented honestly — the aisle graph model does not improve on LightGBM.
4. **No TFT/Transformer:** Not implemented. Claims about transformer costs vs. GBDT have been removed.
5. **Performance metrics:** Actual numeric values come from the pipeline, not hard-coded claims.
6. **Scale architecture:** Documented as proposed, not deployed.
7. **Closed-loop bias:** Identified and mitigation simulated, but not deployed to production.
