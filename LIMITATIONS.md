# Limitations, Assumptions & Scientific Honesty

This document distinguishes **implemented results** from **proposed work**, and explicitly lists
every assumption, simplification, and known limitation of the Perpetual Inventory system.

---

## 1. What Is Implemented vs. Proposed

| Component | Status |
|-----------|--------|
| Synthetic data simulator (seed=42) | ✅ Implemented |
| Statistical diagnostics (ADI/CV², distribution fitting, CUSUM) | ✅ Implemented |
| Demand forecasting (Seasonal Naive, Croston/SBA/TSB, LightGBM) | ✅ Implemented |
| Gap prediction (Heuristics, Logistic Regression, LightGBM, DLinear, GNN) | ✅ Implemented |
| Economic value engine (EV formula, sensitivity sweeps) | ✅ Implemented |
| 7-day workforce optimization (OR-Tools CP-SAT + greedy) | ✅ Implemented |
| Closed-loop bandit simulation | ✅ Implemented |
| 9-tab Streamlit dashboard | ✅ Implemented |
| 2,000-store production architecture | 📐 Proposed design only |
| Real retail data | ❌ Not available — all data is synthetic |
| Real-time POS feed / feature store integration | 📐 Proposed design only |
| Transformer / TFT deep learning models | ❌ Not implemented (see §2.2) |
| Full survival/Tobin censoring correction | ❌ Approximation only (see §3.4) |

---

## 2. Performance Metrics — Honest Reporting

> **⚠️ IMPORTANT:** Specific numeric performance metrics (PR-AUC, Precision@K, etc.)
> are **computed from the simulation pipeline** and vary slightly depending on random seed behaviour
> across Python/NumPy versions.
>
> **Do not cite hard-coded numbers as empirical facts.**
> Run `python scripts/train_gap_prediction.py` and read `data/results/gap_leaderboard.parquet`
> for the actual current values.

Previously, this document contained specific claims such as:
- PR-AUC = 0.9653 (LightGBM), 0.9645 (GNN)
- Precision@250 = 1.000

These numbers were outputs from a specific run and **should not be treated as guaranteed reproducible facts**.
The actual values depend on the simulation run, test set composition, and library versions.
The code framework to reproduce them is in place; run the pipeline to obtain current values.

---

## 3. Data & Simulation Limitations

### 3.1 Synthetic Ground Truth
- The inventory gap (`true_on_hand` - `system_on_hand`) is known **only because we simulated it**.
- Real stores cannot observe true on-hand inventory without physical counting.
- All model evaluation uses simulator ground truth — external validity is unknown.

### 3.2 Simulation Calibration Assumptions
All parameters below are **design choices, not empirically measured facts**:

| Parameter | Value | Source |
|-----------|-------|--------|
| Theft rates by category | 0.5%–5.5% | Calibration assumption |
| Spoilage rates | 0.1%–4.0% | Calibration assumption |
| DSD unrecorded share | 0%–55% | Calibration assumption |
| Recovery rate (audit effectiveness) | 75%–90% | Calibration assumption |
| Associate wage | $21/hr loaded | Calibration assumption |
| Basket abandonment | 12% of $18 | Calibration assumption |
| Substitution salvage | 45% of margin | Calibration assumption |
| False alarm penalty | $0.50/count | Calibration assumption |

### 3.3 Single-Store Prototype
- The implementation covers exactly **one store** (`STORE_0001`), 1,600 SKUs, 730 days.
- Multi-store generalization is not tested.
- Store-to-store variation in demand patterns, theft rates, or layout is not modelled.

### 3.4 Demand Censoring Approximation
- Phantom stockouts cause observed POS sales to be lower than true latent demand.
- The censoring-aware forecaster downweights stockout observations by 10× during training.
- **This is an approximation**, not a full survival/Tobin/Kaplan-Meier censoring correction.
- True censoring correction requires modelling the censoring mechanism explicitly.

### 3.5 Missing-Not-At-Random (MNAR) Labels
- The cycle-count audit policy determines which SKUs get counted.
- SKUs never selected for counting have **no ground-truth labels**.
- This creates **selection bias**: the model learns better for SKUs it tends to choose.
- Mitigation: ε-greedy exploration (simulated in `src/scale/closed_loop_bandit.py`) — **not deployed**.

---

## 4. Model Limitations

### 4.1 Perfect Count Accuracy Assumed
- The optimizer assumes associates count with 100% accuracy.
- Real-world associate counting has 5–10% error rates from fatigue, barcode failures, etc.

### 4.2 Deterministic Count & Setup Times
- Count time = fixed 1.6 min/SKU; aisle setup = fixed 4.0 min/aisle.
- Real count times vary significantly (cosmetics vs. water packs; congested aisles).

### 4.3 Discrete 24-Hour Aggregation
- All transactions are aggregated to daily resolution.
- Intraday events (mid-day theft, afternoon delivery) are not distinguishable.

### 4.4 No Real-Time Feedback Loop
- The model is trained on historical data and does not update in real time.
- Regime shifts (e.g., new theft patterns) are only detected retrospectively.

---

## 5. Deep Learning — Negative Result (Honest Reporting)

### 5.1 Aisle Graph Neural Network (GNN)
- **Hypothesis:** Discrepancy events correlate spatially across adjacent aisles.
- **Experiment:** 1-hop graph message passing over aisle neighbours.
- **Result:** GNN does **not** outperform calibrated LightGBM in this dataset.
- **Conclusion:** Individual SKU-level zero-streak features dominate spatial adjacency.
  GNN adds complexity without accuracy gain → not recommended for production.

### 5.2 Deep Sequence Models (TFT / Transformer)
- **Not implemented** in this repository.
- Heavy temporal sequence models require continuous 30-day windows × 1.12B predictions/day at scale.
- Estimated compute cost would be substantially higher than GBDT inference (exact figure not benchmarked).
- Claim of "$1,500/day vs $38.30/day" and "40× cost difference" previously cited in this document
  were **estimates based on cloud compute pricing, not measured benchmarks** and have been removed.
- If a transformer is eventually implemented, its cost and performance should be measured against
  LightGBM on the same hardware and dataset before citing any comparison.

---

## 6. Economic Model Limitations

### 6.1 Substitution Elasticity
- Customer substitution modelled as a fixed parameter (45% of margin recovered).
- Brand-loyal consumers may abandon entirely; commodity buyers substitute seamlessly.
- Actual substitution rate varies by category, brand, and consumer segment.

### 6.2 Basket Abandonment
- Basket abandonment loss modelled as 12% × $18 average basket margin.
- True impact depends on category, substitutability, and whether a customer leaves the store.

### 6.3 Chain-Wide Economic Projections
- Any chain-wide annualised value estimates (e.g., "$564M+") are extrapolations from the
  single-store simulation and depend heavily on calibration assumptions.
- **Do not cite these as empirical facts.** They are illustrative projections only.

---

## 7. Summary: Recommended Mitigations for Production

| Limitation | Impact | Recommended Mitigation |
|:-----------|:-------|:-----------------------|
| Synthetic ground truth | Limited external validity | Pilot with real store counts as labels |
| MNAR labels / selection bias | Blindness over uncounted SKUs | ε-greedy exploration budget (5%) |
| Associate count noise | Stale ledger corrections | Double-blind spot audits on high-value SKUs |
| Perfect count time assumed | Schedule overruns | Probabilistic time estimates; buffer margin |
| Demand censoring approximation | Biased forecast inputs | Full survival model for censored demand |
| Regime shifts / drift | False phantom alarms | CUSUM monitoring on gap residuals |
| DSD / off-system receipts | Unrecorded inventory inflows | EDI/ASN integration with WMS |
| Single-store generalization | Unknown store-to-store variance | Multi-store pilot before chain rollout |
