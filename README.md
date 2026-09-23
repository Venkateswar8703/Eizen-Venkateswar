# Perpetual Inventory Challenge
### Eizen AI — Senior Data Scientist Take-Home Assignment

> **A Predict-Then-Optimise decision system that detects inventory record errors, estimates their economic cost, and schedules limited store-associate labor to recover the most business value.**

---

## Business Problem

Retailers maintain a *perpetual inventory* ledger — a running digital count of every SKU in every store. The ledger is updated automatically from point-of-sale scans and receiving records. But it drifts from physical reality through:

- Shoplifting and organised retail crime
- Cashier mis-scans (wrong PLU / barcode substitution)
- Perishable spoilage and damage
- Dock receiving errors (wrong case-pack count)
- Off-system vendor direct-store delivery (DSD)
- Backroom misplacement

When the ledger shows stock > 0 but the shelf is physically empty, the store has a **phantom stockout**: automated replenishment never fires, customers find empty shelves, and the store silently loses sales revenue.

A store has only **240 usable labor minutes per day** across 3 associates to audit 1,600 SKUs.

**Core question:** *Where should the store spend its limited counting labor to recover the most expected economic value?*

---

## Solution Architecture

```
DATA (730 days × 1,600 SKUs × 10 categories)
  ↓
MODULE 01: SIMULATION
  Realistic inventory dynamics with 8 error mechanisms,
  regime shifts, DSD drops, and stockout censoring
  ↓
MODULE 02: STATISTICAL ANALYSIS
  ADI/CV² intermittency classification, distribution fitting,
  zero-streak significance, change-point detection
  ↓
MODULE 03: DEMAND FORECASTING
  Seasonal Naive → Croston/SBA/TSB → LightGBM multi-quantile
  Censoring-aware training (phantom stockout downweighting)
  ↓
MODULE 04: GAP PREDICTION
  Tier 0: Heuristics (Zero-streak, Activity Index, Low-SOH)
  Tier 1: Logistic Regression
  Tier 2: LightGBM + Isotonic Calibration (Champion)
  Research: DLinear baseline + Aisle Graph Relational Model
  ↓
MODULE 05: VALUE OF A COUNT
  EV(i,d) = P(gap) × RecoveryRate × ExpectedDailyLoss × Persistence × Margin
  Sensitivity analysis across wages, recovery rates, margins
  ↓
MODULE 06: WORKFORCE OPTIMISATION
  OR-Tools CP-SAT solver (primary) + Greedy baseline
  7-day schedule: 3 associates, 240 min/day, 4-min aisle setup
  90-day compliance floor enforced
  ↓
MODULE 07: SCALE & CLOSED-LOOP BIAS
  Architecture design for 2,000 stores × 40,000 SKUs
  ε-greedy exploration to prevent feedback-loop blindness
  ↓
MODULE 08: STREAMLIT DASHBOARD
  9-tab interactive application with all real generated artifacts
```

---

## Repository Structure

```
.
├── configs/
│   └── simulation_config.yaml      # Store parameters, SKU categories, error mechanisms
├── data/
│   ├── simulated/
│   │   └── store_daily_panel.parquet   # 1.17M row panel (generated)
│   ├── processed/
│   │   ├── product_master.parquet      # Static SKU attributes (generated)
│   │   ├── historical_audit_log.parquet
│   │   └── count_plan_7day.parquet     # 7-day count schedule (generated)
│   └── results/
│       ├── demand_forecasts.parquet
│       ├── forecasting_metrics.parquet
│       ├── gap_predictions.parquet
│       ├── gap_leaderboard.parquet
│       ├── valuation_ev_scored.parquet
│       ├── valuation_sensitivity.parquet
│       ├── optimization_benchmarks.parquet
│       └── closed_loop_bias_metrics.parquet
├── src/
│   ├── simulation/     # Module 01 — Inventory physics simulator
│   ├── stats/          # Module 02 — Statistical diagnostics
│   ├── forecasting/    # Module 03 — Demand forecasting
│   ├── gap_prediction/ # Module 04 — Gap prediction models
│   ├── valuation/      # Module 05 — Economic value engine
│   ├── optimization/   # Module 06 — Workforce scheduler
│   └── scale/          # Module 07 — Closed-loop bandit simulation
├── scripts/
│   ├── generate_data.py            # Module 01: Run simulator
│   ├── run_statistical_analysis.py # Module 02: Stats diagnostics
│   ├── train_forecasting.py        # Module 03: Demand forecasting
│   ├── train_gap_prediction.py     # Module 04: Gap prediction
│   ├── run_valuation.py            # Module 05: EV engine
│   ├── run_optimization.py         # Module 06: 7-day plan
│   ├── run_scale_and_bandits.py    # Module 07: Scale simulation
│   └── run_pipeline.py             # End-to-end pipeline runner
├── app/
│   └── main.py                     # 9-tab Streamlit dashboard
├── tests/
│   ├── test_simulation.py
│   ├── test_stats.py
│   ├── test_forecasting.py
│   ├── test_gap_prediction.py
│   ├── test_valuation.py
│   ├── test_optimization.py
│   └── test_scale.py
├── PROBLEM.md                      # Mathematical formulation
├── STATISTICAL_ANALYSIS.md         # Module 02 findings
├── SCALE_DESIGN.md                 # Module 07 architecture
├── LIMITATIONS.md                  # Honest limitations & assumptions
├── AI_USAGE.md                     # AI assistance disclosure
├── data_dictionary.md              # Schema documentation
├── requirements.txt
└── README.md
```

---

## Dataset / Simulation

The project uses a **fully synthetic simulation** (seed = 42) rather than real retail data. This is deliberate: it provides ground-truth labels for the inventory gap (which real stores cannot observe without physical counts).

**Simulation parameters:**
- 1 store (`STORE_0001`)
- 1,600 SKUs across 10 categories
- 730 days of daily history (104 weeks)
- 32 aisles
- 3 associates, 240 usable labor minutes/day
- 4-minute aisle setup overhead
- 90-day mandatory recount compliance floor

**Error mechanisms simulated:**
| Mechanism | Description |
|-----------|-------------|
| Theft bursts | Compound Poisson theft events (worst in Health & Beauty, Snacks) |
| Perishable spoilage | Exponential decay in Fresh Produce, Dairy, Bakery |
| POS mis-scans | Wrong PLU / cashier barcode substitution |
| Receiving errors | Case-pack count discrepancies at dock |
| DSD unrecorded receipts | Off-system vendor drops not entered in WMS |
| Backroom misplacement | Stock present but not on shelf |
| Regime shifts | Day 400: theft surge; Day 550: dock scanner upgrade |

> **Note:** All simulation parameters (error rates, wages, margins, recovery rates) are **calibration assumptions**, not measured real-world facts. See `LIMITATIONS.md`.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate simulated data

```bash
python scripts/generate_data.py
```

### 3. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

This runs all 7 modules sequentially and generates all data artifacts.

### 4. Run tests

```bash
pytest -q
```

Expected: 28 tests passing.

### 5. Launch the Streamlit app

```bash
streamlit run app/main.py
```

---

## Running Individual Modules

```bash
# Module 01 — Simulation only
python scripts/generate_data.py

# Module 02 — Statistical analysis
python scripts/run_statistical_analysis.py

# Module 03 — Demand forecasting
python scripts/train_forecasting.py

# Module 04 — Gap prediction
python scripts/train_gap_prediction.py

# Module 05 — Economic valuation
python scripts/run_valuation.py

# Module 06 — Workforce optimization
python scripts/run_optimization.py

# Module 07 — Scale & bandit simulation
python scripts/run_scale_and_bandits.py
```

---

## Statistical Analysis (Module 02)

Key findings from the simulated panel:

- **Intermittency classification** using ADI (Average Demand Interval) and CV² (squared coefficient of variation) following the Syntetos-Boylan framework
- **Distribution fitting**: Standard Poisson rejected in favour of Negative Binomial for bursty categories (Health & Beauty, Snacks); ZIP/Hurdle appropriate for long-tail SKUs
- **Zero-streak significance**: Poisson probability test for detecting phantom stockouts from observed zero-sales runs
- **Change-point detection**: CUSUM/EWMA on inventory gap to detect regime shifts
- **Censoring diagnostics**: Quantifies downward bias in demand estimates caused by phantom stockouts (observed sales ≠ true demand when stockouts occur)

---

## Forecasting (Module 03)

**Temporal split:** Train = Days 1–550, Validation = 551–640, Test = 641–730. No random shuffling.

**Models compared:**
- Seasonal Naive (7-day)
- Simple ETS
- Croston's (1972) — for intermittent demand
- Syntetos-Boylan Approximation (SBA)
- Teunter-Syntetos-Babai (TSB)
- LightGBM multi-quantile (q10, q50, q90) — naive and censoring-aware

**Censoring-aware training:** Observations where `system_on_hand ≤ 0` are downweighted by 10× during LightGBM training to reduce phantom stockout downward bias.

**Metrics:** MASE, RMSSE, Pinball Loss. MAPE is not used (inappropriate for zero-inflated/intermittent demand).

Actual numeric results are generated from code and stored in `data/results/forecasting_metrics.parquet`.

---

## Gap Prediction (Module 04)

**Target:** `P(phantom_stockout)` — probability that `system_on_hand > 0` but `true_on_hand = 0`

**Feature engineering (zero future leakage):**
- Normalized zero-sales streak (streak / mean demand)
- Days of supply (SOH / expected demand)
- Forecast residual (standardized)
- Days since last count / last receipt
- Base demand velocity
- Category, aisle, day-of-week, promotion flag
- System on-hand, consecutive zero sales

**Model progression:**
| Tier | Model | Notes |
|------|-------|-------|
| 0 | Zero-Streak Heuristic | Rule-based baseline |
| 0 | High-Activity Index | Heuristic |
| 0 | Low-Recorded-Inventory | Heuristic |
| 1 | Logistic Regression | Regularized linear baseline |
| 1/2 | LightGBM (Raw) | GBDT with class balancing |
| 1/2 | LightGBM (Isotonic Calibrated) | **Champion model** |
| Research | DLinear Baseline (AAAI 2023) | Linear trend+seasonal decomposition |
| Research | Aisle Graph Relational Model | SKU-neighbor message passing |

**Evaluation:** PR-AUC, Precision@K, Brier score, ECE (calibration), Lift over random, Dollar capture share.  
Actual metrics are computed from code and stored in `data/results/gap_leaderboard.parquet`.

> **Research finding:** The GNN/spatial model does not outperform calibrated LightGBM. Individual SKU-level zero-streak features dominate spatial aisle context. See `LIMITATIONS.md`.

---

## Value of a Count (Module 05)

Expected net value of counting SKU `i` on day `d`:

```
EV(i,d) = P(gap) × RecoveryRate(category)
        × PersistenceDays × DailyDemand
        × (UnitMargin + BasketAbandonmentLoss - SubstitutionSalvage)
        - LaborCost(count_duration_min)
        - FalsePositivePenalty
```

**Calibration assumptions** (not empirical facts):
- Recovery rate: 75–90% by category
- Basket abandonment: 12% × $18 avg basket margin
- Substitution salvage: 45% of margin
- Associate wage: $21/hr loaded
- False alarm penalty: $0.50

Sensitivity analysis varies these parameters and stores results in `data/results/valuation_sensitivity.parquet`.

---

## Workforce Optimisation (Module 06)

**7-day integer programming problem:**

```
Maximise: Σ EV(i,d) × x(i,k,d) − Σ AisleSetupCost × y(a,k,d)

Subject to:
  - x(i,k,d) ≤ y(aisle(i),k,d)                   [aisle linking]
  - Σ items·time + Σ aisles·setup ≤ 240 min/day   [labor budget]
  - Σ x(i,k,d) ≤ 1 over all k,d                  [no duplicate counts]
  - Σ x(i,k,d) = 1 if days_since_count + 7 ≥ 90  [compliance floor]
```

**Solvers compared:**
1. Random selection (baseline)
2. ABC velocity ranking
3. Top-K gap probability
4. Greedy submodular (aisle-clustering heuristic)
5. PuLP MILP (LP relaxation)
6. **OR-Tools CP-SAT (primary)**

**Artifact:** `data/processed/count_plan_7day.parquet`

Fields: `day_index`, `associate_id`, `sku_id`, `aisle_id`, `category`, `item_count_min`, `gap_probability`, `expected_count_value_ev`, `is_compliance_mandate`

---

## Scaling Approach (Module 07) — Proposed Architecture

> **IMPORTANT:** The scale architecture described in `SCALE_DESIGN.md` is a **proposed design** for production deployment. Only the single-store prototype is actually implemented.

At 2,000 stores × 40,000 SKUs = 80M store-SKU scoring rows per day, the system requires:
- Distributed GBDT inference (Ray/Treelite)
- Parallel store-level CP-SAT workers
- Point-in-time feature store (Feast/Hopsworks)
- Model monitoring and drift detection
- **Closed-loop bias mitigation**: The model chooses what gets counted; those counts create future labels. ε-greedy exploration (5% random audits) prevents feedback-loop blindness. This is simulated in `src/scale/closed_loop_bandit.py`.

---

## Streamlit Application

```bash
streamlit run app/main.py
```

**9 tabs:**
1. Problem & Value Tree — mathematical formulation
2. Data & Hidden Simulation — simulator mechanics, gap distributions
3. Statistical Diagnostics — intermittency quadrants, zero-streak analysis
4. Demand Forecasting — model comparison leaderboard, censoring fan chart
5. Gap Prediction & ML — model leaderboard from real artifacts
6. Count-Plan Optimisation — actual 7-day schedule from CP-SAT
7. Live What-If Levers — interactive parameter exploration
8. Scale to 2,000 Stores — proposed architecture, bandit simulation
9. Executive P&L ROI — benchmark policy comparison

The app requires all pipeline artifacts to be generated first (`python scripts/run_pipeline.py`).

---

## Testing

```bash
pytest -q
```

| Test File | Scope |
|-----------|-------|
| `test_simulation.py` | Physics conservation, determinism, compliance floor |
| `test_stats.py` | Distribution fitting, intermittency, censoring diagnostics |
| `test_forecasting.py` | Temporal split, baseline models, metrics |
| `test_gap_prediction.py` | No-leakage features, calibration, PR-AUC |
| `test_valuation.py` | EV formula, sensitivity, submodular selection |
| `test_optimization.py` | CP-SAT constraints, compliance, greedy baseline |
| `test_scale.py` | Closed-loop bandit simulation |

---

## Reproducibility

- All randomness uses `numpy.random.default_rng(seed=42)` or `random_state=42`
- All paths are relative to the project root
- No external data dependencies
- Fresh clone workflow: `pip install -r requirements.txt` → `python scripts/run_pipeline.py` → `pytest -q` → `streamlit run app/main.py`

---

## Limitations

Key limitations are documented in [`LIMITATIONS.md`](LIMITATIONS.md). Summary:

1. **Synthetic ground truth** — inventory gaps are known only because we simulated them. Real stores cannot observe true on-hand without physical counts.
2. **Simulation assumptions** — all error rates, recovery rates, wages, and margins are calibration parameters chosen to be realistic, not empirically measured.
3. **Single-store prototype** — the implementation covers one store. Scale architecture is proposed, not implemented.
4. **Demand censoring** — the censoring-aware forecaster downweights stockout periods but does not implement full survival/Tobin censoring correction.
5. **Missing-not-at-random labels** — the model chooses which SKUs get counted, creating selection bias in future training labels.
6. **Perfect count accuracy assumed** — associate audits are modelled as 100% accurate.
7. **GNN negative result** — the Aisle Graph Relational Model does not outperform calibrated LightGBM. Documented honestly.

---

## Project Narrative

> "Inventory records can disagree with physical reality. We simulate and detect those gaps. We forecast demand while recognising sales censoring. We estimate which gaps are economically material. We convert predictions into expected value. We optimise limited counting labor. We produce a practical 7-day count plan. We evaluate using decision and business metrics. We explicitly account for uncertainty, selection bias, and limitations."