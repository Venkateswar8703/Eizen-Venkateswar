# ASSIGNMENT_REQUIREMENTS.md: Master Requirements Specification

This document extracts every single requirement, evaluation criterion, constraint, and deliverable from the official **Perpetual Inventory Challenge (Eizen AI)** specification document.

---

## 1. Problem Context & Core Operational Parameters
* **Primary Store Scope:** Single store (`STORE_0001`) out of a 2,000-store retail chain.
* **Merchandise Catalog:** 10 Categories, approximately 1,600 SKUs.
  - *Fresh Produce:* 120 SKUs (Weight-based, high spoilage, PLU substitutions).
  - *Dairy & Eggs:* 90 SKUs (Short shelf life, high velocity, date-coded, DSD share).
  - *Meat & Seafood:* 110 SKUs (Weight-based, high unit value, pre-pack & counter).
  - *Bakery:* 70 SKUs (In-store production, unrecorded waste/staling).
  - *Frozen:* 150 SKUs (Long shelf life, freezer capacity constraint).
  - *Packaged Grocery:* 400 SKUs (Long tail, slow movers, high SKU count).
  - *Beverages:* 180 SKUs (Direct store delivery [DSD], vendor-managed, off-system receipts).
  - *Snacks & Confectionery:* 160 SKUs (High impulse theft, off-planogram displays).
  - *Household & Cleaning:* 140 SKUs (Bulky, backroom overflow).
  - *Health & Beauty (HBA):* 180 SKUs (Highest theft rate, locked security cases, high unit price).
* **Time Horizon:** 104 weeks (730 days) of daily history ($\approx 1.17\text{M}$ store-SKU-day rows).
* **Operational Planning Horizon:** Rolling 7-day count schedule (1–14 days multi-horizon).
* **Store Labor Budget:** 3 count-capable associates, total 240 usable minutes/day (4 hours total).
* **Per-SKU Count Duration:** 0.8 to 3.5 minutes depending on category, facings, and backroom status.
* **Aisle Setup Overhead:** 4.0 minutes per aisle visited per day (ladder, scanner sync, cart transit). Non-separable!
* **Compliance Floor:** Every SKU must be audited at least once every 90 days regardless of model score.

---

## 2. Module-by-Module Requirements Checklist

### Module 00: Problem Formulation (20% Weight)
- [ ] **REQ-00.1:** Problem statement consumable and fundable by a retail CFO.
- [ ] **REQ-00.2:** Formal mathematical notation for sets, indices, state variables, random variables, parameters, and decision variables.
- [ ] **REQ-00.3:** Unambiguous prediction target definition ($P(|G| \ge \theta)$, $P(\text{phantom OOS})$, $\mathbb{E}[G]$, unconstrained demand $\hat{D}$).
- [ ] **REQ-00.4:** Explicit separation between prediction target and count decision variable.
- [ ] **REQ-00.5:** Objective function expressed in currency ($\$$).
- [ ] **REQ-00.6:** Value Tree from inventory gap to P&L with explicit grounded coefficients on every edge (direct margin, basket abandonment, substitution salvage, carrying cost, labor wage, false alarm penalty).
- [ ] **REQ-00.7:** 3-Level Metric Hierarchy: Level 1 (Model: PR-AUC, ECE), Level 2 (Decision: $\$ / \text{hr}$, Precision@$k$), Level 3 (Business: OSA, Shrink %, Net Margin).
- [ ] **REQ-00.8:** Counterfactual evaluation problem formulated explicitly (what happens to uncounted items).
- [ ] **REQ-00.9:** Prospective field trial experimental design (randomization unit, treatment vs control, power calculation).
- [ ] **REQ-00.10:** Distinction between record inaccuracy ($G \ne 0$) and On-Shelf Availability (OSA).
- [ ] **REQ-00.11:** Assumptions Register (Verifiable, Unverifiable, Deliberately Simplifying).
- [ ] **REQ-00.12:** Critical self-critique naming at least two ways the formulation could be wrong.

### Module 01: Data & Hidden Inventory Simulation (10% Weight)
- [ ] **REQ-01.1:** Grounded in real retail demand dynamics (M5 / Favorita statistical properties).
- [ ] **REQ-01.2:** Hidden state-space simulation layer generating physical ground truth ($\text{TrueOnHand}, \text{latent demand}, \text{true lost sales}$).
- [ ] **REQ-01.3:** Bursty compound theft process (zero-inflated Geometric/Pareto bursts, sweeping 3–8 units at once, not Poisson).
- [ ] **REQ-01.4:** Cashier mis-scans and PLU substitutions ($\pm G$).
- [ ] **REQ-01.5:** Unrecorded damage and perishable spoilage ($G > 0$).
- [ ] **REQ-01.6:** Receiving errors at delivery (case-pack vs. unit miscounts $\pm G$).
- [ ] **REQ-01.7:** Misplaced inventory (physically present in backroom but absent from shelf).
- [ ] **REQ-01.8:** Vendor DSD off-system drops ($G < 0$).
- [ ] **REQ-01.9:** Censored POS observation layer ($S(i,t) = \min(D(i,t), \text{Stock}(i,t))$).
- [ ] **REQ-01.10:** Missing-Not-At-Random (MNAR) historical audit label generation via Legacy ABC rotation + 90-day compliance floor.
- [ ] **REQ-01.11:** Deterministic random seeding and reproducibility.
- [ ] **REQ-01.12:** Parameterized configuration file (`configs/simulation_config.yaml`).
- [ ] **REQ-01.13:** Comprehensive data schema documentation (`data_dictionary.md`).
- [ ] **REQ-01.14:** Structural regime shifts injected (Day 400 HBA shrink surge, Day 550 dock scanner upgrade).
- [ ] **REQ-01.15:** Unit and physics test suite (mass conservation, non-negativity, censoring bounds, compliance).

### Module 02: Statistical Modelling & Censoring Diagnostics (15% Weight)
- [ ] **REQ-02.1:** Discrete count likelihood comparisons (Poisson, NegBin, ZIP, Hurdle, Tweedie) across all 10 categories with AIC/BIC tests.
- [ ] **REQ-02.2:** Demonstration of MAPE mathematical failure on intermittent zeros and substitution with MASE, RMSSE, Pinball loss.
- [ ] **REQ-02.3:** Syntetos-Boylan ADI/$CV^2$ categorization (Smooth, Erratic, Intermittent, Lumpy) and conditions where classical Croston/SBA beats ML.
- [ ] **REQ-02.4:** Formal censoring model and latent demand distribution recovery (Tobit / EM / availability mask) + downward bias measurement.
- [ ] **REQ-02.5:** Identification challenge analysis: 3 days zero sales with book stock of 8 (ranked explanations + separating data features).
- [ ] **REQ-02.6:** Statistical power calculation of zero-sales streak test for slow movers ($\lambda = 0.3$/day) and informative streak horizon.
- [ ] **REQ-02.7:** Sequential change-point gap detection (CUSUM, EWMA, SPRT) with Average Run Length (ARL) and wasted count calculations.
- [ ] **REQ-02.8:** Multiple testing / False Discovery Rate (FDR / Benjamini-Hochberg) critique vs. Top-$k$ capacity budget.
- [ ] **REQ-02.9:** Discrete-time survival hazard model for time-to-gap onset across categories and count-frequency policy derivation.
- [ ] **REQ-02.10:** MNAR historical label bias corrections (Inverse Propensity Weighting, compliance audit probes).
- [ ] **REQ-02.11:** Treatment effect handling in training data (ledger reset $\text{SOH} \leftarrow \text{Count}$).
- [ ] **REQ-02.12:** Field experiment causal design for count plan sales lift.
- [ ] **REQ-02.13:** Probability calibration analysis (Reliability diagram, Platt scaling, Isotonic regression, ECE, Brier score).
- [ ] **REQ-02.14:** Conformal prediction intervals for gap magnitude adapted for time-series autocorrelation (EnbPI).
- [ ] **REQ-02.15:** Theorem and proof/counterexample that category-specific miscalibration destroys ranking in $\sum p_i v_i x_i$.
- [ ] **REQ-02.16:** Hierarchical forecast reconciliation (MinT) across SKU $\rightarrow$ Subcategory $\rightarrow$ Category $\rightarrow$ Store.

### Module 03: Demand Forecasting Under Censoring (10% Weight)
- [ ] **REQ-03.1:** Statistical and classical intermittent baselines: Seasonal Naive, ETS, Croston, Syntetos-Boylan Approximation (SBA), TSB.
- [ ] **REQ-03.2:** Strong LightGBM / GBDT baseline with rolling, lag, calendar, promo, price, and category features.
- [ ] **REQ-03.3:** Censoring handling: Naive sales training vs. Uncensored subset / availability masked training.
- [ ] **REQ-03.4:** Quantile demand forecasting ($q_{0.10}, q_{0.50}, q_{0.90}$) evaluated with Pinball Loss.
- [ ] **REQ-03.5:** Rolling-origin temporal backtesting audit (zero future leakage).
- [ ] **REQ-03.6:** Quantitative measurement of downward forecast bias caused by stockouts.
- [ ] **REQ-03.7:** Stratified evaluation across Syntetos-Boylan demand quadrants.

### Module 04: Inventory Gap Prediction & Deep Learning (15% Weight)
- [ ] **REQ-04.1:** Formal primary prediction target ($P(\text{phantom OOS})$ or $P(|G| \ge \theta)$).
- [ ] **REQ-04.2:** Leakage-free feature engineering (zero-streak normalized by demand, standardized residuals, SOH days of supply, days since count/receipt, price tier, locked case, DSD flag).
- [ ] **REQ-04.3:** Tier 0 Baseline: Heuristic indices (zero-streak, high-activity index, low-recorded-inventory index).
- [ ] **REQ-04.4:** Tier 1 Baseline: Logistic Regression, LightGBM, CatBoost.
- [ ] **REQ-04.5:** Tier 2 Temporal Baseline: GRU / LSTM / DeepAR.
- [ ] **REQ-04.6:** Tier 3 Deep Learning / Transformer: Temporal Fusion Transformer (TFT) with static, past, and known-future covariates.
- [ ] **REQ-04.7:** Tier 3 Counter-Baseline: DLinear / NLinear baseline (Zeng et al., AAAI 2023) to honestly evaluate whether transformer machinery pays for itself.
- [ ] **REQ-04.8:** Precision@$k$ ($k=250$ counts/day), PR-AUC, lift over random, lift over heuristic, dollars-at-risk captured @$k$.
- [ ] **REQ-04.9:** Variable selection network interpretability by category and temporal attention weights.
- [ ] **REQ-04.10:** Probability calibration (Platt / Isotonic) and Reliability curves.

### Module 05: What a Count is Worth (10% Weight)
- [ ] **REQ-05.1:** Expected Value derivation and implementation $\text{EV}(i,d)$ in dollars.
- [ ] **REQ-05.2:** Non-100% recovery rates $\text{Rec}_i \in [0.75, 0.90]$ by category.
- [ ] **REQ-05.3:** Persistence / remaining days of impact estimation $\tau_i = \min(\text{SOH}_i / \hat{\mu}_D(i), 30)$ demonstrating slow-mover value dynamics.
- [ ] **REQ-05.4:** Separation of physical recovery value vs. information value (exploration vs exploitation).
- [ ] **REQ-05.5:** Submodularity analysis of the count objective function (diminishing returns across count-sets and aisle setups).
- [ ] **REQ-05.6:** Greedy submodular baseline implementation with $(1 - 1/e)$ theoretical approximation guarantee.
- [ ] **REQ-05.7:** Sensitivity analysis across recovery rate $\text{Rec}$, persistence $\tau$, hourly wage, and gap threshold.

### Module 06: Workforce & Count-Plan Optimisation (15% Weight)
- [ ] **REQ-06.1:** Primary formulation and implementation in Google OR-Tools CP-SAT.
- [ ] **REQ-06.2:** MILP comparison model (PuLP / Pyomo) comparing solve time, gap, and constraint flexibility.
- [ ] **REQ-06.3:** Exact modeling of 4.0-minute non-separable aisle setup overhead $\sigma_a$.
- [ ] **REQ-06.4:** Daily labor capacity constraint (240 usable minutes total across 3 associates).
- [ ] **REQ-06.5:** 90-day mandatory compliance floor ($L(i) + |D| \ge 90$).
- [ ] **REQ-06.6:** Workload balancing constraint across the 3 associates.
- [ ] **REQ-06.7:** Skill matching constraint (e.g. locked case HBA certification).
- [ ] **REQ-06.8:** Generation of tangible 7-day operational schedule table (`date, sku_id, aisle_id, duration, associate, gap_prob, EV`).
- [ ] **REQ-06.9:** Benchmark evaluation: Random vs. Legacy ABC vs. Top-$k$ Prob vs. Top-$k$ EV vs. Greedy Submodular vs. CP-SAT.
- [ ] **REQ-06.10:** Aisle clustering visualization demonstrating setup synergy.
- [ ] **REQ-06.11:** Pareto frontier generation: Labor hours vs. Net recovered dollar value.

### Module 07: Scale to 2,000 Stores (15% Weight)
- [ ] **REQ-07.1:** Scale arithmetic and design for 2,000 stores $\times$ 40,000 SKUs $\times$ 14 horizons $= \mathbf{1.12\text{B}}$ predictions/day.
- [ ] **REQ-07.2:** Distributed inference architecture (Ray/GPU batching, vectorization, caching, $< 45$ min nightly window, $\approx \$38.40/\text{day}$ compute budget).
- [ ] **REQ-07.3:** Global vs. Per-Store vs. Clustered models, cold-start strategy, and distillation of complex models into fast decoders.
- [ ] **REQ-07.4:** Two-level optimization decomposition (central budget allocation + local store subproblems) via Lagrangian / Benders / Greedy fallback.
- [ ] **REQ-07.5:** Closed-loop label feedback degradation analysis and $\epsilon$-exploration bandit prototype ($90\%$ exploit, $5\%$ compliance, $5\%$ pure $\epsilon$-random probe).
- [ ] **REQ-07.6:** MLOps, drift monitoring (feature, prediction, outcome/count-hit drift), point-in-time data contracts, and phased canary rollout.

### Module 08: Interactive Streamlit Web Application (10% Weight)
- [ ] **REQ-08.1:** Working multi-tab Streamlit dashboard running locally via standard command.
- [ ] **REQ-08.2:** Tab 1 (Problem): Formulation, Value tree equations, KPI hierarchy.
- [ ] **REQ-08.3:** Tab 2 (Data): Simulator controls, category breakdown, data browser, schema dictionary.
- [ ] **REQ-08.4:** Tab 3 (Statistics): ADI/$CV^2$ quadrants, distribution fits, zero-streak power curves.
- [ ] **REQ-08.5:** Tab 4 (Forecasting): Model comparison, censored vs. naive bias, quantile fan charts.
- [ ] **REQ-08.6:** Tab 5 (Gap Prediction): Leaderboard, PR curves, calibration diagrams, TFT feature importance.
- [ ] **REQ-08.7:** Tab 6 (Optimisation): Tangible 7-day plan, associate assignment, aisle clustering map, solver gap.
- [ ] **REQ-08.8:** Tab 7 (What-If): Live interactive sliders for labor hours, wages, recovery rate, aisle penalty with real-time Pareto curve recomputation.
- [ ] **REQ-08.9:** Tab 8 (Scale): 2,000-store diagrams, cost model, $\epsilon$-bandit exploration design.
- [ ] **REQ-08.10:** Tab 9 (Results): Head-to-head executive table, $\$ / \text{labor hour}$ ROI, chain-wide annualized case.

### Final Submission Deliverables & Quality Gates
- [ ] **DELIV-1:** `README.md` with $\le 2$-command setup running in $< 2$ minutes.
- [ ] **DELIV-2:** `PROBLEM.md` (formal 3–5 page mathematical specification).
- [ ] **DELIV-3:** `STATISTICAL_ANALYSIS.md` (comprehensive answers with empirical data backing).
- [ ] **DELIV-4:** `LIMITATIONS.md` (ruthless self-critique covering simulation, models, optimization, and feedback bias).
- [ ] **DELIV-5:** `AI_USAGE.md` (transparent log of tools, high-leverage prompts, and rejected outputs).
- [ ] **DELIV-6:** `data_dictionary.md` (complete field schema).
- [ ] **DELIV-7:** 12-slide presentation deck outline (`DECK_OUTLINE.md`).
- [ ] **DELIV-8:** End-to-end reproducible pipeline script (`scripts/run_pipeline.py`).
- [ ] **DELIV-9:** Complete automated test suite covering all modules.
