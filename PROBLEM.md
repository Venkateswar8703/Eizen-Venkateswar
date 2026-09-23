# PROBLEM.md: Formal Problem Formulation & Economic Specification
**Project:** Perpetual Inventory Challenge — Cycle Count Optimisation  
**Author:** Applied Science & Operations Research  
**Scope:** Store `STORE_0001` (1,600 SKUs across 10 Categories, 104 Weeks History)

---

## 1. Executive Summary (CFO Business Case)

Modern automated store replenishment relies on a digital book ledger ($\text{SOH}$). When shrinkage, cashier mis-scans, unrecorded spoilage, or receiving errors create **phantom inventory** ($\text{SOH} > 0$ while physical shelf $\text{TOH} = 0$), automated replenishment fails to trigger. The SKU sits at zero on the shelf indefinitely, generating an invisible, cumulative drain on store margin, e-commerce fulfillment rates, and customer loyalty. 

With $1,600$ SKUs and a fixed store labor budget of **4 hours (240 minutes) per day**, manual full-store audits are economically impossible. This project implements an end-to-end **Predict-Then-Optimise** system that estimates the probability and dollar impact of latent inventory discrepancies, balances non-separable aisle travel setup costs, and generates an optimal rolling 7-day associate count schedule that maximizes recovered gross margin per labor dollar invested.

```
                    ┌──────────────────────────────────────────────┐
                    │          THE PHANTOM INVENTORY CYCLE         │
                    └──────────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Physical Stock vanishes (Theft/Spoil)  │
                      │  TrueOnHand = 0  |  Book SOH = 10      │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ POS Sales drop to ZERO (Censored)      │
                      │ ERP sees SOH > Reorder Point           │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ REPLENISHMENT NEVER TRIGGERS!          │
                      │ Customer faces an empty shelf (OOS)    │
                      └────────────────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │ Targeted Cycle Count fixes ledger      │
                      │ Replenishment triggers -> Sales Resume │
                      └────────────────────────────────────────┘
```

---

## 2. Formal Mathematical Notation

### 2.1 Sets and Indices
* $i \in \mathcal{I} = \{1, 2, \dots, N\}$: Set of all Stock Keeping Units (SKUs), $N \approx 1,600$.
* $a \in \mathcal{A} = \{1, 2, \dots, A\}$: Set of aisles/bays in the store, $A \approx 32$.
* $a(i) \in \mathcal{A}$: The designated physical aisle where SKU $i$ is merchandised.
* $c \in \mathcal{C} = \{1, 2, \dots, 10\}$: Set of product categories (e.g., *Health & Beauty*, *Produce*, *Dairy*).
* $\mathcal{I}_c \subset \mathcal{I}$: Subset of SKUs belonging to category $c$.
* $d \in \mathcal{D} = \{1, 2, \dots, H\}$: Days in the rolling operational planning horizon ($H = 7$).
* $t \in \mathcal{T} = \{1, 2, \dots, T\}$: Historical observation time steps (days, $T = 730$).
* $w \in \mathcal{W} = \{1, 2, 3\}$: Set of count-capable store associates available for cycle audits.

---

### 2.2 State and Random Variables
* $\text{TOH}(i, t) \in \mathbb{Z}_{\ge 0}$: **True On-Hand** physical inventory of SKU $i$ at the beginning of day $t$ (unobserved by the ERP).
* $\text{SOH}(i, t) \in \mathbb{Z}$: **System On-Hand** (book inventory) recorded in the perpetual inventory ledger at start of day $t$.
* $G(i, t) \triangleq \text{SOH}(i, t) - \text{TOH}(i, t)$: **Inventory Gap** of SKU $i$ on day $t$.
  * $G(i, t) > 0$: **Phantom Inventory** (System overstates stock; risk of silent stockout).
  * $G(i, t) < 0$: **Ghost Shortage** (System understates stock; risk of over-ordering & spoilage).
* $D(i, t) \in \mathbb{Z}_{\ge 0}$: Latent, unconstrained customer demand for SKU $i$ on day $t$.
* $S(i, t) \in \mathbb{Z}_{\ge 0}$: Observed Point-of-Sale (POS) transactions at checkout:
  $$S(i, t) = \min\left(D(i, t),\, \text{TOH}(i, t) + Q_{\text{rec}}(i, t)\right)$$
* $\Theta(i, t) \ge 0$: Shrinkage/theft quantity (bursty compound process).
* $M(i, t) \in \mathbb{Z}$: Cashier mis-scan / wrong PLU quantity error.
* $\Xi(i, t) \ge 0$: Unrecorded damaged/spoiled units removed from the shelf.
* $R_{\Delta}(i, t) \in \mathbb{Z}$: Receiving discrepancy (e.g., case vs. each pack errors).
* $\Psi(i, t) \ge 0$: Misplaced inventory (physically in backroom or wrong aisle, unavailable to shopper).

---

### 2.3 Deterministic System Parameters
* $p_i \in \mathbb{R}_{+}$: Retail selling price of SKU $i$ ($\$$/unit).
* $c_i \in \mathbb{R}_{+}$: Unit cost of goods sold (COGS) of SKU $i$ ($\$$/unit).
* $m_i \triangleq p_i - c_i$: Gross margin per unit ($\$$/unit).
* $h_i \in \mathbb{R}_{+}$: Daily inventory holding cost rate ($\$$/unit/day).
* $t_i \in [0.8, 3.5]$: Expected standard labor time to audit SKU $i$ (minutes).
* $\sigma_a = 4.0$: Setup overhead to open and inspect aisle $a$ on day $d$ (minutes: ladder retrieval, scanner sync, cart transit).
* $C_{w, d} \in \mathbb{R}_{+}$: Usable cycle-count labor minutes for associate $w$ on day $d$ ($\sum_w C_{w,d} \le 240$ min/day).
* $L(i) \in \mathbb{Z}_{\ge 0}$: Elapsed days since SKU $i$ was last physically counted.
* $R_{\text{max}} = 90$: Maximum allowable recount interval (regulatory / SOX compliance floor).
* $\kappa(i) \in \{0, 1\}$: Special skill flag (e.g., locked security case access, deli scale calibration).
* $A_{w} \in \{0, 1\}$: Associate $w$ skill certification indicator.

---

### 2.4 Decision Variables (Optimisation Layer)
* $x_{i, d} \in \{0, 1\}$: Binary indicator whether SKU $i$ is audited on day $d \in \mathcal{D}$.
* $y_{a, d} \in \{0, 1\}$: Binary indicator whether aisle $a$ is opened/visited for audits on day $d$.
* $z_{i, w, d} \in \{0, 1\}$: Binary indicator whether associate $w$ is assigned to count SKU $i$ on day $d$.

---

## 3. The Prediction Target: Rigorous Definition & Justification

### 3.1 Primary Operational Target
We define the primary predictive target as a joint event:
$$\pi(i, d) \triangleq P\left(G(i, d) \ge 1 \;\land\; \text{TOH}(i, d) = 0 \;\middle|\; \mathcal{F}_{t_0}\right)$$
where $\mathcal{F}_{t_0}$ represents the historical filtration of observable POS, receipts, prices, promotions, and past audit outcomes up to planning baseline time $t_0$.

### 3.2 Why Not Just Predict Point Gap $\mathbb{E}[G(i, t)]$?
1. **Asymmetry of Risk:** A gap of $+5$ units on an item with $\text{TOH} = 20$ causes zero stockouts today (safety stock absorbs it). But a gap of $+2$ units on an item with $\text{TOH} = 0$ freezes replenishment and halts $100\%$ of sales.
2. **Decision Relevance:** Labor is allocated to stop immediate revenue bleed. The exact integer size of a gap matters less than whether it causes an **active phantom stockout** ($\text{TOH} = 0 \land \text{SOH} > 0$).
3. **Multi-Task Framing:** We simultaneously predict:
   * **Classification:** $\hat{\pi}(i, d) = P(\text{Phantom Out-of-Stock})$.
   * **Censored Demand:** $\hat{\mu}_D(i, d) = \mathbb{E}[D(i, d) \mid \mathcal{F}_{t_0}]$ (unconstrained customer demand).
   * **Gap Magnitude:** $\hat{g}(i, d) = \mathbb{E}[G(i, d) \mid G(i, d) > 0]$.

---

## 4. The Value Tree: Gap to P&L

Every dollar recovered through cycle counts is rigorously broken down into quantifiable balance-sheet and income-statement impacts:

```
                               ┌──────────────────────────────────────────────┐
                               │           EXPECTED COUNT VALUE EV(i,d)       │
                               └──────────────────────────────────────────────┘
                                                       │
                     ┌─────────────────────────────────┴─────────────────────────────────┐
                     ▼                                                                   ▼
       ┌───────────────────────────┐                                       ┌───────────────────────────┐
       │   GROSS BENEFIT RECOVERED │                                       │    COST OF AUDIT & NOISE  │
       └───────────────────────────┘                                       └───────────────────────────┘
                     │                                                                   │
     ┌───────────────┼───────────────┐                                   ┌───────────────┴───────────────┐
     ▼               ▼               ▼                                   ▼                               ▼
┌─────────┐    ┌───────────┐   ┌───────────┐                       ┌───────────┐                   ┌───────────┐
│ Direct  │    │ Basket    │   │ Holding & │                       │ Direct    │                   │ False     │
│ Margin  │    │ Spillover │   │ Spoilage  │                       │ Labor Wage│                   │ Alarm Cost│
│ Loss    │    │ Loss      │   │ Relief    │                       │ Cost      │                   │ (Trust)   │
└─────────┘    └───────────┘   └───────────┘                       └───────────┘                   └───────────┘
```

### 4.1 Value Equation Terms
For SKU $i$ on day $d$, the Expected Net Value $v_{i, d}$ of performing a physical count is:

$$v_{i, d} = \hat{\pi}(i, d) \cdot \text{Rec}_i \cdot \tau_i \cdot \left[ \hat{\mu}_D(i) \cdot \left( m_i + \beta_i \cdot \bar{M}_{\text{basket}} - \alpha_i \cdot m_{\text{sub}} \right) \right] + \mathbb{I}_{(\hat{g} < 0)} \cdot |\hat{g}_i| \cdot h_i - \text{LaborCost}(i) - \text{FalseAlarm}(i)$$

Where empirical coefficients are established as follows:

| Term | Symbol | Baseline Value | Source / Empirical Rationale |
| :--- | :---: | :---: | :--- |
| **Recovery Efficiency** | $\text{Rec}_i$ | $0.75 - 0.90$ | Fraction of gaps resolved. Fresh/Packaged = $0.90$; HBA (theft/loss) = $0.75$. |
| **Persistence Horizon** | $\tau_i$ | $\min\left(\frac{\text{SOH}_i}{\hat{\mu}_D(i)},\, 30\right)$ | Expected days until natural self-correction (truck receipt / negative balance walk). Fast movers $\approx 3$d; slow movers $\approx 25$d. |
| **Unit Margin** | $m_i$ | $\$0.40 - \$12.50$ | $p_i - c_i$ directly calculated from product master. |
| **Basket Spillover** | $\beta_i \cdot \bar{M}_{\text{basket}}$ | $0.12 \times \$18.00 = \$2.16$ | Probability shopper abandons full basket due to missing core destination item (Gruen & Corsten, 2007). |
| **Substitution Relief** | $\alpha_i \cdot m_{\text{sub}}$ | $0.45 \times m_i$ | Fraction of demand salvaged by customer buying competing brand in same category. |
| **Labor Wage Cost** | $c_{\text{wage}} \cdot t_i$ | $\$0.35/\text{min} \times t_i$ | Loaded store associate cost ($\$21.00/\text{hour}$). |
| **False Alarm Penalty**| $\text{Cost}_{\text{FP}}$ | $\$0.50$ | Associate frustration, disruption of stocking tasks, model trust erosion. |

---

## 5. Mathematical Optimization Formulation (Module 06 Core)

### 5.1 Objective Function (Currency: Net Dollars Recovered)
$$\max_{x, y, z} \quad \sum_{d \in \mathcal{D}} \sum_{i \in \mathcal{I}} v_{i, d} \cdot x_{i, d} \;-\; \lambda_{\text{setup}} \sum_{d \in \mathcal{D}} \sum_{a \in \mathcal{A}} \sigma_a \cdot y_{a, d}$$

### 5.2 Constraints Specification

#### 1. Horizon Uniqueness (Hard Constraint)
Each SKU is counted at most once in any 7-day planning window:
$$\sum_{d \in \mathcal{D}} x_{i, d} \le 1 \quad \forall i \in \mathcal{I}$$

#### 2. Non-Separable Aisle Activation (Hard Constraint)
A SKU can only be counted if its designated aisle is activated and setup overhead is paid:
$$x_{i, d} \le y_{a(i), d} \quad \forall i \in \mathcal{I}, \; \forall d \in \mathcal{D}$$

#### 3. Associate Daily Labor Capacity (Hard Constraint)
Total count duration plus assigned aisle setups cannot exceed daily shift capacity:
$$\sum_{i \in \mathcal{I}} t_i \cdot z_{i, w, d} \le C_{w, d} \quad \forall w \in \mathcal{W}, \; \forall d \in \mathcal{D}$$
$$\sum_{w \in \mathcal{W}} \sum_{i \in \mathcal{I}} t_i \cdot z_{i, w, d} + \sum_{a \in \mathcal{A}} \sigma_a \cdot y_{a, d} \le \sum_{w \in \mathcal{W}} C_{w, d} \quad \forall d \in \mathcal{D}$$

#### 4. Assignment Linkage (Hard Constraint)
If SKU $i$ is counted on day $d$, exactly one associate must be assigned:
$$\sum_{w \in \mathcal{W}} z_{i, w, d} = x_{i, d} \quad \forall i \in \mathcal{I}, \; \forall d \in \mathcal{D}$$

#### 5. 90-Day Regulatory Compliance Floor (Hard Constraint)
Any SKU reaching 90 days without a physical audit must be scheduled within the horizon:
$$\sum_{d \in \mathcal{D}} x_{i, d} \ge 1 \quad \forall i \in \{j \in \mathcal{I} \mid L(j) + H \ge R_{\text{max}}\}$$

#### 6. Skill Compatibility (Hard Constraint)
Restricted items (e.g., locked fragrance cases) can only be audited by certified staff:
$$z_{i, w, d} \le A_{w} \quad \forall i \text{ with } \kappa(i) = 1, \; \forall w \in \mathcal{W}, \; \forall d \in \mathcal{D}$$

#### 7. Workload Balancing (Soft / Fairness Constraint)
Associate daily active minutes must not deviate from the team mean by more than $\Delta = 20$ minutes:
$$\left| \sum_{i \in \mathcal{I}} t_i \cdot z_{i, w, d} - \frac{1}{|\mathcal{W}|}\sum_{w'} \sum_{i} t_i \cdot z_{i, w', d} \right| \le \Delta + s_{w, d}^{\text{bal}} \quad \forall w, d$$

#### 8. Category Starvation Floor (Soft Constraint)
Prevent the optimizer from completely abandoning low-margin categories (e.g., Packaged Grocery):
$$\sum_{i \in \mathcal{I}_c} x_{i, d} \ge m_c - s_{c, d}^{\text{cat}} \quad \forall c \in \mathcal{C}, \; \forall d \in \mathcal{D}$$

---

## 6. Business & Science KPI Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: EXECUTIVE BUSINESS KPIS (CFO / VP Retail Operations)               │
│ - Net Recovered Margin ($/month)                                            │
│ - On-Shelf Availability (OSA %): Target > 96.5%                             │
│ - Store Shrink % of Revenue: Reduction from 1.8% to 1.3%                    │
│ - E-commerce Bopis Pick-Fill Rate: Target > 98.2%                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: OPERATIONAL & DECISION METRICS (Store Manager / Industrial Eng)    │
│ - Dollars Recovered per Labor Hour ($/hr): Benchmark > $85.00/hr            │
│ - Precision@k (k = 250 counts/day): Hit rate of true discrepancies          │
│ - Aisle Clustering Density (SKUs counted per aisle visited)                 │
│ - 90-Day Compliance Breach Count: Target = 0                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: MACHINE LEARNING & STATISTICAL METRICS (Data Science Team)         │
│ - PR-AUC (Precision-Recall Area Under Curve, class-imbalanced)              │
│ - Expected Calibration Error (ECE) & Brier Score on predicted probabilities │
│ - Quantile Loss / Pinball Loss on Latent Demand Forecasts                   │
│ - DLinear vs. LightGBM Lift Ratio (research extension only; TFT not implemented)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Counterfactual Evaluation Framework

### The Core Science Dilemma
**The Fundamental Problem of Causal Inference in Auditing:** When we count a SKU, we alter reality: the inventory ledger is corrected, phantom stockouts are immediately replenished, and future sales resume. For uncounted items, we **never observe** the counterfactual ground truth.

### Evaluation Protocols
1. **Semi-Synthetic Offline Ground Truth (Module 01):**
   Evaluate historical policies directly against the hidden simulator layer ($\text{TOH}(i,t)$), allowing exact counterfactual measurement of what sales *would have occurred* had a count not fixed the gap.
2. **Quasi-Experimental Design (The 90-Day Compliance Anchor):**
   The mandatory 90-day compliance floor forces audits on SKUs regardless of model score. Because their timing is driven strictly by elapsed time ($L(i) = 89$), these counts act as **as-if-randomly assigned audit probes**, providing an unconfounded test sample to estimate unbiased model precision.
3. **Prospective Field Trial Design (Module 02 Question 12):**
   * **Unit of Randomization:** Matched store pairs or stepped-wedge aisle clusters.
   * **Treatment:** Model-driven count lists vs. Legacy ABC rotation.
   * **Primary Metric:** Store-level sales lift and shrink audit variance after 12 weeks.

---

## 8. Assumptions Register

| Assumption | Category | Risk / Impact | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **A1: Poisson/NegBin Demand** | Verifiable | High if bursty spikes dominate. | Test goodness-of-fit across categories in Module 02. |
| **A2: Constant Recovery Rate ($\text{Rec}_i$)**| Unverifiable | Moderate. Store staff may miss misplaced backroom items. | Sensitivity analysis in Module 05 across $\text{Rec} \in [0.5, 1.0]$. |
| **A3: Instantaneous Replenishment Trigger**| Deliberately Simplifying | Low. Supplier lead times create 2-day delivery lag. | Model expected replenishment delay $\tau_{\text{lead}} = 2$ days. |
| **A4: Independence of Aisle Traversal** | Deliberately Simplifying | Low. True walk path follows TSP within aisles. | Add intra-aisle TSP routing extension in Module 06. |

---

## 9. Critical Self-Critique: Two Ways This Formulation Could Be Wrong

1. **The Aisle-Clustering Paradox:**  
   By adding a 4-minute setup penalty per aisle ($\sigma_a = 4$), the optimizer strongly prefers counting 15 low-value items in one aisle rather than visiting 3 separate aisles for high-value items. If the gap probability model is noisy, this clustering creates **systematic blindness** to isolated high-shrink items in distant aisles.
2. **The Feedback Loop Collapse (Closed-Loop Label Bias):**  
   If the model only counts high-probability items, the training dataset for future quarters will only contain labels for items the model selected. The model will progressively forget about categories that drift silently into high shrink, creating fatal blind spots unless an $\epsilon$-exploration budget is enforced.
