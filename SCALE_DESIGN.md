# Enterprise Scale Architecture: 2,000 Stores & 1.12 Billion Daily Predictions (Module 07)

> **⚠️ PROPOSED ARCHITECTURE — NOT IMPLEMENTED**
>
> This document describes a **proposed production deployment design** for a hypothetical 2,000-store chain.
> The repository implements a **single-store prototype** (STORE_0001, 1,600 SKUs, 730 days).
> Hardware cost estimates are based on published cloud pricing as of 2024–2025 and are
> **estimates, not measured benchmarks**. Throughput figures are theoretical based on
> published LightGBM/Treelite benchmarks, not validated on this codebase at scale.

## 1. Executive Summary & Scale Dimensions

Deploying the Perpetual Inventory Optimization system across a national grocery chain entails scaling from a single store ($1,600$ SKUs) to an enterprise fleet:

$$\begin{aligned}
N_{\text{stores}} &= 2,000 \\
N_{\text{SKUs/store}} &= 40,000 \\
N_{\text{horizons}} &= 14 \text{ days} \\
\text{Total Daily Predictions} &= 2,000 \times 40,000 \times 14 = \mathbf{1,120,000,000 \ (1.12 \text{ Billion})}
\end{aligned}$$

At this volume, naive architectures fail due to memory saturation, feature store serialization bottlenecks, and uncontrolled cloud compute bills. This document proposes a production architecture, data pipeline, model topologies, cost models, monitoring safeguards, and distributed optimization engine. All cost and timing figures are estimates based on cloud provider pricing and theoretical throughput benchmarks.

---

## 2. End-to-End System Architecture

```
                                  +---------------------------------------+
                                  |     Point-of-Sale (POS) & ERP         |
                                  | (Sales, Receiving, DSD, Scrap, Audits)|
                                  +---------------------------------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      | Apache Kafka / Event Hubs   |
                                      +-----------------------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      | Apache Flink (Tumbling 1d)  |
                                      +-----------------------------+
                                                     |
                                                     v
                                      +-----------------------------+
                                      | Cloud Lakehouse (Parquet/   |
                                      | Iceberg on S3 / BigQuery)   |
                                      +-----------------------------+
                                                     |
                          +--------------------------+--------------------------+
                          |                                                     |
                          v                                                     v
            +---------------------------+                         +---------------------------+
            | Feast / Hopsworks         |                         | Offline Training Pipeline |
            | Offline Feature Store     |                         | (Ray Train / MLflow)      |
            +---------------------------+                         +---------------------------+
                          |                                                     |
                          v                                                     v
            +---------------------------+                         +---------------------------+
            | Daily Batch Inference     | <-----------------------| Champion Model Registry   |
            | Engine (Ray + Treelite)   |                         | (Calibrated LightGBM GBDT)|
            +---------------------------+                         +---------------------------+
                          |
                          v
            +-------------------------------------------------------------+
            | Scored Predictions & Value Engine: EV(i, d)                 |
            +-------------------------------------------------------------+
                          |
                          v
            +-------------------------------------------------------------+
            | Distributed Store-Level Solver (2,000 Parallel Ray Tasks)   |
            | OR-Tools CP-SAT + Greedy Fallback (60s SLA per store)       |
            +-------------------------------------------------------------+
                          |
                          v
            +-------------------------------------------------------------+
            | Handheld Terminals (Zebra TC57 / iOS Associate Store App)   |
            | Daily 7-Day Dynamic Cycle Audit Schedules                   |
            +-------------------------------------------------------------+
```

---

## 3. Data Ingestion & Feature Engineering

### 3.1 Point-in-Time Correctness & Partitioning
- **Partitioning Scheme:** Data lakehouse tables are partitioned by `(date, store_id_mod_100)` to optimize parallel reader throughput without small-file overhead.
- **Watermarking & Late-Arriving Data:** DSD receipts and direct vendor deliveries frequently arrive 24–48 hours late. The Flink stream maintains a 3-day watermarking window, writing append-only correction logs rather than mutating historical partitions in-place.
- **Feature Computation:** Point-in-time features (zero streaks, cumulative sales, rolling forecast residuals, days since last physical audit) are computed as vectorized window functions in DuckDB / PySpark / Polars, guaranteeing zero temporal data leakage.

---

## 4. Inference Engine & Hardware Cost Model

### 4.1 Batch Inference vs Streaming
- Inventory cycle counts are executed once daily during early morning associate shifts ($06:00 - 10:00$). 
- Therefore, **nightly batch inference at 03:00 UTC** is vastly superior to 24/7 streaming inference in compute efficiency, cost, and operational simplicity.

### 4.2 Why LightGBM + Treelite Dominates Deep Learning at Scale
- **Treelite / ONNX C-Tree Compilation:** Compiles LightGBM decision trees into vectorized C/C++ machine code utilizing AVX-512 SIMD instructions.
- **Throughput:** $1.2 \times 10^6$ predictions/second per 32-core CPU node ($37,500$ predictions/sec/core).
- **Cluster Sizing for 1.12B predictions:**
  $$\text{Execution Time} = \frac{1.12 \times 10^9 \text{ rows}}{32 \text{ nodes} \times 1.2 \times 10^6 \text{ rows/sec}} \approx 29.1 \text{ seconds (pure scoring)}$$
  Including feature loading and parquet serialization, end-to-end inference takes **$14.2\text{ minutes}$**.

### 4.3 Estimated Daily Compute Budget (Proposed — Not Measured)

> **⚠️ The figures below are cost estimates** based on AWS spot pricing (2024–2025 rates).
> They have not been validated by running this workload in production.
> Actual costs will depend on spot availability, data volume, and reserved capacity.

| Resource | Specification | Quantity | Daily Runtime | Unit Cost | Est. Daily Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inference Cluster** | AWS `c6i.8xlarge` (32 vCPU, 64 GB) | 16 Spot Instances | 0.5 hours | ~$1.36 / hr | ~$10.88 |
| **Feature Extraction** | AWS `r6i.4xlarge` (Spark/Polars) | 8 Spot Instances | 0.5 hours | ~$1.00 / hr | ~$4.00 |
| **Distributed Solver** | AWS `c6i.4xlarge` (16 vCPU) | 20 Spot Instances | 0.5 hours | ~$0.68 / hr | ~$6.80 |
| **Storage & Egress** | S3 / Cloudflare R2 | 500 GB daily I/O | Continuous | ~$0.015 / GB | ~$7.50 |
| **Metadata & Logging** | Managed Postgres / Redis | 1 instance | 24 hours | ~$0.38 / hr | ~$9.12 |
| **ESTIMATED TOTAL** | — | — | — | — | **~$38.30 / day** |

**Note:** These estimates assume stable spot pricing and theoretical throughput. Any chain-wide ROI
calculations based on single-store simulation results should be treated as illustrative only.

---

## 5. Model Topologies & Cold-Start Strategy

```
                       [Store Clustering Engine]
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
  Cluster 1: Urban High-    Cluster 2: Suburban High-  Cluster 3: Rural Low-
  Shrink Fresh Flagships    Volume Hypermarkets        Velocity Convenience
  (Fine-Tuned Head)         (Fine-Tuned Head)          (Fine-Tuned Head)
         ^                         ^                         ^
         +-------------------------+-------------------------+
                                   |
                       [Global Backbone GBDT]
                 (Trained across 80M pooled rows)
```

1. **Global Backbone + Cluster Fine-Tuning:**
   - Rather than maintaining 2,000 individual fragile models (which overfit on small sample sizes), we train a single **Global GBDT Backbone** on pooled cross-store transaction data with learnable store demographic and category embeddings.
   - Stores are clustered into $K=12$ operational clusters based on shrinkage intensity, sales volume, and shrink rate.
2. **Cold-Start Handling:**
   - **New Stores:** Assigned to the median cluster of their geographic format; prior gap probability defaults to empirical category priors until 30 days of POS history accumulate.
   - **New SKUs:** Inherit the shrinkage prior and velocity profile of their parent sub-category and price tier.

---

## 6. Continuous Monitoring & Production Drift Safeguards

| Failure Mode | Detection Metric | Alert Threshold | Automated Remediation |
| :--- | :--- | :--- | :--- |
| **Feature Drift** | Population Stability Index (PSI) | $\text{PSI} > 0.20$ | Auto-fallback to robust baseline; flag missing upstream telemetry |
| **Zero-Streak Shift** | Kolmogorov-Smirnov (KS) Statistic | $p < 0.01$ | Recompute streak features; inspect supplier delivery logs |
| **Prediction Drift** | Jensen-Shannon Divergence | $\text{JSD} > 0.15$ | Recalibrate probabilities using recent 14-day holdout |
| **Outcome Drift** | Rolling Precision@250 on Audits | Drop $> 15\%$ | Trigger automated retraining pipeline |
| **Feedback Loop Trap** | Audit Exploration Coverage | $\epsilon < 0.05$ | Enforce $\epsilon$-greedy $5\%$ random audit budget allocation |

---

## 7. Distributed Workforce Optimization at Scale

Solving a single centralized optimization problem across 2,000 stores simultaneously would involve $2,000 \times 40,000 \times 7 \times 3 \approx 1.68 \text{ Billion binary variables}$, which is computationally intractable for exact MILP.

### 7.1 Two-Level Hierarchical Decomposition
1. **Level 1 (Central Master Problem):**
   - Allocates weekly corporate labor hour budgets $L_s$ to store $s \in \{1, \dots, 2000\}$ using convex marginal return knapsack allocation based on aggregate store risk $\sum_{i} \text{EV}_s(i)$.
2. **Level 2 (Store-Level Subproblem):**
   - **Embarrassingly Parallel:** Each store solves its local 7-day associate scheduling problem independently via Ray actors.
   - Solver Engine: **OR-Tools CP-SAT** configured with a hard $45\text{-second SLA}$.
   - **Warm-Start:** Initialized with the greedy submodular heuristic solution.
   - **Fallback:** If CP-SAT reaches timeout without finding a feasible proof, the system automatically falls back to the deterministic Greedy Submodular Schedule, guaranteeing $100\%$ uptime SLA.

---

## 8. Summary: Implemented vs. Proposed

| Item | Status |
|------|--------|
| Single-store prototype (1,600 SKUs, 730 days) | ✅ **Implemented** |
| Closed-loop bandit simulation (`src/scale/`) | ✅ **Implemented** |
| Full architectural specification (2,000 stores) | 📐 **Proposed design** |
| Hardware cost model (~$38/day estimate) | 📐 **Estimated, not measured** |
| Distributed Ray inference cluster | 📐 **Proposed** |
| Feast/Hopsworks feature store | 📐 **Proposed** |
| 2,000-store parallel CP-SAT solver farm | 📐 **Proposed** |
| GBDT vs. Transformer throughput comparison | 📐 **Estimate based on literature** |
| PSI/KS drift detection in production | 📐 **Proposed** |
| Global backbone + cluster fine-tuning | 📐 **Proposed** |
