# Model Assumptions, Edge Cases & Practical Limitations

## 1. Overview & Methodological Integrity

Every applied machine learning and decision science system relies on mathematical abstractions of reality. In accordance with rigorous scientific practice, this document catalogs the assumptions, edge cases, negative research results, and operational limitations of the Perpetual Inventory Predict-Then-Optimise framework.

---

## 2. Research Findings: Deep Learning & Graph Neural Networks (GNNs)

### 2.1 Spatial Aisle GNN Findings
- **Hypothesis:** Discrepancy events (theft, misplaced items) exhibit spatial correlation across adjacent store aisles and parent categories.
- **Experiment:** We constructed a spatial adjacency graph over 30 store aisles and evaluated a 2-layer Graph Convolutional Network (GCN) with neighborhood feature aggregation alongside tabular GBDT models.
- **Empirical Results:**
  - *Calibrated LightGBM GBDT:* PR-AUC = **0.9653**, Precision@250 = **1.000**
  - *Spatial Aisle GNN:* PR-AUC = **0.9645**, Precision@250 = **1.000**
- **Conclusion:** While the GNN is scientifically sound and achieves high accuracy, it did not outperform the calibrated LightGBM model. Feature importance analysis reveals that individual SKU-level zero-sales streaks and normalized forecast residuals provide significantly stronger discriminative signals for phantom stockouts than spatial aisle adjacency. Therefore, introducing PyTorch Geometric into the real-time production inference path is not justified by the marginal accuracy difference and would unnecessarily increase hardware and latency overhead.

### 2.2 Deep Sequence Models (TFT / Transformer)
- **Observations:** Heavy temporal sequence models (Temporal Fusion Transformers) require continuous sequence windowing ($L=30$ days) across 1.12 billion daily predictions at scale. The compute footprint ($\approx \$1,500/\text{day}$) is $40\times$ higher than vectorized GBDT models compiled with Treelite ($\approx \$38.30/\text{day}$) without delivering a commensurate improvement in precision on zero-inflated tabular counts.

---

## 3. Data & Simulation Simplifications

1. **Perfect Count Accuracy Assumption:**
   - The primary optimization model assumes associate audits have $100\%$ accuracy in detecting the true stock on hand when physically counted.
   - *Real-world Reality:* Human associates experience count fatigue, barcode scan errors, and miss items hidden behind shelf dividers ($\approx 5-10\%$ error rate).
2. **Deterministic Count & Setup Times:**
   - The operational solver models item count time as fixed $t_i = 1.6\text{ minutes}$ and aisle setup overhead as fixed $s_a = 4.0\text{ minutes}$.
   - *Real-world Reality:* Count duration varies based on high-density displays (e.g. cosmetics) vs bulk items (e.g. water packs), and aisle transit times depend on customer congestion.
3. **Discrete 24-Hour POS Aggregation:**
   - Transactions are aggregated at daily resolution.
   - *Real-world Reality:* Mid-day theft or delivery delays occurring between 10:00 AM and 2:00 PM are not distinguishable from end-of-day discrepancies until the next nightly batch.

---

## 4. Supply Chain & Economic Edge Cases

1. **Extreme Supply Disruption (Regime Shifts):**
   - In the event of catastrophic supplier stockouts (e.g. port strikes or severe weather), zero sales across entire categories are caused by upstream shortages rather than store-level phantom stockouts. While the Poisson hazard model captures store-level signals, global macro shocks require central upstream ASN flags to suppress false-positive audit alerts.
2. **Imperfect Substitution Modeling:**
   - The economic value engine models customer substitution as a fixed category parameter ($\alpha_i = 0.45$). In reality, brand-loyal consumers may abandon the basket entirely if their preferred SKU is missing, whereas generic commodity buyers substitute seamlessly.

---

## 5. Summary of Recommended Mitigations for Production

| Limitation | Impact | Production Mitigation |
| :--- | :--- | :--- |
| Associate Audit Noise | Stale ledger corrections | Introduce double-blind spot audits on high-value SKUs |
| Human Count Fatigue | Degraded audit quality after 2 hours | Split 240-min budget into two 120-min morning/afternoon shifts |
| Supplier Disruptions | False phantom alarms | Link upstream EDI/ASN shipment feeds to zero-streak detector |
| Feedback Loop Bias | Blindness over unaudited SKUs | Enforce mandatory 5% $\epsilon$-greedy random audit budget |
