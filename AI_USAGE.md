# AI & LLM Assistance Transparency Statement

In compliance with professional research and engineering standards, this document details how Artificial Intelligence and Large Language Models (LLMs) were utilized during the design, development, and validation of the Perpetual Inventory Challenge project.

---

## 1. Role of AI in the Project Lifecycle

The development of this project followed an augmented pair-programming paradigm where the human engineer directed all mathematical formulations, architectural decisions, and scientific hypotheses, while AI tools assisted with boilerplate code generation, documentation structuring, and refactoring.

```
+-------------------------------------------------------------------------------+
|                             Human-Directed Workflow                          |
|                                                                               |
|  1. Problem Formulation & Formal Math -> Human Mathematical Modeling          |
|  2. Experimental Design & Hypotheses   -> Human Scientific Rigour             |
|  3. Model Architectures & Solvers     -> Guided Engineering & Verification    |
|  4. Code Generation & Boilerplate     -> LLM Assisted (DeepMind Antigravity)  |
|  5. Empirical Validation & Pytest     -> Automated Execution (100% Passing)   |
|  6. Business Synthesis & Critique     -> Human Decision Analysis              |
+-------------------------------------------------------------------------------+
```

---

## 2. Specific Contributions & Interaction Breakdown

### 2.1 Code Implementation Assistance
- **Generative Simulation (Module 01):** Assisted in translating non-homogeneous Poisson processes, compound Poisson theft bursts, and discrete event error mechanisms into vectorized NumPy/Pandas pipelines.
- **Statistical Fitting (Module 02):** Accelerated implementation of Scipy Negative Binomial and Zero-Inflated Poisson log-likelihood functions.
- **Forecasting & ML (Modules 03–04):** Generated baseline wrappers for Croston/TSB and LightGBM multi-quantile regression, DLinear models, and GNN spatial graph builders.
- **Optimization & Solvers (Module 06):** Formulated initial syntax for OR-Tools CP-SAT and PuLP MILP constraints (aisle linking, daily associate budget, compliance floors).
- **Streamlit Frontend (Module 08):** Generated Plotly chart definitions and dark-mode CSS styling.

### 2.2 Verification, Testing & Error Elimination
- All AI-generated code was subjected to automated unit and integration tests (`pytest -q` across 28 test cases).
- Execution profiling was performed to eliminate Python loops and vectorize array operations across the 1.17M row panel.
- No synthetic or fabricated metrics were accepted; all results presented in `STATISTICAL_ANALYSIS.md`, `SCALE_DESIGN.md`, and the Streamlit app reflect outputs from code executed on real data artifacts.

---

## 3. Human Design Decisions & Algorithmic Guardrails

1. **Rejecting Model Over-Engineering:**
   - The Aisle Graph Neural Network (GNN) was implemented and evaluated. It did **not** outperform Calibrated LightGBM in this dataset; individual SKU-level zero-streak features dominated spatial aisle context. This negative result is documented honestly in `LIMITATIONS.md`.
   - A Temporal Fusion Transformer (TFT) was considered as a research extension but was **not implemented** in this repository. No empirical comparison between TFT and LightGBM inference cost was performed; any prior claim of a 40× cost difference was an unvalidated estimate and has been removed.
2. **Economic Grounding:**
   - All dollar coefficients in the Value Tree (margin, basket abandonment loss, substitution salvage, associate wage) are **calibration assumptions** used for the synthetic simulator. They were chosen to be plausible for a grocery retailer but have not been empirically measured or validated against real retailer data. They should be re-estimated with actual retailer data before any production deployment.
3. **Deterministic Reproducibility:**
   - Enforced fixed random seeds (`seed=42`) across all simulation, feature engineering, and model training scripts.
