# DriftGuard: Project Limitations & Statistical Constraints

This document outlines the architectural boundaries, statistical constraints, and dataset caveats of DriftGuard. These limitations provide essential context for interpreting the empirical findings and identifying priorities for future work.

---

## 1. Dataset Limitations

### 1.1 Single Dataset Without Stated Commercial License
- The project is evaluated exclusively on a single e-commerce dataset (`data/E-Commerce.csv`, 8,900 cleaned rows) originating from an educational coursework repository ([mvinayak96/ecommerce-sales-analysis](https://github.com/mvinayak96/ecommerce-sales-analysis)).
- The dataset carries no explicit commercial license. It is utilized strictly for academic research, benchmarking, and demonstration purposes. Generalizability to other domains (such as healthcare, financial fraud, or high-dimensional tabular systems) cannot be inferred from these results.

### 1.2 Synthetic Sampling from a Finite Clean Pool
- Because live deployments encounter open streaming distributions, this evaluation simulates streaming batches by sampling batches of $N=300$ rows with replacement from a clean pool of 4,450 rows (the project is a research prototype, not a production tool).
- Consequently, individual rows repeat across batches over a 48-batch stream. While batch-to-batch composition varies, the sampling pool cannot model true continuous covariate emergence or open-world cardinality expansion.

### 1.3 `product_rating` Column Distributional Concern
- Inspection of `product_rating` reveals that it contains 33 distinct numerical values strictly between 1.00 and 5.00, with approximately equal shares of integer star ratings:
  - 5.0 stars: 22.25%
  - 4.0 stars: 19.49%
  - 3.0 stars: 19.07%
  - 2.0 stars: 18.86%
  - 1.0 stars: 18.01%
- In authentic consumer review systems, rating distributions typically exhibit strong negative skew (bimodal with high concentrations at 5.0 and 1.0, and few intermediate ratings). The observed uniform distribution across integer stars suggests that this column may have been synthetically generated or simulated in the source dataset.

---

## 2. Statistical & Detection Constraints

### 2.1 Synthetic Injected Drift versus Natural Drift
- All evaluated drift scenarios represent mathematically controlled perturbations:
  - Multiplicative scaling (`shift_mean`)
  - Bounded additive displacement (`shift_additive`)
  - Categorical mode reassignment (`shift_category_mix`)
  - Concept shift (`label_shift`)
- Real-world data drift often manifests as complex, non-linear interactions across multiple features simultaneously (e.g. seasonal promotional surges coupled with localized supply chain disruptions), which are not fully captured by single-variable or additive synthetic shifts.

### 2.2 Sensitivity to Hard Thresholds and Batch Sizes
- The empirical findings are dependent on the configured parameters:
  - $\text{PSI} \ge 0.25$
  - $\text{p-value} < 0.01$
  - Batch size $N=300$
- At $N=150$, quantile binning instability likely causes high false-alarm rates under `psi_only` (3.3 false alarms per 48-batch run).
- For small drift perturbations ($\times 1.15$ price increase), the PSI rarely exceeds $0.25$, causing the default `both` rule to miss 76.7% of runs under Fixed_1 and 90.0% under Adaptive.

### 2.3 Non-Comparable Raw PSI and Absence of Multiple-Testing Correction
- **No Multiple-Testing Correction:** Because drift tests are evaluated independently across 6 features per batch, testing multiple hypotheses simultaneously without Bonferroni or False Discovery Rate (FDR) corrections inflates family-wise error rates under unadjusted testing rules. While the combined rule `both` observed 0 false alarms over 30 clean runs, this reflects empirical threshold conservatism rather than formal family-wise error control.
- **Raw PSI Across Feature Types:** Population Stability Index values are not normalized across features with differing cardinality. Categorical features with higher cardinality (`product_main_category` with 18 categories) possess a higher baseline sampling noise floor (mean PSI 0.0955, max 0.1438) than continuous numeric columns or bounded ratings. Consequently, ranking features purely by raw PSI can obscure subtle but true drift in low-cardinality features.

### 2.4 Persistence Rule Discontinuity Near Threshold Boundaries
- The persistence rule mandates that an alert is confirmed only when two consecutive checks are flagged. A single clean check resets the persistence counter to zero.
- As observed in the diagnostic trace for Seed 44 under medium sudden drift, when true distribution drift hovers near the $0.25$ threshold ($0.21$ to $0.33$), sampling noise causes flagged status to fluctuate from batch to batch. In that scenario, Adaptive monitoring reverted to normal mode ($k=4$) while drift was still active, and Fixed_4 confirmed drift earlier than Fixed_1 solely because it sampled batch 20 directly, bypassing the fluctuating batches 17–19. Hard thresholding combined with strict counter reset introduces non-linear sensitivity to sampling points.

---

## 3. Architectural & Evaluation Scope

### 3.1 Un-Evaluated LLM Explanation Layer
- The system includes a hybrid decision architecture with an optional Large Language Model (LLM) explanation layer and structured JSON output parser.
- The LLM layer is implemented with a rule-based fallback but was not quantitatively evaluated. Due to lack of access to an active external API key and cost constraints, the LLM layer was not evaluated in the 30-seed automated benchmarks. All decision and retraining evaluations were executed using the deterministic 6-row rules engine fallback.

### 3.2 Evaluation Scope of Experiment D
- Experiment D was evaluated as a rules-only benchmark; expected actions come from the same rules table, so it is an end-to-end routing check and not an independent test of appropriateness.
- The "drift and degraded" scenario combines price drift with a target shift (label shift 1.8), so most of the performance drop comes from the target shift rather than input feature drift alone.
- Scenario parameters were chosen after a 3-seed pilot so that each scenario premise holds.
- The LLM layer was not run or evaluated.

### 3.3 Platform & Runtime Testing
- The code was developed and tested on Windows 11 with Python 3.14.6 only. The project is a research prototype, not a production tool.
- Cross-platform behavior on Linux/macOS and containerized environments has not been verified in continuous integration.

