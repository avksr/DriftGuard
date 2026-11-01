# DriftGuard: Technical & Statistical Methodology

This document provides a comprehensive technical reference for the data processing, statistical detection algorithms, monitoring strategies, decision logic, and evaluation protocols implemented in DriftGuard.

---

## 1. Dataset & Preprocessing

### 1.1 Raw Dataset Provenance
- **Dataset:** Indian E-Commerce Orders
- **Source:** [mvinayak96/ecommerce-sales-analysis](https://github.com/mvinayak96/ecommerce-sales-analysis)
- **Nature:** Tabular order records from an Indian e-commerce marketplace (academic DSBA coursework project).
- **Raw Row Count:** 8,906 rows.

### 1.2 Data Cleaning (`src/data_loader.py`)
1. **Column Selection & Renaming:** Drops unstructured long text fields (`product_name`, `description`), identifiers (`CustomerID`, `InvoiceNo`), and single-value columns. Renames columns to standardized snake_case identifiers.
2. **Deduplication:** Removes exact duplicate rows.
3. **Price Sanitization:** Filters out records where `retail_price <= 0` or `discounted_price <= 0` or `discounted_price > retail_price`.
4. **Timestamp Parsing:** Parses order date strings; rows with unparseable dates are removed.
5. **Whitespace Stripping:** Strips leading and trailing whitespace from string categorical columns.
6. **Cleaned Row Count:** 8,900 valid rows.

### 1.3 Target & Feature Definitions
- **Target Column:** `discount_pct`
  - Formula: $\text{discount\_pct} = \frac{\text{retail\_price} - \text{discounted\_price}}{\text{retail\_price}}$
  - Strict bounded domain: $[0.0, 1.0]$.
- **Monitored Features:**
  - `retail_price` (numeric): Continuous retail price in Indian Rupees (INR).
  - `product_rating` (numeric): Product review rating in $[1.0, 5.0]$.
  - `Region` (categorical): Geographical customer zone (e.g., North, South, East, West).
  - `product_main_category` (categorical): Top-level product category (e.g., Clothing, Electronics).
  - `BrandGroup` (categorical): Top 10 most frequent brands; all remaining brands grouped into `Other`.
  - `OrderMonth` (categorical): Month integer (1 to 12) derived from the order timestamp.

### 1.4 Reference and Clean Pool Split
- Cleaned rows (8,900 rows) are randomly partitioned 50/50 using a fixed random seed (`random_seed: 42`):
  - **Reference Data:** 4,450 rows (used for baseline model training and reference distribution histograms/quantiles).
  - **Clean Pool:** 4,450 rows (used by the batch simulator to sample streaming production batches).
- The split is strictly disjoint and reproducible. Preprocessing transformations (such as brand frequency rankings and quantile bin edges) are computed exclusively on the reference data to prevent leakage.

---

## 2. Baseline Machine Learning Model (`src/baseline_model.py`)

- **Model Family:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`).
- **Hyperparameters:** `n_estimators=100`, `random_state=42`, default tree depth and split criteria.
- **Data Partitioning:** Reference split (4,450 rows) is split 80/20 into baseline train (3,560 rows) and reference validation (890 rows).
- **Baseline Error Metric:** Mean Absolute Error (MAE) on reference validation data: $\approx 0.1287$.

---

## 3. Custom Statistical Drift Detection (`src/drift_detector.py`)

All statistical tests are implemented from scratch without third-party drift libraries.

### 3.1 Two-Sample Kolmogorov-Smirnov (KS) Test (Numeric Features)
Evaluates whether reference sample $X$ and batch sample $Y$ originate from the same continuous distribution:
1. Pool and sort all observations from $X$ and $Y$.
2. Compute empirical cumulative distribution functions:
   $$F_X(t) = \frac{1}{n_X} \sum_{i=1}^{n_X} \mathbb{I}(X_i \le t), \quad F_Y(t) = \frac{1}{n_Y} \sum_{j=1}^{n_Y} \mathbb{I}(Y_j \le t)$$
3. KS statistic is the maximum absolute divergence:
   $$D = \sup_t |F_X(t) - F_Y(t)|$$
4. Compute asymptotic p-value using the Kolmogorov distribution series approximation with effective sample size $n_e = \sqrt{\frac{n_X n_Y}{n_X + n_Y}}$.

### 3.2 Population Stability Index (PSI)
Quantifies distributional shift by comparing bin frequency proportions between reference ($P$) and incoming batch ($Q$):
$$\text{PSI} = \sum_{k=1}^K (Q_k - P_k) \ln\left(\frac{Q_k}{P_k}\right)$$
- **Numeric Features:**
  - Bin edges are defined using 10 reference quantiles ($0.0, 0.1, \dots, 1.0$).
  - Duplicate quantile edges (arising from repeated discrete values) are deduplicated.
  - A Laplace smoothing epsilon ($10^{-4}$) is applied to prevent division by zero or $\ln(0)$ on empty batch bins.
- **Categorical Features:**
  - Bins correspond to unique categories present in the reference distribution.
  - Unseen categories in incoming batches are mapped to an `__unseen__` bin with reference frequency smoothed to $10^{-4}$.

### 3.3 Chi-Square Goodness-of-Fit Test (Categorical Features)
For categorical columns, evaluates observed batch category counts $O_k$ against expected counts $E_k = n_{\text{batch}} \cdot P_k$:
$$\chi^2 = \sum_{k=1}^K \frac{(O_k - E_k)^2}{E_k}$$
p-value is evaluated with degrees of freedom $df = K - 1$.

### 3.4 Feature Flagging Rule
A single feature check is flagged if:
- Under rule `both` (default): $(\text{p-value} < 0.01) \land (\text{PSI} \ge 0.25)$
- Under rule `ks_only`: $\text{p-value} < 0.01$
- Under rule `psi_only`: $\text{PSI} \ge 0.25$

A batch is marked **flagged** if at least one monitored feature meets the flag condition.

### 3.5 Persistence Rule for Confirmed Alerts
To prevent transient sampling noise from triggering false alarms, a **confirmed drift alert** requires **two consecutive flagged checks**:
- Check $t$ flagged + Check $t+1$ flagged $\implies$ **Confirmed Alert** raised at Check $t+1$.
- Any single clean check resets the persistence counter to zero.

---

## 4. Monitoring Strategies (`src/adaptive_monitor.py`)

A stream consists of $M=48$ consecutive batches of size $N=300$. Drift is injected at batch 16 (0-indexed).

1. **Fixed(1):** Full drift detection check executed on every batch ($k=1$). Always runs 48 checks.
2. **Fixed(4):** Full drift detection check executed every 4th batch ($k=4$, batches 0, 4, 8, 12, ...). Always runs 12 checks.
3. **Adaptive(4, 1, 3):**
   - **Normal Mode:** Runs checks at interval $k_{\text{normal}} = 4$ (batches 0, 4, 8, 12, ...).
   - **Alert Mode:** Triggered immediately when any check is **flagged**. The sampling interval collapses to $k_{\text{alert}} = 1$ (checks every batch).
   - **Recovery Rule:** Returns to normal mode ($k=4$) only after observing $N_{\text{clean}} = 3$ consecutive clean checks.

---

## 5. Performance Monitoring & Decision Agent (`src/performance_monitor.py`, `src/decision_agent.py`)

### 5.1 Performance Monitor
- Tracks rolling model Mean Absolute Error (MAE) over a sliding window of 5 batches.
- **Degradation Threshold:** Evaluated relative to baseline validation MAE:
  $$\text{MAE}_{\text{threshold}} = 1.20 \times \text{MAE}_{\text{baseline}}$$
- **Sustained Degradation:** Requires 2 consecutive degraded evaluations to mark status as `degraded`.
- If ground-truth labels are delayed or unavailable, status is marked `unverified`.

### 5.2 Deterministic Decision Rules Table
The decision engine maps monitoring signals to operational actions via a deterministic 6-row decision table:

| Condition # | Data Quality Status | Confirmed Drift Alert | Model Performance Status | Allowed Actions | Default / Chosen Action |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Failed | Any | Any | `['INVESTIGATE']` | `INVESTIGATE` |
| **2** | Passed | True | Unverified (labels hidden) | `['REQUEST_LABELS', 'INVESTIGATE']` | `REQUEST_LABELS` |
| **3** | Passed | True | Normal ($\text{MAE} \le 1.2\times$) | `['INVESTIGATE', 'DO_NOTHING']` | `INVESTIGATE` |
| **4** | Passed | True | Degraded ($\text{MAE} > 1.2\times$, sustained) | `['PROPOSE_RETRAIN', 'INVESTIGATE']` | `PROPOSE_RETRAIN` |
| **5** | Passed | False | Degraded ($\text{MAE} > 1.2\times$, sustained) | `['INVESTIGATE', 'DO_NOTHING']` | `INVESTIGATE` |
| **6** | Passed | False | Normal ($\text{MAE} \le 1.2\times$) | `['DO_NOTHING']` | `DO_NOTHING` |

### 5.3 Guardrails & Orchestration
- If an LLM response is requested but produces an invalid action, unparseable JSON, or violates allowed actions, a strict programmatic guardrail overrides the output and reverts to the default deterministic action.
- Retraining is strictly forbidden unless both confirmed drift and sustained performance degradation are present (Row 4).

---

## 6. Retraining Pipeline (`src/retrainer.py`)

When `PROPOSE_RETRAIN` is triggered, candidate models are evaluated using dual-window validation:
1. **Training Split:** Candidate model trains on 75% of the recent 5 drifted batches combined with 25% historical reference data.
2. **Dual-Window Acceptance Criteria:**
   - **Recent Window Improvement:** Must improve MAE on the recent window evaluation split by at least 5% ($\text{gain} \ge 0.05$):
     $$\frac{\text{MAE}_{\text{current}} - \text{MAE}_{\text{candidate}}}{\text{MAE}_{\text{current}}} \ge 0.05$$
   - **Catastrophic Forgetting Guardrail:** MAE on historical reference validation data must not degrade by more than 10% ($\text{loss} \le 0.10$):
     $$\frac{\text{MAE}_{\text{candidate, ref}} - \text{MAE}_{\text{current, ref}}}{\text{MAE}_{\text{current, ref}}} \le 0.10$$
   - Candidates violating either criterion are rejected.

---

## 7. Experimental Protocols (Experiments A–D)

All research benchmarks are run across 30 random seeds (seeds 42 to 71) with 48 streaming batches ($N=300$).

### Experiment A: Detector Sensitivity & False-Alarm Trade-off
- Evaluates detection delay and false-alarm frequency across three detector flagging rules: `ks_only`, `psi_only`, and `both`.
- Evaluated across clean and drifted streams at three batch sizes: $N=150$, $N=300$, and $N=600$.

### Experiment B: Monitoring Strategy Benchmark
- Compares `Fixed_1`, `Fixed_4`, and `Adaptive(4,1,3)` across four drift scenarios:
  1. Large sudden price drift ($\times 2.0$)
  2. Medium sudden price drift ($\times 1.4$)
  3. Medium gradual price drift ($\times 1.4$ ramped over 8 batches)
  4. Small sudden price drift ($\times 1.15$)
  5. No drift (clean baseline)
- Metrics: Detection delay (batches from drift start to confirmed alert), missed detection rate, number of full checks executed, and percentage check savings.

### Experiment C: Feature Ranking Accuracy (Precision@k)
- Evaluates whether detector ranking places truly perturbed features at the top of the list across 5 evaluation batches post-drift.
- Metric: Precision@k = $\frac{\text{number of truly changed features in top } k}{k}$.
- Evaluated across single-feature, two-feature, and three-feature drift scenarios.

### Experiment D: End-to-End Decision Agent Evaluation
- Evaluates whether the decision agent selects the expected action across four streaming scenarios (batches 16 to 21):
  1. `drift_perf_fine`: Feature drift on `Region` with normal MAE $\implies$ `INVESTIGATE`.
  2. `drift_and_degraded`: Feature drift on `retail_price` + concept shift (1.8x) $\implies$ `PROPOSE_RETRAIN`.
  3. `perf_worse_no_drift`: Concept shift without feature drift $\implies$ `INVESTIGATE`.
  4. `drift_labels_hidden`: Feature drift on `retail_price` with hidden ground-truth labels $\implies$ `REQUEST_LABELS`.

---

## 8. Master Parameter Table (from `config.yaml`)

| Parameter | Configuration Key | Configured Value | Description |
| :--- | :--- | :--- | :--- |
| **Reference Split Fraction** | `data.reference_fraction` | 0.50 | 50% reference, 50% clean streaming pool |
| **Top Brands Grouped** | `data.top_brands` | 10 | Top 10 brands retained, remainder to `Other` |
| **Baseline Train Split** | `data.baseline_train_fraction`| 0.80 | 80% train / 20% validation on reference |
| **Batch Size** | `batch_size` | 300 | Rows per streaming production batch |
| **Batches per Run** | `n_batches_per_run` | 48 | Batches in one full temporal stream |
| **Drift Start Batch** | `drift_start_batch` | 16 | 0-indexed batch where perturbation starts |
| **Gradual Ramp Batches** | `gradual_ramp_batches` | 8 | Batches over which gradual drift linearly ramps |
| **Default Flag Rule** | `drift_rule` | `both` | Requires KS p-value $< 0.01$ and PSI $\ge 0.25$ |
| **PSI Quantile Bins** | `psi_bins` | 10 | Decile binning on reference distribution |
| **PSI Flag Threshold** | `psi_flag_threshold` | 0.25 | Significant distributional divergence cutoff |
| **p-value Threshold** | `p_value_threshold` | 0.01 | Significance level for KS and $\chi^2$ tests |
| **Persistence Required** | `persistence_required` | 2 | Consecutive flagged checks for confirmed alert |
| **Normal Interval** | `adaptive.normal_interval` | 4 | Batches between checks in normal mode |
| **Alert Interval** | `adaptive.alert_interval` | 1 | Batches between checks in alert mode |
| **Recovery Checks** | `adaptive.recovery_clean_required`| 3 | Consecutive clean checks to exit alert mode |
| **Performance Degradation Limit**| `performance.degradation_threshold`| 1.20 | $1.20 \times \text{baseline MAE}$ |
| **Consecutive Degraded Required**| `performance.consecutive_degraded_required`| 2 | Consecutive degraded batches for status |
| **Retraining Recent Window**| `retraining.recent_window_batches`| 5 | Post-drift batches included in retraining |
| **Retraining Min Improvement**| `retraining.min_improvement`| 0.05 | Minimum 5% MAE reduction on recent split |
| **Max Old Data Regression**| `retraining.max_old_data_regression`| 0.10 | Maximum 10% MAE increase on reference split |
| **Small Drift Sizes** | `drift_sizes.small` | Price $\times 1.15$, Drop 0.10, Rating $-0.2$ | Mild perturbation |
| **Medium Drift Sizes** | `drift_sizes.medium` | Price $\times 1.40$, Drop 0.25, Rating $-0.5$ | Standard benchmark perturbation |
| **Large Drift Sizes** | `drift_sizes.large` | Price $\times 2.00$, Drop 0.45, Rating $-1.0$ | Severe perturbation |

