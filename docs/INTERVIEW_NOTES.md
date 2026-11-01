# DriftGuard: Technical Interview Guide & Module Explanations

This document provides two-sentence module summaries with key design decisions, followed by 12 technical interview questions and concise, code-grounded answers.

---

## 1. Module Summaries & Core Design Decisions

### `src/data_loader.py`
- **Explanation:** Ingests the raw Indian e-commerce order records, cleans non-positive prices and unparseable timestamps, and partitions data into disjoint reference and streaming pool sets. It computes the target variable (`discount_pct`) and encodes high-cardinality brand categories.
- **Main Design Decision:** Computed all preprocessing transformations (such as quantile bin edges and top brand rankings) exclusively on the 50% reference split to eliminate data leakage into the streaming pool.

### `src/baseline_model.py`
- **Explanation:** Trains and serializes a baseline `RandomForestRegressor` model predicting discount percentages on reference training data. It evaluates validation performance to establish baseline MAE for downstream drift degradation tracking.
- **Main Design Decision:** Selected Random Forest over linear regression because discount structures exhibit non-linear interactions across price, brand tier, and product category.

### `src/drift_detector.py`
- **Explanation:** Implements custom, zero-dependency statistical tests (two-sample Kolmogorov-Smirnov test, Population Stability Index, and Chi-Square goodness-of-fit) for tabular batches. It generates comprehensive batch reports containing per-feature statistics, p-values, and priority rankings.
- **Main Design Decision:** Built detectors entirely from scratch rather than wrapping existing libraries (e.g. Evidently or Alibi Detect) to guarantee complete visibility and interview explainability for every statistical computation.

### `src/batch_simulator.py`
- **Explanation:** Samples consecutive production batches from the clean pool and injects mathematically controlled perturbations across sudden, gradual, spread, and mix types. It attaches exact ground truth metadata (drift multiplier, affected columns) to every batch for empirical benchmarking.
- **Main Design Decision:** Enforced feature-specific domain bounds during injection (such as keeping prices strictly positive and clamping ratings within $[1.0, 5.0]$) to prevent unphysical distribution artifacts.

### `src/adaptive_monitor.py`
- **Explanation:** Implements `Fixed_1`, `Fixed_4`, and `Adaptive(4,1,3)` monitoring strategies to govern statistical check execution schedules. The adaptive monitor operates as a finite-state machine that collapses its sampling interval upon detecting an anomaly and recovers after consecutive clean checks.
- **Main Design Decision:** Enforced a two-check persistence rule before declaring confirmed alerts, distinguishing transient sampling noise from genuine distribution drift.

### `src/performance_monitor.py`
- **Explanation:** Computes rolling model Mean Absolute Error over a sliding 5-batch window against baseline reference error. It tracks whether model degradation exceeds the configured $1.20\times$ threshold over consecutive batches.
- **Main Design Decision:** Implemented an explicit `unverified` state when ground-truth labels are delayed or hidden, preventing the decision agent from falsely assuming performance is normal when labels are absent.

### `src/decision_agent.py`
- **Explanation:** Maps detector signals, performance states, and data quality flags to operational actions via a deterministic 6-row decision table. It supports an optional LLM explanation layer backed by strict schema validation and rule guardrails.
- **Main Design Decision:** Configured hard programmatic guardrails that override any non-compliant LLM recommendations and revert immediately to the deterministic rules table fallback.

### `src/retrainer.py`
- **Explanation:** Executes controlled candidate model retraining using recent drifted batches combined with historical reference data. It validates candidates against dual-window criteria requiring recent window improvement and bounded reference data regression.
- **Main Design Decision:** Enforced a 10% catastrophic forgetting ceiling on historical reference data to ensure candidate models do not overfit to localized drift at the expense of generalizability.

### `src/incident_logger.py`
- **Explanation:** Records structured monitoring alerts, operational decisions, and validation metrics to persistent disk storage in JSON and CSV formats. It provides the data source for auditability and dashboard incident reports.
- **Main Design Decision:** Maintained synchronized CSV and JSON exports to support both tabular spreadsheet analysis and programmatic REST ingestion.

### `app.py`
- **Explanation:** Provides a minimal, four-tab Streamlit dashboard displaying live simulation presets, feature distribution comparisons, incident logs, and research benchmark figures. It complies strictly with Section 8.1 UI rules using neutral grey reference traces and reserving red exclusively for confirmed alerts.
- **Main Design Decision:** Pre-cached datasets and simulation objects using `@st.cache_data` and `@st.cache_resource`, ensuring preset execution times remain close to the 10-second target.

### `experiments/run_all.py`
- **Explanation:** Orchestrates automated 30-seed research experiments (A, B, C, D), generates summary statistics, renders publication figures, and outputs run metadata manifests. It supports granular `--exp` targeting and `--quick` smoke testing.
- **Main Design Decision:** Implemented completed seed caching and non-interactive Matplotlib backend (`Agg`) to prevent memory leaks and allow headless execution across long experimental runs.

---

## 2. Twelve Technical Interview Questions & Answers

### Q1: Why did you combine the Kolmogorov-Smirnov test and PSI rather than using one alone?
**Answer:** The KS test is highly sensitive to distribution differences in continuous variables but produces high false-alarm rates at small sample sizes ($N=150$) due to sampling variance. PSI quantifies practical distributional distance across discrete bins but can be insensitive to small shifts that don't cross quantile thresholds. Requiring both (KS p-value $< 0.01$ and $\text{PSI} \ge 0.25$) eliminated all false alarms across 30 clean runs in Experiment A while preserving high sensitivity to meaningful drift.

### Q2: Why did you use a random 50/50 split between reference data and the streaming pool rather than a time-based split?
**Answer:** A time-based split would conflate natural temporal trends with our controlled synthetic drift injections, making it impossible to establish an uncontaminated ground truth. By partitioning the dataset randomly into a reference set and a clean pool, we guaranteed that the clean pool possessed the identical distribution to the reference set prior to synthetic perturbation, providing an exact zero-drift baseline.

### Q3: What is the persistence rule and why was it necessary?
**Answer:** The persistence rule requires two consecutive flagged checks before raising a confirmed drift alert. A single unflagged check resets the persistence counter to zero. This was necessary because single-batch statistical tests inevitably encounter sampling noise that occasionally crosses a threshold; requiring temporal persistence eliminated transient false alarms without adding excessive detection delay.

### Q4: Why was Adaptive monitoring slower to detect gradual drift than sudden drift?
**Answer:** Under gradual drift, the perturbation ramps linearly from 0 to 1 over 8 batches. In the early ramp batches (batches 16–20), the distribution shift is small, generating a PSI below the 0.25 threshold. Because Adaptive checks only every 4th batch in normal mode, it does not collapse its interval to every batch until a check is flagged, causing it to observe the drift later in the ramp cycle ($13.3$ batches delay vs $7.3$ for Fixed_1).

### Q5: What does the `product_rating` noise-floor finding reveal about raw PSI?
**Answer:** When rating drift shifted `product_rating` by $-0.5$, its PSI reached only $0.01$ to $0.04$, while clean sampling noise in `product_main_category` produced a mean baseline PSI of $0.0955$ and max of $0.1438$. This demonstrates that raw PSI is not comparable across features with different numbers of categories or discrete values; high-cardinality categorical features naturally exhibit higher baseline PSI fluctuations under pure sampling noise than bounded numerical features.

### Q6: Why is Experiment D evaluated as rules-only, and what does it not show?
**Answer:** Experiment D was run with rules-only because the primary architectural goal was verifying deterministic routing logic across the pipeline without external API dependencies or non-deterministic variance. It shows that streaming detector and performance monitor signals correctly trigger expected actions (120 of 120 runs matched), but it does not evaluate whether the rules themselves represent an optimal business policy, nor does it benchmark LLM text generation quality.

### Q7: How does the retraining pipeline prevent catastrophic forgetting?
**Answer:** The retraining module enforces a dual-window validation check: candidate models trained on recent drifted batches must improve MAE on the recent window evaluation split by at least 5%, and must not degrade MAE on the historical reference validation split by more than 10%. If historical performance degrades beyond 10%, the candidate model is rejected, preventing the model from overfitting to localized drift while forgetting base patterns.

### Q8: What accounts for the surprising strategy ordering in Seed 44 under medium sudden drift?
**Answer:** In Seed 44, the PSI hovered right around the 0.25 boundary ($0.21$ to $0.33$) across batches 17–19. Because the persistence rule requires two flagged checks in a row and resets on a single clean check, Fixed_1 kept resetting its counter whenever a batch dipped to 0.23. Fixed_4 confirmed drift earlier simply because it checked batch 16, skipped the fluctuating batches 17–19, and checked batch 20 where PSI happened to be 0.31, satisfying its second consecutive check on its own sample points.

### Q9: Why did `psi_only` exhibit high false alarms at batch size $N=150$?
**Answer:** PSI divides the reference distribution into 10 quantile bins, allocating an expected 15 observations per bin at $N=150$. At that small sample size, random Poisson sampling variance causes individual bins to receive substantially more or fewer points than expected, artificially inflating the $(Q_k - P_k) \ln(Q_k / P_k)$ terms and pushing overall PSI above 0.25 despite no underlying population change.

### Q10: What is the limitation of the uniform star rating distribution observed in the raw data?
**Answer:** The raw dataset's `product_rating` feature contains roughly equal 18% to 22% proportions across integer stars 1.0, 2.0, 3.0, 4.0, and 5.0. Authentic customer rating distributions are typically heavily skewed and bimodal (predominantly 5 stars and 1 star). This uniform spread suggests the rating feature in the source academic dataset may have been artificially generated.

### Q11: What would you build or improve with two additional months of engineering time?
**Answer:** I would implement four concrete enhancements:
1. Normalized PSI: Scale raw PSI by feature degrees of freedom or bootstrap empirical p-values so feature rankings are comparable across continuous and high-cardinality columns.
2. Exponentially Weighted Moving Average (EWMA) persistence: Replace the fragile hard-reset counter with an EWMA or CUSUM statistic to eliminate boundary flip-flops.
3. Multi-testing FDR control: Implement Benjamini-Hochberg False Discovery Rate corrections across simultaneous feature tests.
4. Production Kafka/FastAPI streaming harness: Wrap the pipeline in an asynchronous microservice with Redis state caching for production deployment.

### Q12: How was the AI coding assistant utilized during development, and what guardrails did you enforce?
**Answer:** The assistant was used as an autonomous pair-programmer for scaffolding, test writing, and running reproducible benchmarks. Strict ground rules were enforced: all claims required empirical evidence from executed terminal commands, no statistical thresholds were altered to artificially improve numbers, all results were read directly from generated CSV files, and UI constraints (zero emojis, single accent color, no marketing hype) were enforced to maintain production-grade engineering standards.
