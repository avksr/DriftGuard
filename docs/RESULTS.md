# DriftGuard: Empirical Research Results

This document presents the complete numerical results from Experiments A, B, C, and D, transcribed directly from the summary CSV files in `results/`. Each table is accompanied by a neutral interpretation stating what was measured and what the data show.

---

## Main Research Question

**Can adaptive monitoring detect distribution drift almost as fast as checking every batch while running far fewer checks?**

**Answer (neutral summary):**
Adaptive monitoring used 25% to 45% fewer full checks than checking every batch; it matched Fixed_1's delay on large sudden drift (1.0 batch), added about 2 batches on medium sudden drift (4.27 vs 2.37) and about 6 on medium gradual drift (13.30 vs 7.30), and was faster than Fixed_4 on all detected cases; it did not detect small drift under the default rule; no strategy produced false alarms in the no-drift runs.

---

## Experiment A: Detector Sensitivity and False-Alarm Trade-Offs

Source: [`results/experiment_a_summary.csv`](file:///c:/Users/avksr/DRIFT_GUARD/DriftGuard/results/experiment_a_summary.csv)  
Evaluated across 30 random seeds (seeds 42 to 71) on 48-batch streams under rules `ks_only`, `psi_only`, and `both`.

| Scenario | Rule | Detection Rate (Mean ± Std) | False Alarms (Mean ± Std) | Detection Delay (Batches, Mean ± Std) | Total Check Time (s, Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `large_sudden_price` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.00 \pm 0.00$ | $2.84 \pm 0.85$ |
| `large_sudden_price` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.00 \pm 0.00$ | $2.86 \pm 1.39$ |
| `large_sudden_price` | `psi_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.00 \pm 0.00$ | $2.75 \pm 0.92$ |
| `medium_sudden_price` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $2.37 \pm 1.92$ | $2.79 \pm 1.20$ |
| `medium_sudden_price` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.00 \pm 0.00$ | $2.76 \pm 1.05$ |
| `medium_sudden_price` | `psi_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $2.37 \pm 1.92$ | $2.61 \pm 0.96$ |
| `medium_gradual_price` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $7.30 \pm 2.51$ | $2.75 \pm 1.18$ |
| `medium_gradual_price` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $2.30 \pm 0.95$ | $2.64 \pm 0.91$ |
| `medium_gradual_price` | `psi_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $7.30 \pm 2.51$ | $2.75 \pm 1.06$ |
| `small_sudden_price` | `both` | $0.233 \pm 0.430$ | $0.000 \pm 0.000$ | $13.57 \pm 8.90$ | $2.55 \pm 1.14$ |
| `small_sudden_price` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.10 \pm 0.31$ | $2.54 \pm 0.89$ |
| `small_sudden_price` | `psi_only` | $0.233 \pm 0.430$ | $0.000 \pm 0.000$ | $13.57 \pm 8.90$ | $2.61 \pm 0.87$ |
| `medium_sudden_region` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.47 \pm 1.01$ | $2.77 \pm 1.00$ |
| `medium_sudden_region` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.00 \pm 0.00$ | $2.70 \pm 1.11$ |
| `medium_sudden_region` | `psi_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.47 \pm 1.01$ | $2.51 \pm 0.90$ |
| `medium_sudden_bs150` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $1.70 \pm 1.44$ | $2.63 \pm 1.11$ |
| `medium_sudden_bs150` | `ks_only` | $1.000 \pm 0.000$ | $0.067 \pm 0.254$ | $1.00 \pm 0.00$ | $2.64 \pm 0.97$ |
| `medium_sudden_bs150` | `psi_only` | $1.000 \pm 0.000$ | $1.200 \pm 0.805$ | $1.03 \pm 0.81$ | $2.61 \pm 0.87$ |
| `medium_sudden_bs600` | `both` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $3.23 \pm 4.01$ | $3.01 \pm 1.20$ |
| `medium_sudden_bs600` | `ks_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $0.90 \pm 0.31$ | $2.82 \pm 0.99$ |
| `medium_sudden_bs600` | `psi_only` | $1.000 \pm 0.000$ | $0.000 \pm 0.000$ | $3.23 \pm 4.01$ | $3.07 \pm 1.37$ |
| `no_drift` ($N=300$) | `both` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $2.49 \pm 0.94$ |
| `no_drift` ($N=300$) | `ks_only` | $0.000 \pm 0.000$ | $0.133 \pm 0.346$ | — | $2.33 \pm 0.93$ |
| `no_drift` ($N=300$) | `psi_only` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $2.48 \pm 0.85$ |
| `no_drift_bs150` | `both` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $1.21 \pm 0.30$ |
| `no_drift_bs150` | `ks_only` | $0.000 \pm 0.000$ | $0.200 \pm 0.484$ | — | $1.29 \pm 0.33$ |
| `no_drift_bs150` | `psi_only` | $0.000 \pm 0.000$ | $3.300 \pm 1.664$ | — | $1.30 \pm 0.29$ |
| `no_drift_bs600` | `both` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $1.37 \pm 0.26$ |
| `no_drift_bs600` | `ks_only` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $1.44 \pm 0.40$ |
| `no_drift_bs600` | `psi_only` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | — | $1.33 \pm 0.34$ |

### Interpretation
We measured the detection rate, false-alarm frequency, and detection delay across three detector rules and three sample batch sizes over 48 streaming batches. The data show that while `ks_only` detected small drift with a delay of $1.10$ batches, it produced false alarms on uncorrupted batches at $N=150$ and $N=300$, whereas the combined rule `both` yielded 0 false alarms observed in 30 runs across all tested batch sizes at the cost of missing 23 of 30 small drift runs where PSI remained below 0.25.

---

## Experiment B: Fixed versus Adaptive Monitoring Strategies

Source: [`results/experiment_b_summary.csv`](file:///c:/Users/avksr/DRIFT_GUARD/DriftGuard/results/experiment_b_summary.csv)  
Evaluated across 30 random seeds (seeds 42 to 71) on 48-batch streams under default rule `both`.

| Scenario | Strategy | Detection Delay (Mean ± Std) | Missed Detection Rate (Mean ± Std) | False Alarms (Mean ± Std) | Full Checks Run (Mean ± Std) | Check Rate % (Mean ± Std) | Total Runtime (s, Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `large_sudden` | `Adaptive` | $1.00 \pm 0.00$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $36.0 \pm 0.0$ | $75.00 \pm 0.00$ | $2.42 \pm 0.76$ |
| `large_sudden` | `Fixed_1` | $1.00 \pm 0.00$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $3.11 \pm 0.98$ |
| `large_sudden` | `Fixed_4` | $4.00 \pm 0.00$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.74 \pm 0.27$ |
| `medium_sudden` | `Adaptive` | $4.27 \pm 4.90$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $32.1 \pm 3.6$ | $66.94 \pm 7.54$ | $2.15 \pm 0.75$ |
| `medium_sudden` | `Fixed_1` | $2.37 \pm 1.92$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $2.96 \pm 0.92$ |
| `medium_sudden` | `Fixed_4` | $10.29 \pm 8.03$ | $0.067 \pm 0.254$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.76 \pm 0.22$ |
| `medium_gradual` | `Adaptive` | $13.30 \pm 5.70$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $26.2 \pm 3.0$ | $54.65 \pm 6.33$ | $1.72 \pm 0.59$ |
| `medium_gradual` | `Fixed_1` | $7.30 \pm 2.51$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $3.02 \pm 0.85$ |
| `medium_gradual` | `Fixed_4` | $19.26 \pm 5.55$ | $0.100 \pm 0.305$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.78 \pm 0.24$ |
| `categorical_medium_sudden` | `Adaptive` | $2.07 \pm 2.33$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $35.5 \pm 1.4$ | $73.96 \pm 2.88$ | $2.04 \pm 0.70$ |
| `categorical_medium_sudden` | `Fixed_1` | $1.47 \pm 1.01$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $2.78 \pm 0.88$ |
| `categorical_medium_sudden` | `Fixed_4` | $5.20 \pm 3.95$ | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.69 \pm 0.25$ |
| `small_sudden` | `Adaptive` | $11.33 \pm 11.68$ | $0.900 \pm 0.305$ | $0.000 \pm 0.000$ | $13.5 \pm 2.3$ | $28.12 \pm 4.89$ | $0.86 \pm 0.32$ |
| `small_sudden` | `Fixed_1` | $13.57 \pm 8.90$ | $0.767 \pm 0.430$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $2.94 \pm 0.84$ |
| `small_sudden` | `Fixed_4` | $24.00 \pm 0.00$ | $0.967 \pm 0.183$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.75 \pm 0.24$ |
| `no_drift` | `Adaptive` | — | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.74 \pm 0.28$ |
| `no_drift` | `Fixed_1` | — | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $48.0 \pm 0.0$ | $100.00 \pm 0.00$ | $2.87 \pm 0.85$ |
| `no_drift` | `Fixed_4` | — | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | $12.0 \pm 0.0$ | $25.00 \pm 0.00$ | $0.74 \pm 0.25$ |

### Interpretation
We evaluated detection delay, missed detections, and computational check savings across `Fixed_1`, `Fixed_4`, and `Adaptive(4,1,3)` monitors over five drift conditions. The numbers show that Adaptive monitoring matched Fixed_1's 1.0-batch delay on large sudden drift while saving 25% of full checks, added 1.9 batches of delay on medium sudden drift while saving 33.1% of checks, and added 6.0 batches on gradual drift while saving 45.4% of checks, whereas Fixed_4 missed 2 of 30 medium sudden runs and 3 of 30 gradual runs.

### Threshold Dynamics (Seed 44 Diagnostic Trace)
In Seed 44 under medium sudden drift, the observed detection delays followed an unexpected order: Fixed_1 confirmed drift after 7 batches (batch 23), Fixed_4 after 4 batches (batch 20), and Adaptive after 12 batches (batch 28). Diagnostic inspection revealed the underlying mechanism:
1. **Fluctuation Near Threshold:** After drift injection at batch 16, true feature PSI hovered near the $0.25$ threshold boundary ($0.21$ to $0.33$). Sampling noise caused the flag condition to alternate between true and false from batch to batch.
2. **Persistence Reset:** The persistence rule requires two consecutive flagged checks to confirm an alert. A single unflagged check resets the persistence counter to zero.
3. **Adaptive Mode Reversion:** Because batches 17, 18, and 19 fell just below threshold ($0.21$ to $0.24$), Adaptive observed three clean checks and reverted to normal mode ($k=4$), delaying its next check while drift was still active.
4. **Sampling Timing of Fixed_4:** Fixed_4 evaluated batches 16 and 20 directly, skipping batches 17–19. Both batch 16 and batch 20 exceeded $0.25$, satisfying the two-consecutive rule earlier than Fixed_1, which was repeatedly reset by the intermediate batches. Hard thresholds combined with consecutive persistence counters create sensitivity to batch sampling points near the decision boundary.

---

## Experiment C: Feature Ranking Accuracy (Precision@k)

Source: [`results/experiment_c_summary.csv`](file:///c:/Users/avksr/DRIFT_GUARD/DriftGuard/results/experiment_c_summary.csv)  
Evaluated across 30 random seeds (seeds 42 to 71) across 5 evaluation batches post-drift.

| Scenario | Precision@k (Mean ± Std) | Rank 1 Correct (Mean ± Std) | Evaluation Batches Evaluated |
| :--- | :--- | :--- | :--- |
| `1feat_price_large_sudden` | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 5.0 |
| `1feat_price_medium_sudden` | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 5.0 |
| `1feat_price_medium_gradual` | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 5.0 |
| `1feat_region_medium_sudden` | $1.000 \pm 0.000$ | $1.000 \pm 0.000$ | 5.0 |
| `1feat_rating_medium_sudden` | $0.000 \pm 0.000$ | $0.000 \pm 0.000$ | 5.0 |
| `2feat_price_rating_medium_sudden` | $0.500 \pm 0.000$ | — | 5.0 |
| `3feat_price_rating_region_medium_sudden` | $0.667 \pm 0.000$ | — | 5.0 |

### Interpretation
We measured the precision of top-$k$ feature rankings against the ground truth of truly changed columns across single, two-, and three-feature drift scenarios. The results show that while `retail_price` and `Region` achieved $1.000$ precision@1, rating drift achieved $0.000$ precision because its post-perturbation PSI ($0.004$ to $0.040$) remained lower than the clean sampling noise floor of categorical columns such as `product_main_category` (mean clean PSI $0.0955$).

### Rating Injection Protocol & Feature Noise Floor
- **Rating Injection Protocol:** `product_rating` is always perturbed additively (-0.5, clipped to 1 to 5) in every scenario. Earlier multi-feature results were invalid because ratings had been multiplied (producing invalid rating values up to 7); the old files are archived in `results/archive/` (`experiment_c_old.csv`, `experiment_c_summary_old.csv`, `experiment_c_old.png`).
- **Feature Noise Floor (Batch Size 300, 30 Clean Batches):**

| Feature | Type | Mean Clean PSI | Max Clean PSI | Std Clean PSI |
| :--- | :--- | :--- | :--- | :--- |
| `product_main_category` | categorical | 0.0955 | 0.1438 | 0.0218 |
| `BrandGroup` | categorical | 0.0399 | 0.1026 | 0.0201 |
| `retail_price` | numeric | 0.0363 | 0.0937 | 0.0172 |
| `product_rating` | numeric | 0.0148 | 0.0354 | 0.0099 |
| `Region` | categorical | 0.0098 | 0.0405 | 0.0082 |
| `OrderMonth` | categorical | 0.0083 | 0.0245 | 0.0070 |

*Note on PSI Comparability:* Raw PSI is not comparable across features with different numbers of categories or feature types, and was not normalised in this implementation. High-cardinality categorical features naturally exhibit higher baseline sampling noise than bounded numerical features.

---

## Experiment D: End-to-End Decision Agent Evaluation

Source: [`results/experiment_d_summary.csv`](file:///c:/Users/avksr/DRIFT_GUARD/DriftGuard/results/experiment_d_summary.csv)  
Evaluated across 30 random seeds (seeds 42 to 71) streaming batches 16 through 21 into the rules engine.

| Scenario | Total Runs | Premise-OK Runs | Match Rate (All Runs) | Match Rate (Premise-OK) | Propose Retrain Total | Propose Retrain Expected | Unnecessary Retrain Rate | Chosen Action Counts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `drift_perf_fine` | 30 | 30 | 1.000 | 1.000 | 0 | 0 | 0.000 | INVESTIGATE: 30 |
| `drift_and_degraded` | 30 | 30 | 1.000 | 1.000 | 30 | 30 | 0.000 | PROPOSE_RETRAIN: 30 |
| `perf_worse_no_drift` | 30 | 30 | 1.000 | 1.000 | 0 | 0 | 0.000 | INVESTIGATE: 30 |
| `drift_labels_hidden` | 30 | 30 | 1.000 | 1.000 | 0 | 0 | 0.000 | REQUEST_LABELS: 30 |

### Interpretation & Evaluation Caveats
We evaluated whether streaming detector and performance monitor outputs into the deterministic rules table selected the expected action across four synthetic operating conditions. In all 120 runs across 30 seeds, the selected action matched the rules table expectation with zero unnecessary retrain proposals.

**Caveats:**
1. **Routing Check, Not Policy Optimality:** Evaluated as rules-only; expected actions come from the same rules table, so it is an end-to-end routing check and not an independent test of appropriateness.
2. **Target Shift Dominance:** The "drift and degraded" scenario combines price drift with a target shift (label shift 1.8), so most of the performance drop comes from the target shift rather than input feature drift alone.
3. **Pilot Calibration:** Scenario parameters were chosen after a 3-seed pilot so that each scenario premise holds.
4. **LLM Layer Not Evaluated:** The LLM layer is implemented with a rule-based fallback but was not run or quantitatively evaluated in these benchmarks.

---

## Runtime Benchmarks and Platform Environment

- **Development & Verification Environment:** The code was developed and tested on Windows 11 with Python 3.14.6 only. The project is a research prototype, not a production tool.
- **Experimental Pipeline Execution Times:**
  - The full 30-seed pipeline took roughly 40 minutes in the working folder.
  - The `--quick` run (2 seeds) took 485 s in the fresh clone on the author's laptop.
  - Note: Execution times vary by machine depending on processor and disk I/O.

