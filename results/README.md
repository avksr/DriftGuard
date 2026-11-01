# DriftGuard Results Directory Guide

This directory contains the empirical research artifacts produced by the automated experimental pipeline (`experiments/run_all.py`), including raw evaluation CSV files, statistical summaries, publication-ready visualization charts, and execution manifests.

---

## Benchmark Artifacts Summary

| File | Experiment | Contents | Seeds | Generating Commit (from Manifest) |
| :--- | :--- | :--- | :--- | :--- |
| `experiment_a.csv` | Experiment A | Raw per-run detection logs across 10 scenarios and 3 flag rules (900 rows) | 30 | `fd11597` |
| `experiment_a_summary.csv` | Experiment A | Mean ± std for detection rate, delay, false alarms, and check runtime | 30 | `fd11597` |
| `experiment_a.png` | Experiment A | Multi-panel plot of detection delay, false alarms, and sample size trade-offs | 30 | `fd11597` |
| `experiment_b.csv` | Experiment B | Raw per-run logs for `Fixed_1`, `Fixed_4`, and `Adaptive(4,1,3)` across 5 scenarios (450 rows) | 30 | `62f7561` |
| `experiment_b_summary.csv` | Experiment B | Mean ± std for detection delay, missed detection rate, check counts, and check savings | 30 | `62f7561` |
| `experiment_b.png` | Experiment B | Comparative bar charts of detection delay vs check savings across drift shapes | 30 | `62f7561` |
| `experiment_c.csv` | Experiment C | Top-$k$ feature rankings and precision@$k$ across 7 drift scenarios (210 rows) | 30 | `97885d0` |
| `experiment_c_summary.csv` | Experiment C | Mean ± std precision@$k$ and rank-1 identification accuracy | 30 | `97885d0` |
| `experiment_c.png` | Experiment C | Feature ranking precision@$k$ bar chart comparing single and multi-feature scenarios | 30 | `97885d0` |
| `experiment_d.csv` | Experiment D | End-to-end decision agent routing logs across 4 streaming scenarios (120 rows) | 30 | `fd11597` |
| `experiment_d_summary.csv` | Experiment D | Action match rate, premise verification rate, and unnecessary retrain rate | 30 | `fd11597` |
| `experiment_d.png` | Experiment D | Bar chart showing 100% action routing fidelity and 0% unnecessary retrain rate | 30 | `fd11597` |
| `run_manifest.json` | All (A–D) | Complete JSON execution manifest recording timestamps, git commit hashes, and config parameters | 30 | `fd11597` |
| `incidents.csv` | Dashboard / Logger | Structured CSV export of logged simulation incidents and retraining decisions | N/A | Interactive |
| `incidents.json` | Dashboard / Logger | Detailed JSON export of logged simulation incidents with rule evaluations | N/A | Interactive |

*Note on Experiment A Commit Lineage:* The 8 primary drift scenarios were executed under commit `62f7561`. The two 48-batch clean streams at alternative batch sizes (`no_drift_bs150` and `no_drift_bs600`) were appended in commit `fd11597` to provide fair, uncorrupted false-alarm baselines across all sample sizes without altering existing numbers.

---

## Notes on Archived Files (`results/archive/`)

The `results/archive/` folder preserves superseded experimental outputs for transparency and auditability:

1. **`experiment_c_old.csv`, `experiment_c_summary_old.csv`, `experiment_c_old.png`:**
   - **Reason for Archiving:** Superseded due to a rating injection bug in earlier multi-feature scenarios. In those early runs, `product_rating` was inadvertently perturbed by multiplying existing ratings by a factor (producing out-of-domain ratings up to 7.00).
   - **Current Implementation:** Replaced in commit `97885d0` with additive, bounded shifts (`shift_additive`: -0.5 clipped strictly to $[1.0, 5.0]$) matching the single-feature rating scenario.

2. **`experiment_d.csv`, `experiment_d_summary.csv`, `experiment_d.png` (archived single-batch run):**
   - **Reason for Archiving:** Initial pilot evaluated decision logic on a single isolated batch (batch 16). Because performance degradation requires two consecutive degraded batches to confirm `degraded` status, the model performance premise was unconfirmed at batch 16.
   - **Current Implementation:** Replaced in commit `fd11597` with a streaming harness evaluating batches 16 through 21, allowing both drift persistence and performance degradation premises to hold simultaneously.

3. **`incidents.csv`, `incidents.json` (archived):**
   - Preliminary incident logs generated during initial development and smoke testing of the decision agent and dashboard.

