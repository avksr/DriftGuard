# DriftMonitor — Project Specification (Simplified Master Version)

**Full title:** DriftMonitor: Adaptive Monitoring of Machine Learning Models Under Data Drift, with an LLM-Assisted Decision Layer

**Purpose of this file:** (1) a permanent record of what the project is and why every decision was made, and (2) the single source of truth to give to Antigravity so it can build the project step by step.

**Level:** written for a second-year student. Every advanced idea has been either explained simply or removed.

**Dataset (updated):** Indian E-Commerce Orders (8,906 orders, file `E-Commerce.csv` from the `mvinayak96/ecommerce-sales-analysis` GitHub repository). This replaces the UCI Online Retail dataset used in the first version. See Section 4.

---

## 0. How to read / use this file

- Sections 1–3: understand the idea (for you, and for your report/viva).
- Sections 4–9: the exact technical design (for Antigravity).
- Section 10: week-by-week plan.
- Section 11: how to talk to Antigravity, with ready-made prompts.
- Section 12: limitations, honesty rules, glossary.

If you ever get confused during the project, come back to the **one-sentence summary** below.

> **One-sentence summary:** We build a system that watches incoming data for a trained ML model, detects when the data has changed, finds which columns changed, checks the model less often when things are calm and more often when things look suspicious, and then an LLM writes a plain-English incident report and recommends one safe action — and we run experiments to measure whether this "adaptive" checking is actually better than checking on a fixed schedule.

---

## 1. The idea in plain words

### 1.1 The problem
A machine learning model is trained on past data. In the real world, new data slowly (or suddenly) starts looking different from the training data: customers buy different quantities, prices change, a new country starts ordering more. The model was never trained for this, so its predictions may get worse. This is called **drift**.

Companies therefore *monitor* their models. Two simple weaknesses of basic monitoring:

1. **Fixed schedule problem:** checking every batch wastes computation when nothing is happening; checking rarely (say every 4th batch) reacts late when something does happen.
2. **"Alert but no explanation" problem:** a plain alert says "drift detected" but not *which feature*, *how bad*, or *what to do next*.

### 1.2 Our answer
DriftMonitor does five things:

1. **Detects** whether new data differs from reference data (two statistical methods: KS test and PSI).
2. **Explains** which features changed most (a ranked list).
3. **Adapts** how often it checks (rare checks when calm, frequent checks when suspicious) — and we measure whether this helps.
4. **Recommends** one action from a small safe list, using rules plus an LLM that writes the explanation.
5. **Recovers** carefully: if retraining is proposed, a new model is accepted only if it passes a fixed test.

### 1.3 The research part (kept small and honest)
The main question:

> **Can adaptive monitoring (checking more often only when there is evidence of trouble) detect drift almost as fast as checking every batch, while doing far fewer checks?**

We answer it by running experiments with *artificially injected* drift where we know the truth (when drift starts and which columns were changed). We report real numbers, including cases where adaptive monitoring does **not** help. A negative result reported honestly is still a good result.

---

## 2. What was kept, simplified and removed (and why)

The two earlier documents (DriftMonitor and DriftGuard) were merged. Here is what happened to each idea.

| Idea from earlier docs | Decision | Reason |
|---|---|---|
| KS test + PSI | **Kept** (core) | Standard, easy to explain, implementable with SciPy/NumPy |
| Feature ranking (which columns drifted) | **Kept**, based on PSI/KS scores | Simple and testable against ground truth |
| Fixed vs adaptive monitoring | **Kept** (main research focus) | This is the research contribution |
| Rule-based adaptive policy | **Kept** | Transparent, easy to implement and explain |
| Hybrid agent (rules + LLM) | **Kept, simplified** | Rules decide what is *allowed*; the LLM explains and picks among allowed actions |
| Controlled retraining with acceptance test | **Kept, simplified** | One simple accept/reject rule |
| Streamlit dashboard | **Kept** | Needed to demonstrate the project |
| UCI Online Retail dataset | **Replaced** by the Indian E-Commerce Orders dataset | Your choice (Indian-market relevance); one flat table, small enough to run experiments fast (Section 4) |
| Wasserstein distance | **Removed** | Third detector adds work, little extra insight |
| SHAP feature attribution | **Removed** | Hard to explain; PSI-based ranking is enough |
| Evidently AI library | **Removed** | We implement KS/PSI ourselves so we understand them |
| LangChain / LiteLLM | **Removed** | One small function that calls the LLM is enough |
| Time-series forecasting task | **Removed** | Too complex; replaced by a simple tabular regression task |
| "Concept drift" detection | **Explained only** | We simulate one scenario but do not build a concept-drift detector |
| Reinforcement learning for monitoring | **Removed** | Out of scope |
| Multi-agent, Kubernetes, auto-deployment | **Removed** | Out of scope |
| Streamlit Cloud deployment | **Optional stretch** | Do it only if everything else is finished |
| Strong originality claims | **Removed** | We only claim a careful comparison, not a new invention |
| Mean/MAPE metrics | **MAE only (+RMSE optional)** | MAPE breaks when values are near zero |

---

## 3. Core concepts explained simply

### 3.1 Reference data vs incoming batches
- **Reference data:** the "normal" data the model was built on. It is our yardstick.
- **Batch:** a chunk of new data (for us: a fixed number of rows, e.g. 300) that arrives one after another, like a stream.

### 3.2 Data drift (input drift)
The distribution of an input column in new batches differs from the reference. Example: average `retail_price` suddenly doubles.

### 3.3 Performance degradation
The model's error gets worse. We can only know this if the true answers (labels) are available. **Important rule of the project: drift does not automatically mean the model got worse.** The system must say "drift detected, performance not verified" when labels are missing.

### 3.4 Concept drift (explained, one simulated scenario only)
The *relationship* between inputs and target changes (e.g., the same product, price and region now get a different discount than before). Inputs may look the same, but the model gets worse. We simulate this once, to test that the system recommends a sensible action when the model is worse but no feature drifted.

### 3.5 KS test (Kolmogorov–Smirnov)
Compares two samples of a **numeric** column. It measures the biggest gap between their cumulative distributions (the **KS statistic**, 0 to 1) and gives a **p-value**.
- Small p-value ⇒ the two samples probably come from different distributions.
- Caution: with very large samples, even tiny, unimportant differences give tiny p-values. That is why we keep batches at a fixed moderate size and *also* use PSI.

### 3.6 PSI (Population Stability Index)
Split the reference values into bins (e.g., 10 bins by quantiles). Count the share of reference rows and of new rows in each bin. Then:

```
PSI = sum over bins of (new_share - ref_share) * ln(new_share / ref_share)
```

Common rule of thumb (put in config, not hard-coded): `< 0.1` stable, `0.1 – 0.25` moderate change, `> 0.25` major change.
- Empty bins would cause log(0), so we add a tiny number (e.g., 1e-4) to every share.
- **The bin edges must always come from the reference data**, never from the new batch.
- For **categorical** columns (e.g., Region), the "bins" are simply the categories.

### 3.7 Adaptive monitoring
The system has two modes:
- **NORMAL mode:** run a full drift check only every N-th batch (cheap).
- **ALERT mode:** run a full check on every batch (careful).

It switches NORMAL → ALERT when a check finds drift signals, and ALERT → NORMAL after several clean checks in a row. It never hides or deletes alerts to look better.

### 3.8 Hybrid agent
"Agentic" here means a **small, controlled** AI helper — not a free chatbot:
- **Python rules** compute the facts and decide which actions are *allowed*.
- **The LLM** reads a structured JSON report, writes a plain-English incident summary, and picks one action from the allowed list.
- If the LLM answers something outside the allowed list, or fails, the system uses the rule-based action. The LLM can never override the numbers.

### 3.9 Root-cause investigation (what we do and do not claim)
We only claim: "these columns changed the most, measured by PSI/KS". We do **not** claim *why* (business reason) — that would need evidence we do not have. Reports separate **Observed facts** from **Possible explanations (hypotheses)**.

---

## 4. Dataset, preprocessing and prediction task

### 4.1 Dataset (updated: Indian E-Commerce Orders)

- **File:** `E-Commerce.csv` from the GitHub repository `mvinayak96/ecommerce-sales-analysis`.
- **Size (per the repository README):** 8,906 orders, 17 columns. The README credits the Great Learning DSBA program as the source.
- **Columns named in the README:** `Customer_uniq_id`, `Region` (North / East / South / West), `Order_Date`, `Expected_Delivery_Date`, `Delivered_Date`, `product_name`, `product_main_category`, `product_subcategory`, `retail_price`, `discounted_price`, `product_rating`, `Brand`, plus two long-text columns (`product_specifications`, `description`) that we drop. The remaining columns are not named in the README.
- **What is verified and what is not (important):** the README and repository page were checked, but the CSV file itself could **not** be opened during planning. So the exact column names, the date range, the missing values and the price formats are **unverified**. That is why the first step of the project is a mandatory data audit (Section 4.2). Do not trust any column name in this spec until the audit confirms it.
- **Licence and sharing:** this looks like a course dataset with no stated licence. Use it for your academic project, cite the source, and do **not** commit the CSV to your own GitHub repository. Put download instructions in `data/README.md` instead.

**Why this dataset and not the other candidates**

| Candidate | What the sources show | Decision |
|---|---|---|
| Indian E-Commerce Sales & Customer Analytics (Kaggle) | Described as several linked tables (orders, customers, products, payments, shipments, returns, reviews). The Kaggle page content could not be read. | **Not chosen for the core.** Joining many tables adds work and risk for a one-semester-level project. Good future extension. |
| Indian Retail Sales Dataset 2023–2025 (Rustam Poonia) | README describes about 10,000 orders (9,994 after removing duplicates), 2023-01-01 to 2025-12-31, with sales, profit, discount, region, category. But the repository contains only the README and a SQL file, **no CSV**. Some patterns are unusually uniform (almost equal order counts per region, near-constant average discounts), which is typical of generated data, though this cannot be confirmed. | **Backup.** Use only if you find the actual data file and the audit of the main dataset fails. |
| Indian E-Commerce Orders (mvinayak96) | The CSV is in the repository, one flat table, 8,906 rows, with order/delivery dates, prices, rating, brand, category, region. | **Chosen.** Weaknesses: small, date range unverified, course-dataset provenance. |

### 4.2 Data audit (mandatory first step, before any other code)

`src/data_audit.py` loads the CSV and prints a report. It must answer:

1. Shape, column names, data types.
2. Missing values per column (count and percent).
3. Number of duplicate rows.
4. For `retail_price` and `discounted_price`: are they numbers or text (e.g., with commas or currency symbols)? Min, max, mean, number of zeros/negatives, number of rows where `discounted_price > retail_price`.
5. For `Order_Date`, `Delivered_Date`: date format, earliest and latest date, number of unparseable values.
6. Number of unique values in `Region`, `product_main_category`, `product_subcategory`, `Brand`, and the top 10 brands with counts.
7. Distribution of `product_rating` (range, how many missing, how many distinct values).
8. The distribution of the two candidate targets (Section 4.5): `discount_pct` and `delivery_days`.

**Decision rules after the audit (write the outcome in the README):**

| Audit finding | Action |
|---|---|
| A needed column is missing or has more than about 20% missing values | Drop that feature, or switch the target to the fallback |
| `Order_Date` missing, unparseable, or covering less than about two months | Skip the natural time-based exploration (Section 4.6, option 2). The main experiments are unaffected |
| Fewer than 3 usable features (after cleaning) | Stop and discuss; consider the backup dataset |
| `discount_pct` is almost constant (standard deviation near zero) | Use the fallback target `delivery_days` |

### 4.3 Cleaning rules (apply identically to every batch)

1. Drop `product_specifications` and `description` (long text, unused).
2. Convert price columns to numbers and parse dates (check whether day or month comes first; use the audit result).
3. Drop exact duplicate rows.
4. Drop rows with missing `retail_price`, `discounted_price`, `Region`, or `product_main_category`.
5. Drop rows with `retail_price <= 0` or `discounted_price <= 0`.
6. Rows with `discounted_price > retail_price`: count them in the audit, then drop them and record the count.
7. Cap extreme values of `retail_price` at the 99.5th percentile **computed on the reference data only**.
8. Save the cleaned file so every later step uses exactly the same data.

### 4.4 Analytical unit and features

**Unit:** one row = one order line of the dataset.

Proposed features (confirm against the audit; keep them few so results are easy to explain):

| Feature | Type | How it is made |
|---|---|---|
| `retail_price` | numeric | as given (capped) |
| `product_rating` | numeric | as given (rows with missing rating: fill with the reference median, or drop the feature if too many are missing) |
| `Region` | categorical | as given |
| `product_main_category` | categorical | as given |
| `BrandGroup` | categorical | the 10 most frequent brands in the reference data keep their name; all others (and brands never seen in the reference) become `Other` |
| `OrderMonth` | categorical | month (1–12) from `Order_Date`, only if the dates pass the audit |

`discounted_price` is **not** a feature (it is used to build the target, so using it would leak the answer).

### 4.5 Prediction task (simple on purpose)

- **Main target:** `discount_pct = (retail_price - discounted_price) / retail_price`, a number between 0 and 1. Meaning: "how large a discount will this order line get?" Error is reported as MAE in percentage points.
- **Fallback target (if the audit rules say so):** `delivery_days = Delivered_Date - Order_Date` in days (drop rows where it is missing or negative). Features are then the same list; make sure no delivery date column is used as a feature.
- **Model:** `RandomForestRegressor` or `GradientBoostingRegressor` from scikit-learn, fixed `random_state`.
- **Metric:** MAE. RMSE optional.
- **Sanity check (required):** also compute the MAE of a "predict the average" dummy model. The real model should beat the dummy by at least about 10%. If it does not, change the target or features before going further, because a model that learned nothing makes performance monitoring meaningless.
- The model does not have to be strong. It has to be a stable baseline whose error we can watch.

### 4.6 Two ways of building batches

1. **Controlled (synthetic) experiments: the main evidence.**
   - Take all cleaned rows and split them **randomly** 50/50 into a *reference half* and a *clean pool*.
   - Because the split is random, the clean pool truly has **no drift** compared with the reference. This guarantees a trustworthy "no drift" baseline.
   - Batches (default 300 rows) are sampled from the clean pool. Rows are not repeated inside one batch, but the same row can appear in different batches (the pool is small); mention this in the report.
   - Drift is then injected into chosen batches (Section 5.2).
2. **Natural time-based exploration: a bonus, not the main evidence (only if the audit passes the date check).**
   - Reference = the earliest part of the data; incoming batches = later orders in date order.
   - Show what the system finds on real data. Do **not** claim any of it is "true drift", because we have no ground truth. Describe it as exploratory.

### 4.7 Avoiding leakage

- The model is trained only on the reference half (80% train, 20% held-out for the baseline MAE).
- Bin edges, price caps, brand groups and median fill values are all computed from the reference data only.
- Never use rows from the clean pool or from later dates to prepare anything for the reference.

---

## 5. System design (modules)

### 5.1 Repository structure

```text
DriftMonitor/
├── app.py                      # Streamlit dashboard
├── config.yaml                 # all thresholds and settings
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example                # API key placeholder only
├── PROJECT_SPEC.md             # this file
├── data/
│   └── README.md               # where to download the dataset
├── models/                     # saved models (gitignored)
├── src/
│   ├── data_audit.py           # Week-1 check of the raw CSV (columns, dates, missing values)
│   ├── data_loader.py          # load, clean, features, reference/clean-pool split
│   ├── batch_simulator.py      # make batches, inject drift, keep ground truth
│   ├── baseline_model.py       # train/evaluate/save the baseline model
│   ├── quality_checks.py       # schema, missing values, invalid values
│   ├── drift_detector.py       # KS, PSI, categorical comparison, feature ranking
│   ├── adaptive_monitor.py     # fixed and adaptive monitoring logic
│   ├── performance_monitor.py  # MAE per batch, degradation rule
│   ├── decision_agent.py       # rules + LLM explanation + guardrails
│   ├── retrainer.py            # train candidate model + accept/reject rule
│   └── incident_logger.py      # save incidents/decisions to a CSV/JSON log
├── experiments/
│   └── run_experiments.py      # runs Experiments A–D, saves results tables/plots
├── tests/
│   ├── test_drift_detector.py
│   ├── test_adaptive_monitor.py
│   ├── test_decision_agent.py
│   └── test_retrainer.py
└── results/                    # CSV tables and figures from experiments
```

Simplify further if time runs short (e.g., merge `quality_checks.py` into `data_loader.py`).

### 5.2 `batch_simulator.py` — creating batches and injecting drift
**Batches:** fixed size (default 300 rows), sampled from the clean pool with a seed.

**A "run"** = a sequence of e.g. 48 batches. A scenario decides from which batch (e.g., batch 16) drift starts.

**Drift types to implement (each is a small function):**

| Name | What it does |
|---|---|
| `shift_mean(feature, factor)` | multiplies a numeric feature (e.g., `retail_price × 1.5`) |
| `shift_additive(feature, delta)` | adds a constant to a numeric feature (e.g., `product_rating − 0.6`, kept inside the valid range) |
| `change_spread(feature, factor)` | stretches values around the mean |
| `shift_category_mix(feature, drop)` | reassigns a share of the rows of the most frequent category to the other categories (e.g., lower the most common `Region`'s share by 25 percentage points) |
| `label_shift(factor)` | multiplies the target `discount_pct` (capped at 1) (simulates concept drift: inputs same, outcomes different) |

**Drift shapes:** `sudden` (full strength from start batch) and `gradual` (strength grows linearly from 0 to full over 10 batches).

**Drift sizes:** `small`, `medium`, `large` (set factors in `config.yaml`).

**Ground truth saved with every run:** drift start batch, which features were changed, drift type. This is what lets us compute detection delay and ranking precision.

### 5.3 `quality_checks.py`
For each batch check: expected columns exist, data types correct, missing-value share below a limit, no non-positive prices, `discounted_price` never above `retail_price`, no unseen category labels beyond a small tolerance.
Output: `{"passed": true/false, "issues": [...]}`.
If a batch fails, mark it "invalid", skip drift testing for it, and show it in the dashboard (so bad data is not mistaken for drift).

### 5.4 `drift_detector.py`
Functions:
- `ks_test(ref_values, new_values)` → `(statistic, p_value)`
- `psi_numeric(ref_values, new_values, n_bins=10)` → float (bin edges from reference quantiles; handle duplicate edges)
- `psi_categorical(ref_series, new_series)` → float
- `chi_square_categorical(ref_series, new_series)` → p-value (the categorical counterpart of the KS p-value)
- `detect_drift(ref_df, batch_df, config)` → a dictionary:

```json
{
  "batch_id": 27,
  "features": {
    "retail_price": {"ks_stat": 0.31, "ks_p": 0.0001, "psi": 0.42, "flagged": true},
    "Region":       {"chi2_p": 0.40, "psi": 0.03, "flagged": false}
  },
  "ranking": ["retail_price", "product_main_category", "..."],
  "drift_detected": true
}
```

**Feature flag rule (default, configurable):** flagged if `PSI ≥ 0.25` **and** KS/chi-square `p < 0.01`. Config can switch the rule to `"ks_only"`, `"psi_only"`, `"both"` — needed for Experiment A.

**Ranking:** sort features by PSI (highest first); break ties with the KS statistic.

**Batch-level drift:** `drift_detected = at least one feature flagged`.

### 5.5 `adaptive_monitor.py`
All strategies use the **same** definitions so the comparison is fair.

- **Full check:** run `detect_drift` on the batch. This is the expensive thing we count.
- **Flagged check:** a full check where `drift_detected = True`.
- **Confirmed alert:** two flagged checks in a row. (This persistence rule reduces false alarms and is applied to every strategy.)

**Strategies:**

| Strategy | Behavior |
|---|---|
| `Fixed(k=1)` | full check on every batch |
| `Fixed(k=4)` | full check on every 4th batch |
| `Adaptive` | NORMAL mode: check every 4th batch. When a check is flagged → switch to ALERT mode: check every batch. After 3 consecutive clean checks in ALERT mode → return to NORMAL. |

Each strategy returns, per run: the list of checks done, which were flagged, the batch of the first confirmed alert, number of full checks, and total time spent inside `detect_drift`. Every mode switch is logged with its reason.

### 5.6 `performance_monitor.py`
- Stores the baseline MAE (from the reference validation part).
- For each batch with labels: compute MAE.
- **Degradation rule:** degraded if MAE > 1.2 × baseline MAE for **2 consecutive labelled batches** (numbers in config).
- Supports `labels_available = False` (returns status `"unverified"`) to simulate delayed labels.

### 5.7 `decision_agent.py` — the hybrid agent

**Step 1 — Rules (Python, authoritative).** Inputs: drift result, quality result, performance status. Output: the set of allowed actions and a default action.

Allowed actions (the only ones that exist):
`CONTINUE`, `INVESTIGATE`, `INTENSIFY_MONITORING`, `REQUEST_LABELS`, `PROPOSE_RETRAIN`

| Situation | Default action | Allowed actions |
|---|---|---|
| No drift, performance normal | `CONTINUE` | CONTINUE |
| Drift detected, labels unavailable | `REQUEST_LABELS` | REQUEST_LABELS, INTENSIFY_MONITORING |
| Drift detected, labels available, performance normal | `INVESTIGATE` | INVESTIGATE, INTENSIFY_MONITORING |
| Drift detected, performance degraded | `PROPOSE_RETRAIN` | PROPOSE_RETRAIN, INVESTIGATE |
| No feature drift, performance degraded | `INVESTIGATE` | INVESTIGATE, PROPOSE_RETRAIN |
| Batch failed quality checks | `INVESTIGATE` | INVESTIGATE |

**Step 2 — LLM (explanation and choice).**
Send the LLM a JSON report (batch id, top features and scores, quality status, performance status, allowed actions). Ask for JSON back:

```json
{
  "summary": "2–4 sentences, plain English",
  "observed_facts": ["..."],
  "possible_explanations": ["... (hypotheses, not facts)"],
  "recommended_action": "ONE of the allowed actions",
  "follow_up_checks": ["..."]
}
```

Prompt rules: use only numbers given; never invent causes; never claim performance dropped unless the report says so.

**Step 3 — Guardrails (Python).**
- If `recommended_action` is not in the allowed list → use the default action.
- If the LLM call fails, times out, or returns invalid JSON → use the default action and a template-generated summary (so the project works with **no API key**).
- The LLM never computes drift statistics.

**LLM provider:** keep it behind **one function** `call_llm(prompt) -> str` so you can swap providers. Pick whichever you can use for free or cheaply (e.g., a Gemini API key or an OpenAI key). Check current pricing/free-tier limits yourself before choosing. Store the key in `.env`, never in code or GitHub.

### 5.8 `retrainer.py`
Retraining happens **only** when the action is `PROPOSE_RETRAIN` (never automatically on every alert, and never auto-deployed).

Procedure:
1. Log why retraining is considered.
2. Build training data = reference-half data + the most recent labelled batches (chronologically earlier than the validation window).
3. Train a **candidate** model (same model type, same settings).
4. Evaluate current model and candidate on the **same** validation window made of the most recent labelled batches that were *not* used for training.
5. **Acceptance rule (in config):** accept only if
   - candidate MAE is at least **5% lower** than current MAE on the recent validation window, **and**
   - candidate MAE on the old reference validation set is no more than **10% worse** than before (so it does not forget old behavior).
6. Log the decision (accept/reject) and the metrics. Replace the active model only inside the simulation.

### 5.9 `incident_logger.py`
Appends a record for every confirmed alert/decision: time, batch id, drift summary, performance status, action, LLM summary, whether the guardrail overrode the LLM. Saved as CSV/JSON so the dashboard and experiments can read it.

---

## 6. Configuration (`config.yaml`) — put every number here

```yaml
random_seed: 42
data:
  csv_path: data/E-Commerce.csv
  target: discount_pct       # fallback: delivery_days (see Section 4.5)
  reference_fraction: 0.5
  top_brands: 10
  baseline_train_fraction: 0.8

batch_size: 300
n_batches_per_run: 48
drift_start_batch: 16
gradual_ramp_batches: 8

drift_rule: both          # ks_only | psi_only | both
psi_bins: 10
psi_flag_threshold: 0.25
p_value_threshold: 0.01
persistence_required: 2   # flagged checks in a row = confirmed alert

adaptive:
  normal_interval: 4
  alert_interval: 1
  clean_checks_to_return: 3

performance:
  degradation_factor: 1.2
  consecutive_batches: 2

retraining:
  min_improvement: 0.05
  max_old_data_regression: 0.10

drift_sizes:
  small:  {price_factor: 1.15, category_share_drop: 0.10, rating_shift: -0.2}
  medium: {price_factor: 1.40, category_share_drop: 0.25, rating_shift: -0.5}
  large:  {price_factor: 2.00, category_share_drop: 0.45, rating_shift: -1.0}
```

---

## 7. Experiments (the research part)

General rules: fixed seeds, **30 repeated runs** per scenario (different seeds), report **mean ± standard deviation**, save raw results as CSV, save plots as PNG. Do not remove unfavorable results.

### Experiment A — KS vs PSI vs both
- **Goal:** compare detector rules.
- **Scenarios:** no drift; small/medium/large drift; sudden vs gradual; one changed feature vs several; batch sizes 150 / 300 / 600.
- **Metrics:** detection rate (share of runs that detect the drift), false-alarm rate on no-drift runs, detection delay (batches), computation time.
- **Expected learning:** KS tends to fire on tiny differences (especially with big batches); PSI measures size of change. We do not assume results — we measure.

### Experiment B — Fixed vs Adaptive (the main one)
- **Goal:** answer the main research question.
- **Strategies:** `Fixed(k=1)`, `Fixed(k=4)`, `Adaptive`, all on the **same** generated runs.
- **Metrics:**
  - Detection delay = (batch of first confirmed alert) − (drift start batch).
  - Missed-detection rate.
  - False alarms per 48-batch no-drift run.
  - Number of full checks.
  - Total detection computation time.
- **What a good conclusion looks like:** e.g., "Adaptive used X% fewer checks than Fixed(1) with Y extra batches of delay, and reacted Z batches faster than Fixed(4)." If adaptive is not better, say so and explain why.

### Experiment C — Feature ranking accuracy
- **Goal:** can the ranking find the columns we deliberately changed?
- **Scenarios:** change 1 feature; change 2 features; change 3 features.
- **Metrics:** **Precision@k** (k = number of truly changed features: share of the top-k ranked features that are truly changed); whether the changed feature is ranked #1 in single-feature cases.

### Experiment D — Decision quality
- **Goal:** check the agent makes sensible, evidence-based recommendations.
- **Four scenarios (expected action written **before** running):**

| Scenario | How to create it | Expected action |
|---|---|---|
| Drift, performance still fine | small shift on a feature the model barely uses (check the model's feature importances in a pilot run), so MAE stays near baseline | `INVESTIGATE` |
| Drift and performance degraded | large combined drift | `PROPOSE_RETRAIN` |
| Performance worse, no feature drift | `label_shift` only | `INVESTIGATE` |
| Drift, labels not yet available | hide labels | `REQUEST_LABELS` |

- **Metrics:** share of decisions matching the expected action; number of unnecessary retrain proposals; number of times the guardrail overrode the LLM; number of LLM replies that were invalid; whether the retrain acceptance rule correctly rejected a deliberately bad candidate (e.g., one trained on too little data).
- Run the LLM version and a **rules-only** version side by side (cheap and informative). Use a small number of runs (e.g., 10 per scenario) because LLM calls cost time/money.

### Output
`experiments/run_experiments.py` writes tables to `results/*.csv` and figures to `results/*.png`. The dashboard has a page that loads and displays them.

---

## 8. Dashboard (Streamlit + Plotly)

Keep it to four simple pages/tabs:

1. **Live Simulation:** sliders/dropdowns for drift type, size, shape, affected features, start batch, strategy (Fixed-1 / Fixed-4 / Adaptive). A "Run" button streams batches and shows: drift status per batch, monitoring mode over time, checks performed so far, MAE over time (or "labels unavailable").
2. **Investigation:** for a chosen batch — table of features with KS/PSI/flag, ranked bar chart, reference vs incoming histogram (numeric) or bar chart (categorical).
3. **Decisions & Incidents:** the incident log with the agent's summary, allowed actions, chosen action, and whether a guardrail changed it. Retraining decisions shown here.
4. **Experiment Results:** loads CSV/PNG from `results/` and shows the comparison tables and charts.

No login, no database. Everything runs locally.

---

## 9. Testing (pytest — small but meaningful)

- `test_drift_detector.py`: identical data → PSI close to 0 and KS p-value large; strongly shifted data → PSI large, flagged; empty-bin handling does not crash; ranking puts the changed feature first.
- `test_adaptive_monitor.py`: switches to ALERT after a flagged check; returns to NORMAL after 3 clean checks; confirmed alert needs 2 flagged checks in a row; Fixed(k) checks exactly every k-th batch.
- `test_decision_agent.py`: invalid LLM action is replaced by the default; invalid JSON falls back to the template; each table row in 5.7 gives the expected default action.
- `test_retrainer.py`: a clearly worse candidate is rejected; a clearly better one is accepted.
- Basic data test: cleaning drops the long-text columns, removes duplicates and non-positive prices, and parses dates; `discount_pct` always stays between 0 and 1.

---

## 10. Six-week plan (5 weeks of work + 1 buffer week)

**Week 1 — Data and baseline**
- Set up repo, virtual environment, Git/GitHub, folder structure, `config.yaml`.
- Download the CSV and run the data audit (Section 4.2). Confirm the columns, date range and target (Section 4.5). Only then write the cleaning rules and build features.
- Create reference half / clean pool split. Train baseline model, record baseline MAE.
- *Deliverable:* `data_audit.py` report, `data_loader.py`, `baseline_model.py`, and the baseline MAE next to the "predict the average" dummy MAE.

**Week 2 — Drift detection**
- Implement KS, PSI (numeric + categorical), chi-square, ranking, quality checks, batch simulator with drift injection.
- Write drift detector tests.
- *Deliverable:* running `detect_drift` on a simulated batch gives a correct, ranked report.

**Week 3 — Monitoring strategies and performance**
- Implement Fixed and Adaptive strategies, performance monitor, incident logger.
- Run a first small comparison to check everything works.
- *Deliverable:* a script that runs 48-batch simulations for all strategies and prints metrics.

**Week 4 — Agent and retraining**
- Rules table, LLM call, JSON validation, guardrails, template fallback.
- Retrainer and acceptance rule.
- *Deliverable:* an incident report and a decision for a simulated drift event, with and without an API key.

**Week 5 — Experiments and dashboard**
- Run Experiments A–D with 30 repeats, save results.
- Build the 4-tab Streamlit dashboard.
- *Deliverable:* results tables/plots and a working dashboard.

**Week 6 — Buffer, documentation, polish**
- Finish tests, README (how to install and run), limitations, short research report/paper draft, presentation.
- Optional: deploy to Streamlit Community Cloud.
- *Deliverable:* a clean GitHub repository and a report.

**If you fall behind, cut in this order:** (1) deployment, (2) Experiment D's LLM-vs-rules comparison (keep just the rules-based check), (3) Experiment A's batch-size variation, (4) natural time-based exploration. **Never cut:** KS+PSI detection, feature ranking, fixed-vs-adaptive experiment, and the dashboard.

---

## 11. Using this file with Antigravity

### 11.1 Setup
1. Create an empty project folder and open it in Antigravity.
2. Save this file in the folder root as `PROJECT_SPEC.md`.
3. Start every new task by telling the agent to read `PROJECT_SPEC.md` first.

### 11.2 Ground rules to paste at the start (important)
```
You are helping a second-year student build the project described in PROJECT_SPEC.md.
Rules:
1. Read PROJECT_SPEC.md fully before writing code.
2. Build ONLY the phase I ask for. Do not add features, libraries, or files that are not in the spec.
3. Keep code simple and readable: type hints, short functions, a docstring and comments explaining WHY, not just what.
4. Put every threshold and setting in config.yaml; no magic numbers in code.
5. Use fixed random seeds. Never use future data to compute anything for the past.
6. Implement KS, PSI and the adaptive logic ourselves with numpy/scipy; do not use drift libraries.
7. Write pytest tests for what you build and run them. Show me the results.
8. Never hard-code API keys. The system must work without an LLM key (template fallback).
9. After each phase, tell me in plain English what you built, how to run it, and what to check.
10. If the spec is unclear or something seems wrong, ask me instead of guessing.
```

### 11.3 Phase prompts (give one at a time; check each before moving on)

**Phase 1 — Setup & data**
> Read PROJECT_SPEC.md. Do Phase 1a only: create the folder structure from section 5.1, `requirements.txt`, `.gitignore`, `.env.example`, and `config.yaml` from section 6. Then write `src/data_audit.py` per section 4.2 and run it on `data/E-Commerce.csv`. STOP after printing the audit report. Do not write `data_loader.py` yet. The column names in the spec are unverified, so tell me exactly what you found and which spec assumptions are wrong.

**Phase 1b — Cleaning and features (only after I confirm the audit results)**
> Implement `data_loader.py` per sections 4.3 to 4.7, using the real column names from the audit. Add a short script that prints dataset size before and after cleaning and the summary of each feature and of the target. Add a data test.

**Phase 2 — Baseline model**
> Do `baseline_model.py` per section 4.5: train the regressor on the reference half, report MAE on a held-out part of the reference half and the MAE of a predict-the-average dummy model, save the model with joblib, and store the baseline MAE in a JSON file.

**Phase 3 — Drift detection**
> Do `quality_checks.py` and `drift_detector.py` per sections 5.3 and 5.4, plus `test_drift_detector.py`. Do not use any drift library.

**Phase 4 — Batch simulator**
> Do `batch_simulator.py` per section 5.2 including ground truth output. Add a script that generates one 48-batch run with a sudden medium drift on `retail_price` starting at batch 16 and prints the detector's report for batches 15–18.

**Phase 5 — Monitoring strategies & performance**
> Do `adaptive_monitor.py`, `performance_monitor.py`, `incident_logger.py` per sections 5.5, 5.6, 5.9, plus `test_adaptive_monitor.py`. Add a script that runs Fixed(1), Fixed(4) and Adaptive on the same run and prints delay, number of checks, time.

**Phase 6 — Agent**
> Do `decision_agent.py` per section 5.7 with the rules table, a single `call_llm` function (provider configurable by env variable), JSON validation, guardrails and the template fallback. Add `test_decision_agent.py`. It must run end-to-end with no API key.

**Phase 7 — Retraining**
> Do `retrainer.py` per section 5.8 plus tests.

**Phase 8 — Experiments**
> Do `experiments/run_experiments.py` per section 7: Experiments A–D, 30 repeated seeds, results saved to `results/` as CSV and PNG. Make it possible to run each experiment separately and with a `--quick` flag (fewer repeats) for testing.

**Phase 9 — Dashboard**
> Do `app.py` per section 8 with four tabs, using Plotly. It should use the existing `src` modules and results files; do not duplicate logic.

**Phase 10 — Documentation**
> Write a clear `README.md` (what it is, install steps, how to run each part, how to run tests, results summary, limitations). Re-run all tests and fix failures.

### 11.4 How to check Antigravity's work (do not skip)
- Run the tests yourself.
- For each module, be able to explain it in your own words. If you cannot, ask the agent to explain it and simplify it.
- Check that numbers in the report come from `results/` files, not from the agent's guesses.
- Commit to Git after every phase with a meaningful message.

---

## 12. Limitations, honesty rules, glossary

### 12.1 Limitations (include in your report)
- One small dataset (under 9,000 rows), a course dataset with unverified provenance and no stated licence; results may not generalize to other data or real production systems.
- Batches are drawn from a small pool, so the same row can appear in several batches and runs are not fully independent.
- Most drift is **injected by us**, so the experiments test the system against known changes, not real-world surprises.
- Detector results depend on thresholds, bin counts and batch size.
- Adaptive monitoring can delay detection if the policy is poorly set; our policy is simple and hand-designed.
- The prediction task is deliberately simple; the baseline model is not meant to be strong.
- LLM output can be incomplete or wrong; that is why guardrails exist, and why it has no authority over numbers.
- Natural time-based findings are exploratory because we have no ground-truth drift labels.

### 12.2 Honesty rules
- Never write results before the experiments are run.
- Do not claim the methods are new. The contribution is a careful, reproducible comparison of fixed vs adaptive monitoring in a complete small pipeline.
- Do not say "drift caused performance drop" unless MAE evidence shows it.
- Report negative or mixed results.

### 12.3 Glossary
- **Batch:** a chunk of new data processed together.
- **Reference data:** the "normal" data used as the comparison standard.
- **Drift / data drift:** input data distribution changes.
- **Concept drift:** relationship between inputs and target changes.
- **KS statistic / p-value:** size of the biggest distribution gap / evidence it is not due to chance.
- **PSI:** score measuring how much a distribution's bin shares have changed.
- **Flagged check:** a check that found drift.
- **Confirmed alert:** two flagged checks in a row.
- **Detection delay:** batches between drift start and confirmed alert.
- **False alarm:** confirmed alert when no drift exists.
- **Overhead:** number and time of full checks.
- **Guardrail:** deterministic rule that limits what the LLM can do.
- **Leakage:** accidentally using future information when training or testing.
- **Precision@k:** share of top-k ranked features that are truly changed.
- **MAE:** average absolute prediction error.

---

## 13. Decisions still open (decide early, write them in the README)
1. Which LLM provider/API key you will use (the system works without one, but Experiment D's LLM comparison needs one).
2. Whether to attempt the optional Streamlit Community Cloud deployment.
3. The final target (`discount_pct` or the fallback `delivery_days`) and the final feature list, after the audit and the dummy-model check.
4. Final batch size and threshold values — adjust only after Experiment A shows how the detectors behave, and record any change.
