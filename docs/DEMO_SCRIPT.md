# DriftGuard: 3-Minute Interview Demonstration Script

This script provides an exact step-by-step walkthrough to demonstrate DriftGuard during an interview or portfolio presentation using the four pre-configured dashboard presets.

**Total Duration:** 3 minutes  
**Application Command:** `streamlit run app.py`  
**Starting View:** Tab 1 — *Live simulation*

---

## 00:00 – 00:45 | Preset 1: Large Sudden Price Drift (The Happy Path)

### What to Click
1. On Tab 1 (*Live simulation*), select **"Large sudden price drift"** from the *Preset scenario* dropdown.
2. Click the primary **"Run simulation"** button (completes in $\approx 7$ seconds).

### What to Say
> "I built DriftGuard to explore an engineering trade-off in ML model monitoring: can we use an adaptive sampling schedule to detect drift almost as fast as checking every single batch, while running far fewer compute-intensive statistical checks?
>
> In this first scenario, we inject severe sudden drift on `retail_price` (doubling prices) starting at batch 16. In the summary strip, notice that Adaptive monitoring and Fixed(1) both raise a confirmed alert at batch 17—achieving an identical detection delay of 1.0 batch. However, Adaptive executed only 36 full statistical checks compared to 48 for Fixed(1), saving 25% of compute checks without sacrificing detection speed."

### What to Point At on Screen
- **Summary Strip:** Point out `Detection delay: 1` for both Fixed(1) and Adaptive, and `Checks saved vs Fixed(1): 25.0%`.
- **Timeline Chart:** Point to the red stars at batch 17 where both monitors trigger confirmed alerts immediately after the dashed vertical drift-start line.
- **Top Feature PSI Chart:** Point to the sharp spike in `retail_price` PSI far exceeding the red dashed threshold line at 0.25.

---

## 00:45 – 01:30 | Preset 2: Medium Sudden Price Drift (The Nuanced Trade-off)

### What to Click
1. In the *Preset scenario* dropdown, select **"Medium sudden price drift"**.
2. Click **"Run simulation"**.

### What to Say
> "Next, we examine medium sudden price drift (a 40% increase on `retail_price`). Under this more realistic magnitude, we see the trade-off emerge clearly.
>
> Fixed(1) confirms drift with a delay of 2 batches, while Adaptive confirms it at batch 20—a delay of 4 batches. In return for those two batches of delay, Adaptive saved 33% of full checks. Notice that Fixed(4), which blindly checks every 4th batch, suffers a delay of 8 batches. Adaptive's state machine successfully collapsed its checking interval from every 4 batches to every batch as soon as the first perturbation was flagged."

### What to Point At on Screen
- **Summary Strip:** Contrast detection delays: Fixed(1) at 2 vs Adaptive at 4 vs Fixed(4) at 8. Point out `33% checks saved`.
- **Timeline Chart:** Show how Adaptive ran in normal mode ($k=4$) until batch 16, flagged the change, collapsed its interval to $k=1$, and confirmed drift at batch 20.
- **Model Error (MAE) Chart:** Point to the blue batch MAE rising above the red dashed degradation limit ($1.2\times$ baseline).

---

## 01:30 – 02:15 | Preset 3: Small Price Drift (The Honest Limitation)

### What to Click
1. In the *Preset scenario* dropdown, select **"Small price drift"**.
2. Click **"Run simulation"**.

### What to Say
> "Here is an honest limitation of hard thresholding in monitoring systems: when drift is subtle (a 15% price increase), the Population Stability Index hovers between 0.05 and 0.15, never crossing the conservative 0.25 threshold.
>
> As a result, our default combined rule—which requires both a statistically significant KS test and PSI $\ge 0.25$—misses this drift in 90% of runs under Adaptive monitoring. In Experiment A, we proved that the Kolmogorov-Smirnov test alone detects this drift in 30 out of 30 runs with a 1.1-batch delay, but KS alone suffers from false alarms on uncorrupted batches. The conservative 0.25 PSI threshold that eliminated false alarms under clean data makes the system insensitive to small distributional shifts."

### What to Point At on Screen
- **Summary Strip:** Show `First confirmed alert: None` and `Detection delay: N/A` for all monitors.
- **Top Feature PSI Chart:** Point to the PSI curve hovering safely below the red dashed threshold at 0.25, showing why neither Fixed nor Adaptive confirmed an alert.

---

## 02:15 – 03:00 | Preset 4: Concept Shift & Decision Actions

### What to Click
1. Select **"Concept shift"** from the *Preset scenario* dropdown and click **"Run simulation"**.
2. Switch tabs: click **Tab 2 (Investigation)**, then click **Tab 3 (Decisions and incidents)**.

### What to Say
> "Finally, Preset 4 simulates pure concept shift: input feature distributions remain identical, but the relationship between features and the target changes, multiplying discount percentages by 1.8.
>
> On Tab 1, feature PSI remains near zero, but the model MAE chart shows severe sustained degradation.
>
> In Tab 3, our deterministic decision agent evaluates the incident. Because performance degraded while feature distributions remained clean, the rules engine routes the incident to `INVESTIGATE` rather than blindly proposing model retraining. The incident log shows the exact reasoning, allows single-click Markdown or JSON export, and verifies candidate model retraining against dual-window validation to prevent catastrophic forgetting."

### What to Point At on Screen
- **Tab 1 MAE Chart:** Point to the model error surging while feature PSI remains flat.
- **Tab 3 Incident Log:** Point to the incident table row showing `chosen_action: INVESTIGATE`, `performance_status: degraded`, and `drift_detected: False`.
- **Export Buttons:** Highlight the three export options (`Download report (Markdown)`, `Download log (CSV)`, `Download log (JSON)`).
