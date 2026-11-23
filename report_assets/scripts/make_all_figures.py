import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config
from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.drift_detector import detect_drift
from src.adaptive_monitor import compare_strategies

# Configure matplotlib global style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 9.5
plt.rcParams["axes.labelsize"] = 9.5
plt.rcParams["xtick.labelsize"] = 8.5
plt.rcParams["ytick.labelsize"] = 8.5
plt.rcParams["legend.fontsize"] = 8.5
plt.rcParams["figure.dpi"] = 300

FIGURES_DIR = os.path.join(PROJECT_ROOT, "report_assets", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Dashboard palette
COLOR_REF = "#7f7f7f"      # neutral grey
COLOR_ACCENT = "#1f77b4"   # blue
COLOR_ALERT = "#d62728"    # red
COLOR_DARK = "#222222"

cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
drift_start = int(cfg.get("drift_start_batch", 16))
ramp_len = int(cfg.get("gradual_ramp_batches", 8))
n_batches = int(cfg.get("n_batches_per_run", 48))
psi_thresh = float(cfg.get("psi_flag_threshold", 0.25))


# ==============================================================================
# Figure 1.1: Drift Patterns Illustration
# ==============================================================================
def make_fig_1_1():
    np.random.seed(42)
    batches = np.arange(n_batches)
    baseline_mean = 1000.0
    sigma = 30.0

    # Pattern 1: No drift
    mean_no_drift = baseline_mean + np.random.normal(0, sigma, size=n_batches)

    # Pattern 2: Sudden drift (+40% at batch 16)
    mean_sudden = baseline_mean + np.random.normal(0, sigma, size=n_batches)
    mean_sudden[drift_start:] += 400.0

    # Pattern 3: Gradual drift (linear ramp over 8 batches to +40%)
    mean_gradual = baseline_mean + np.random.normal(0, sigma, size=n_batches)
    for b in range(drift_start, drift_start + ramp_len):
        frac = (b - drift_start + 1) / ramp_len
        mean_gradual[b] += frac * 400.0
    mean_gradual[drift_start + ramp_len:] += 400.0

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 4.3), sharex=True, sharey=True)

    patterns = [
        (mean_no_drift, "No drift (stationary)", "solid", "o"),
        (mean_sudden, "Sudden drift (+40% jump)", "dashed", "s"),
        (mean_gradual, f"Gradual drift ({ramp_len}-batch ramp)", "dotted", "^"),
    ]

    for idx, (data, label, ls, mk) in enumerate(patterns):
        ax = axes[idx]
        ax.plot(batches, data, color=COLOR_ACCENT, linestyle=ls, marker=mk, markersize=3,
                markevery=2, linewidth=1.2, label=label)
        ax.axvline(drift_start, color=COLOR_ALERT, linestyle="--", linewidth=1.0)
        ax.axhline(baseline_mean, color=COLOR_REF, linestyle=":", linewidth=0.8)
        ax.text(drift_start + 0.5, 1420, "Drift start (B16)", color=COLOR_ALERT, fontsize=8)
        ax.set_ylabel("Batch mean (arbitrary units)")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_ylim(850, 1550)

    axes[-1].set_xlabel("Batch index")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_1_1_drift_patterns.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_1_1_drift_patterns.png")


# ==============================================================================
# Figure 2.1: System Architecture Diagram
# ==============================================================================
def make_fig_2_1():
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axis("off")

    def draw_box(x, y, w, h, text, bg="#f7f7f7", ec="#333333", ls="solid", lw=1.0):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=bg, edgecolor=ec, linestyle=ls, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.0, color="#111111", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label,
                    ha="center", va="bottom", fontsize=7, color="#555555")

    # Column 1: Ingestion & Split
    draw_box(0.02, 0.76, 0.26, 0.16, "Raw E-Commerce Orders\n(CSV: 8,906 rows, 17 cols)")
    draw_box(0.02, 0.48, 0.26, 0.18, "Data Cleaning &\nFeature Construction\n(Drop invalid, leak-free)")
    draw_box(0.02, 0.18, 0.26, 0.20, "50 / 50 Split (Seed 42)\nRef: 4,453 | Pool: 4,453\n(Train 80% / Val 20%)")

    draw_arrow(0.15, 0.76, 0.15, 0.66)
    draw_arrow(0.15, 0.48, 0.15, 0.38)

    # Column 2: Streaming & Detection
    draw_box(0.36, 0.76, 0.28, 0.16, "Batch Simulator\n(300 rows / batch, 48 batches)\n+ Drift Injection Engine")
    draw_box(0.36, 0.48, 0.28, 0.18, "Batch Quality Checks\n(Schema, missing, unseen cat)\n& Drift Detection (KS, PSI, $\\chi^2$)")
    draw_box(0.36, 0.18, 0.28, 0.20, "Monitoring Strategy\nFixed(1), Fixed(4), Adaptive\n+ Persistence Rule (2 checks)")

    draw_arrow(0.28, 0.28, 0.36, 0.84, label="clean pool")
    draw_arrow(0.50, 0.76, 0.50, 0.66)
    draw_arrow(0.50, 0.48, 0.50, 0.38)

    # Column 3: Decision & Dashboard
    draw_box(0.72, 0.76, 0.26, 0.16, "Performance Monitor\nMAE & degradation check\nBaseline: Random Forest")
    draw_box(0.72, 0.48, 0.26, 0.18, "Decision Agent\nRules table + LLM advisor\n+ Guardrail fallbacks")
    draw_box(0.72, 0.18, 0.26, 0.20, "Outputs & Dashboard\nIncident log, Retrainer,\nExperiments A-D, Streamlit UI")

    draw_arrow(0.28, 0.24, 0.72, 0.84, label="ref model")
    draw_arrow(0.64, 0.28, 0.72, 0.57)
    draw_arrow(0.85, 0.76, 0.85, 0.66)
    draw_arrow(0.85, 0.48, 0.85, 0.38)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.1, 1.0)

    out_path = os.path.join(FIGURES_DIR, "fig_2_1_architecture.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_1_architecture.png")


# ==============================================================================
# Figure 2.2: Data Split Diagram
# ==============================================================================
def make_fig_2_2():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axis("off")

    def draw_box(x, y, w, h, text, bg="#f7f7f7", ec="#333333", lw=1.0):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=bg, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.5, color="#111111", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0)
        )
        if label:
            ax.text((x1 + x2) / 2 + 0.02, (y1 + y2) / 2, label,
                    ha="left", va="center", fontsize=7.5, color="#444444")

    # Cleaned dataset
    draw_box(0.20, 0.78, 0.42, 0.16, "Cleaned Dataset\n8,906 rows (100%)\n(Strict leak-free cleaning)", bg="#e8f1f5", ec=COLOR_ACCENT)

    # 50/50 split arrows
    draw_arrow(0.32, 0.78, 0.18, 0.62, "50% split")
    draw_arrow(0.50, 0.78, 0.64, 0.62, "50% split")

    # Reference Half and Clean Pool
    draw_box(0.05, 0.44, 0.32, 0.18, "Reference Half\n4,453 rows (50%)\n(Fixed baseline context)", bg="#efefef", ec=COLOR_REF)
    draw_box(0.48, 0.44, 0.32, 0.18, "Clean Pool\n4,453 rows (50%)\n(Streaming simulation source)", bg="#e8f1f5", ec=COLOR_ACCENT)

    # Split reference into Train / Val
    draw_arrow(0.15, 0.44, 0.10, 0.28, "80%")
    draw_arrow(0.27, 0.44, 0.28, 0.28, "20%")

    draw_box(0.02, 0.12, 0.18, 0.16, "Train Split\n3,562 rows (40%)")
    draw_box(0.22, 0.12, 0.18, 0.16, "Held-out Val\n891 rows (10%)")

    # Clean Pool into Batches
    draw_arrow(0.64, 0.44, 0.64, 0.28)
    draw_box(0.48, 0.12, 0.32, 0.16, "Streaming Batches\n300 rows / batch\n(48 batches per run)")

    # Side callout box
    callout_text = (
        "Computed from reference half only:\n"
        "• PSI Bin Edges (10 deciles)\n"
        "• Retail Price Cap (99.5th = ₹20,576)\n"
        "• Top 10 Brand Groups\n"
        "• Product Rating Median (3.0)"
    )
    rect = patches.FancyBboxPatch(
        (0.83, 0.18), 0.16, 0.70,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#fffde7", edgecolor="#fbc02d", linewidth=1.0
    )
    ax.add_patch(rect)
    ax.text(0.91, 0.53, callout_text, ha="center", va="center",
            fontsize=7.2, color="#333333", multialignment="left")

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.05, 1.0)

    out_path = os.path.join(FIGURES_DIR, "fig_2_2_split.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_2_split.png")


# ==============================================================================
# Figure 2.3: Drift Injection Effect (Clean vs Medium Sudden)
# ==============================================================================
def make_fig_2_3():
    ref_df, pool_df, _ = load_experiment_data(cfg)
    seed = 42

    sim = simulate_run(
        pool_df=pool_df,
        config=cfg,
        run_id=seed,
        drift_type="shift_mean",
        drift_size="medium",
        drift_shape="sudden",
        random_seed=seed,
    )

    clean_batch = sim.batches[0]    # Batch 0 (clean)
    drift_batch = sim.batches[16]   # Batch 16 (first drift batch)

    rep_clean = detect_drift(ref_df, clean_batch, config=cfg, batch_id=0)
    rep_drift = detect_drift(ref_df, drift_batch, config=cfg, batch_id=16)

    psi_clean = rep_clean["features"]["retail_price"]["psi"]
    psi_drift = rep_drift["features"]["retail_price"]["psi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.2), sharey=True)

    # Clip price at 99th percentile for visual clarity
    p99 = np.percentile(ref_df["retail_price"], 99)
    print(f"Figure 2.3 Recomputation:")
    print(f"  Seed: {seed}")
    print(f"  Clean Batch index: 0, Drifted Batch index: 16")
    print(f"  Feature: retail_price")
    print(f"  Clean-batch PSI: {psi_clean:.6f} (display: {psi_clean:.4f})")
    print(f"  Drifted-batch PSI: {psi_drift:.6f} (display: {psi_drift:.4f})")
    print(f"  X-axis clip: 99th percentile = {p99:.2f} INR")
    bins = np.linspace(0, p99, 25)

    # Panel (a): Clean batch vs reference
    ax1.hist(ref_df["retail_price"].clip(upper=p99), bins=bins, density=True,
             color=COLOR_REF, alpha=0.5, label="Reference")
    ax1.hist(clean_batch["retail_price"].clip(upper=p99), bins=bins, density=True,
             histtype="step", color=COLOR_ACCENT, linewidth=1.5, label="Clean Batch 0")
    ax1.set_xlabel("Retail price (₹)")
    ax1.set_ylabel("Probability density")
    ax1.text(0.95, 0.85, f"(a) Clean batch\nPSI = {psi_clean:.4f}", transform=ax1.transAxes,
             ha="right", va="top", bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor=COLOR_REF, alpha=0.8))
    ax1.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.95, 0.65))
    ax1.grid(True, linestyle=":", alpha=0.4)

    # Panel (b): Drifting batch vs reference
    ax2.hist(ref_df["retail_price"].clip(upper=p99), bins=bins, density=True,
             color=COLOR_REF, alpha=0.5, label="Reference")
    ax2.hist(drift_batch["retail_price"].clip(upper=p99), bins=bins, density=True,
             histtype="step", color=COLOR_ALERT, linewidth=1.5, linestyle="--", label="Drift Batch 16 (+40%)")
    ax2.set_xlabel("Retail price (₹)")
    ax2.text(0.95, 0.85, f"(b) Medium sudden drift\nPSI = {psi_drift:.4f} (Flagged)", transform=ax2.transAxes,
             ha="right", va="top", bbox=dict(boxstyle="round", facecolor="#ffffff", edgecolor=COLOR_ALERT, alpha=0.8))
    ax2.legend(loc="upper right", frameon=False, bbox_to_anchor=(0.95, 0.65))
    ax2.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_2_3_injection_effect.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_3_injection_effect.png")


# ==============================================================================
# Figure 2.4: Adaptive State Machine Diagram
# ==============================================================================
def make_fig_2_4():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axis("off")

    norm_interval = cfg.get("adaptive", {}).get("normal_interval", 4)
    alert_interval = cfg.get("adaptive", {}).get("alert_interval", 1)
    clean_req = cfg.get("adaptive", {}).get("clean_checks_to_return", 3)
    persist_req = cfg.get("persistence_required", 2)

    # NORMAL State box
    rect1 = patches.FancyBboxPatch(
        (0.08, 0.35), 0.32, 0.40,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#f0f4f8", edgecolor=COLOR_ACCENT, linewidth=1.5
    )
    ax.add_patch(rect1)
    ax.text(0.24, 0.55, f"NORMAL MODE\n\nFull check every\n{norm_interval}th batch (k={norm_interval})\nCompute saver",
            ha="center", va="center", fontsize=9.0, color="#111111")

    # ALERT State box
    rect2 = patches.FancyBboxPatch(
        (0.60, 0.35), 0.32, 0.40,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor="#fdf2f2", edgecolor=COLOR_ALERT, linewidth=1.5
    )
    ax.add_patch(rect2)
    ax.text(0.76, 0.55, f"ALERT MODE\n\nFull check every\nbatch (k={alert_interval})\nHigh vigilance",
            ha="center", va="center", fontsize=9.0, color="#111111")

    # Transition NORMAL -> ALERT (top curve)
    ax.annotate(
        "", xy=(0.60, 0.65), xytext=(0.40, 0.65),
        arrowprops=dict(arrowstyle="->", color=COLOR_ALERT, lw=1.5,
                        connectionstyle="arc3,rad=-0.25")
    )
    ax.text(0.50, 0.82, f"Flagged check (PSI ≥ {psi_thresh:.2f} or p < 0.01)\nEscalate immediately",
            ha="center", va="center", fontsize=7.8, color=COLOR_ALERT)

    # Transition ALERT -> NORMAL (bottom curve)
    ax.annotate(
        "", xy=(0.40, 0.45), xytext=(0.60, 0.45),
        arrowprops=dict(arrowstyle="->", color=COLOR_ACCENT, lw=1.5,
                        connectionstyle="arc3,rad=-0.25")
    )
    ax.text(0.50, 0.28, f"{clean_req} consecutive clean checks\nDe-escalate",
            ha="center", va="center", fontsize=7.8, color=COLOR_ACCENT)

    # Persistence rule note box at the bottom
    rule_text = (
        f"Persistence Rule: Confirmed alert requires {persist_req} flagged checks in a row.\n"
        "A single clean check resets the consecutive flagged counter to 0."
    )
    rect3 = patches.FancyBboxPatch(
        (0.12, 0.05), 0.76, 0.14,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#fffde7", edgecolor="#fbc02d", linewidth=1.0
    )
    ax.add_patch(rect3)
    ax.text(0.50, 0.12, rule_text, ha="center", va="center",
            fontsize=8.0, color="#333333")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    out_path = os.path.join(FIGURES_DIR, "fig_2_4_adaptive_states.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_4_adaptive_states.png")


# ==============================================================================
# Figure 2.5: Decision Layer Flowchart
# ==============================================================================
def make_fig_2_5():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.axis("off")

    def draw_box(x, y, w, h, text, bg="#f7f7f7", ec="#333333", lw=1.0):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=bg, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.0, color="#111111", multialignment="center")

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03, label,
                    ha="center", va="bottom", fontsize=7.2, color="#444444")

    # Inputs on the left
    draw_box(0.04, 0.74, 0.22, 0.18, "Drift Result\n(Confirmed alert,\ntop features, PSI)")
    draw_box(0.04, 0.44, 0.22, 0.18, "Quality Result\n(Missing values,\nunseen categories)")
    draw_box(0.04, 0.14, 0.22, 0.18, "Performance Status\n(Degradation factor,\nlabels available)")

    # Rules Table in the center
    draw_box(0.34, 0.36, 0.24, 0.34, "Deterministic Rules Table\n(Lookup matrix per spec)\n\n• Allowed actions\n• Default action",
             bg="#eef5f9", ec=COLOR_ACCENT, lw=1.2)

    draw_arrow(0.26, 0.83, 0.34, 0.60)
    draw_arrow(0.26, 0.53, 0.34, 0.53)
    draw_arrow(0.26, 0.23, 0.34, 0.46)

    # LLM Advisor & Guardrails
    draw_box(0.66, 0.62, 0.28, 0.24, "Optional LLM Advisor\n(Prompt with incident context)\nSuggests action & reason",
             bg="#fdfefe", ec="#7f7f7f", lw=1.0)

    draw_box(0.66, 0.26, 0.28, 0.24, "Guardrail Verification\n1. Valid JSON schema\n2. Chosen action $\\in$ allowed\n(Fallback to default if invalid)",
             bg="#fff8e1", ec="#ffa000", lw=1.2)

    draw_arrow(0.58, 0.60, 0.66, 0.74, label="incident state")
    draw_arrow(0.80, 0.62, 0.80, 0.50, label="suggested")
    draw_arrow(0.58, 0.45, 0.66, 0.38, label="allowed actions")

    # Incident Log Output
    draw_box(0.66, 0.04, 0.28, 0.14, "Incident Log & Operator Alert\nAction executed + audit trail",
             bg="#f1f8e9", ec="#4caf50", lw=1.2)

    draw_arrow(0.80, 0.26, 0.80, 0.18, label="confirmed action")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    out_path = os.path.join(FIGURES_DIR, "fig_2_5_decision_flow.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_5_decision_flow.png")


# ==============================================================================
# Figure 2.6: Dashboard Layout Schematic
# ==============================================================================
def make_fig_2_6():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axis("off")

    def draw_card(x, y, w, h, title, items, bg="#f9f9f9", ec="#cccccc"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=bg, edgecolor=ec, linewidth=1.0
        )
        ax.add_patch(rect)
        ax.text(x + 0.02, y + h - 0.05, title, fontsize=8.2, fontweight="bold", color="#111111")
        item_y = y + h - 0.11
        for item in items:
            ax.text(x + 0.03, item_y, f"• {item}", fontsize=7.2, color="#333333")
            item_y -= 0.06

    # Top title bar
    title_rect = patches.Rectangle((0.02, 0.88), 0.96, 0.10, facecolor="#1f77b4", edgecolor="none")
    ax.add_patch(title_rect)
    ax.text(0.50, 0.93, "DriftGuard Dashboard (Streamlit Application Layout)",
            ha="center", va="center", color="#ffffff", fontsize=9.5, fontweight="bold")

    # Tab 1
    draw_card(0.02, 0.46, 0.46, 0.40, "Tab 1: Live simulation", [
        "Controls: Select simulation scenario preset, Run simulation",
        "Summary strip: Fixed(1), Fixed(4), Adaptive metrics",
        "Timeline chart: Checks performed & alert batches",
        "Feature PSI timeline & Model MAE timeline"
    ], bg="#f0f7fb", ec="#90caf9")

    # Tab 2
    draw_card(0.52, 0.46, 0.46, 0.40, "Tab 2: Investigation", [
        "Controls: Select batch to investigate slider",
        "Feature drift metrics table (statistic, p-value, PSI)",
        "Ranked bar chart: Feature PSI comparison",
        "Distribution comparison: Reference vs Batch histograms"
    ], bg="#f7fbf0", ec="#a5d6a7")

    # Tab 3
    draw_card(0.02, 0.03, 0.46, 0.40, "Tab 3: Decisions and incidents", [
        "Incident summary label & Incident log table",
        "Allowed actions & Default action (rules matrix)",
        "LLM recommendation card & Guardrail status check",
        "Export buttons: Download CSV / JSON incident log"
    ], bg="#fffbf0", ec="#ffe082")

    # Tab 4
    draw_card(0.52, 0.03, 0.46, 0.40, "Tab 4: Experiment results", [
        "Controls: Select experiment (A, B, C, D)",
        "Run manifest metadata card (commit, seeds, timestamp)",
        "Summary metrics table across scenarios and strategies",
        "Experiment result visualization plots"
    ], bg="#fcf0fb", ec="#ce93d8")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    out_path = os.path.join(FIGURES_DIR, "fig_2_6_dashboard_layout.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_2_6_dashboard_layout.png")


# ==============================================================================
# Figure 3.3: Seed 44 Trace (Medium Sudden Price Drift)
# ==============================================================================
def make_fig_3_3():
    ref_df, pool_df, _ = load_experiment_data(cfg)
    seed = 44

    sim = simulate_run(
        pool_df=pool_df,
        config=cfg,
        run_id=seed,
        drift_type="shift_mean",
        drift_size="medium",
        drift_shape="sudden",
        random_seed=seed,
    )

    results = compare_strategies(ref_df, sim, config=cfg)

    # Check alert batches match requirements
    f1_alert = results["Fixed_1"].first_confirmed_alert_batch
    f4_alert = results["Fixed_4"].first_confirmed_alert_batch
    ad_alert = results["Adaptive"].first_confirmed_alert_batch
    assert (f1_alert, f4_alert, ad_alert) == (23, 20, 28), f"Alert batch mismatch: {(f1_alert, f4_alert, ad_alert)}"

    # Batch range 12 to 30
    batches = list(range(12, 31))

    # Top-feature PSI per batch (from Fixed_1 which checks every batch)
    f1_timeline = {rec.batch_id: rec for rec in results["Fixed_1"].checks_timeline}
    psi_values = [f1_timeline[b].top_feature_psi for b in batches]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.3), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})

    # Panel (a): Top-feature PSI
    ax1.plot(batches, psi_values, color=COLOR_ACCENT, linestyle="-", marker="o", markersize=4, label="retail_price PSI")
    ax1.axhline(psi_thresh, color=COLOR_ALERT, linestyle="--", linewidth=1.0, label=f"PSI threshold ({psi_thresh})")
    ax1.axvline(drift_start, color=COLOR_DARK, linestyle=":", linewidth=1.0, label=f"Drift start (B{drift_start})")

    # Mark flagged batches on Panel (a)
    for b, psi in zip(batches, psi_values):
        if psi >= psi_thresh:
            ax1.plot(b, psi, marker="s", markersize=5, color=COLOR_ALERT)

    ax1.set_ylabel("Top-feature PSI")
    ax1.set_ylim(-0.02, max(psi_values) * 1.15)
    ax1.legend(loc="upper left", frameon=False)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.text(0.98, 0.88, "(a) Top-feature PSI per batch", transform=ax1.transAxes, ha="right", va="top", fontsize=8.5)

    # Panel (b): Strategy checks and alerts
    strategies = [("Fixed_1", 2, results["Fixed_1"]),
                  ("Fixed_4", 1, results["Fixed_4"]),
                  ("Adaptive", 0, results["Adaptive"])]

    for name, y_pos, mon_res in strategies:
        tl = {rec.batch_id: rec for rec in mon_res.checks_timeline}
        for b in batches:
            rec = tl.get(b)
            if rec and rec.check_executed:
                if rec.drift_detected:
                    # Flagged check
                    ax2.plot(b, y_pos, marker="s", markersize=5, color=COLOR_ALERT, fillstyle="full")
                else:
                    # Clean check
                    ax2.plot(b, y_pos, marker="o", markersize=4, color=COLOR_REF, fillstyle="full")
            else:
                # Skipped check
                ax2.plot(b, y_pos, marker="x", markersize=3, color="#cccccc")

        # Mark confirmed alert
        alert_b = mon_res.first_confirmed_alert_batch
        if alert_b and alert_b in batches:
            ax2.plot(alert_b, y_pos, marker="*", markersize=10, color=COLOR_ALERT, markeredgecolor="black", zorder=5)
            ax2.text(alert_b + 0.3, y_pos + 0.15, f"Alert (B{alert_b})", color=COLOR_ALERT, fontsize=7.5, fontweight="bold")

    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Adaptive", "Fixed(4)", "Fixed(1)"])
    ax2.set_xlabel("Batch index")
    ax2.set_xlim(11.5, 30.5)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.text(0.98, 0.88, "(b) Strategy check & alert events", transform=ax2.transAxes, ha="right", va="top", fontsize=8.5)

    # Custom legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_REF, markersize=5, label="Clean check"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_ALERT, markersize=5, label="Flagged check"),
        Line2D([0], [0], marker="x", color="#cccccc", markersize=5, label="Skipped check"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLOR_ALERT, markeredgecolor="black", markersize=8, label="Confirmed alert"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", frameon=False, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_3_3_seed44_trace.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print("Generated fig_3_3_seed44_trace.png")


if __name__ == "__main__":
    make_fig_1_1()
    make_fig_2_1()
    make_fig_2_2()
    make_fig_2_3()
    make_fig_2_4()
    make_fig_2_5()
    make_fig_2_6()
    make_fig_3_3()

