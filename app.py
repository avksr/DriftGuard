"""
app.py – DriftGuard Streamlit Dashboard

Four tabs (Section 8 of PROJECT_SPEC.md):
1. Live Simulation  – configure & stream a synthetic run batch-by-batch
2. Investigation    – drill into one batch's KS/PSI feature stats + charts
3. Decisions & Incidents – browse the incident log + per-batch LLM summary
4. Experiment Results   – load CSV tables and PNG plots from results/

All heavy logic lives in the src package; the dashboard only orchestrates
and visualises. If a module is missing or a file is absent, a friendly
info message is shown instead of a traceback.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV safely, return None on failure."""
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read `{path.name}`: {exc}")
        return None


def _load_config() -> Dict[str, Any]:
    import yaml
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open("r") as f:
        return yaml.safe_load(f) or {}


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1 – Live Simulation
# ──────────────────────────────────────────────────────────────────────────────

def _tab_live_simulation() -> None:
    st.header("🔴 Live Simulation")
    st.markdown(
        "Configure a synthetic run and watch the monitoring system react "
        "batch-by-batch. Results are computed in real-time using the `src` modules."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        drift_type = st.selectbox(
            "Drift type",
            ["none", "shift_mean", "shift_additive", "change_spread",
             "shift_category_mix", "label_shift"],
            index=1,
        )
        drift_size = st.selectbox("Drift size", ["small", "medium", "large"], index=1)
    with col2:
        drift_shape = st.selectbox("Drift shape", ["sudden", "gradual"], index=0)
        strategy = st.selectbox(
            "Monitoring strategy", ["Fixed(k=1)", "Fixed(k=4)", "Adaptive"], index=2
        )
    with col3:
        drift_start = st.slider("Drift start batch", 1, 44, 16)
        n_batches = st.slider("Number of batches", 12, 48, 48, step=4)

    if not st.button("▶  Run simulation", type="primary"):
        st.info("Configure the parameters above and click **Run simulation** to start.")
        return

    # ── Load data & model ────────────────────────────────────────────────────
    try:
        from src.data_loader import prepare_datasets
        from src.batch_simulator import simulate_run
        from src.drift_detector import detect_drift
        from src.quality_checks import check_batch_quality
        from src.adaptive_monitor import FixedMonitor, AdaptiveMonitor
        from src.performance_monitor import PerformanceMonitor
    except ImportError as exc:
        st.error(f"Could not import required modules: {exc}")
        return

    cfg = _load_config()
    cfg["drift_start_batch"] = drift_start
    cfg["n_batches_per_run"] = n_batches

    with st.spinner("Loading data and baseline model …"):
        try:
            ref_df, pool_df, _ = prepare_datasets(cfg)
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            return

    # ── Run simulation ───────────────────────────────────────────────────────
    with st.spinner("Simulating batches …"):
        try:
            sim = simulate_run(
                pool_df=pool_df,
                config=cfg,
                run_id=0,
                drift_type=drift_type,
                drift_size=drift_size,
                drift_shape=drift_shape,
                random_seed=cfg.get("random_seed", 42),
            )
        except Exception as exc:
            st.error(f"Batch simulation failed: {exc}")
            return

    # ── Choose monitor ───────────────────────────────────────────────────────
    if strategy == "Fixed(k=1)":
        monitor = FixedMonitor(k=1, config=cfg)
    elif strategy == "Fixed(k=4)":
        monitor = FixedMonitor(k=4, config=cfg)
    else:
        from src.adaptive_monitor import AdaptiveMonitor
        monitor = AdaptiveMonitor(config=cfg)

    perf_monitor = PerformanceMonitor(config=cfg)

    # ── Stream results ───────────────────────────────────────────────────────
    records = []
    progress_bar = st.progress(0.0)
    status_container = st.empty()

    for b_idx, batch_df in enumerate(sim.batches):
        check_record = monitor.process_batch(b_idx, ref_df, batch_df)
        perf_record = perf_monitor.evaluate_batch(batch_df, batch_id=b_idx, labels_available=True)

        gt = sim.ground_truth_per_batch[b_idx] if b_idx < len(sim.ground_truth_per_batch) else {}

        records.append({
            "batch": b_idx,
            "mode": check_record.mode,
            "check_done": check_record.check_executed,
            "drift_detected": check_record.drift_detected,
            "confirmed_alert": check_record.confirmed_alert,
            "mae": perf_record.batch_mae,
            "perf_status": perf_record.status,
            "drift_injected": gt.get("drift_injected", False),
        })

        progress_bar.progress((b_idx + 1) / len(sim.batches))
        icon = "⚠️" if check_record.confirmed_alert else ("🔍" if check_record.drift_detected else "✅")
        status_container.caption(
            f"Batch {b_idx:02d}  |  Mode: {check_record.mode}  |  {icon}  |  "
            f"MAE: {perf_record.batch_mae:.4f}" if perf_record.batch_mae else
            f"Batch {b_idx:02d}  |  Mode: {check_record.mode}  |  {icon}"
        )

    progress_bar.empty()
    status_container.empty()

    result_df = pd.DataFrame(records)
    st.success(f"✅ Simulation complete — {len(sim.batches)} batches processed.")

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    total_checks = result_df["check_done"].sum()
    n_alerts = result_df["confirmed_alert"].sum()
    first_alert = result_df[result_df["confirmed_alert"]]["batch"].min() if n_alerts > 0 else None
    delay = (first_alert - drift_start) if (first_alert is not None and drift_type != "none") else None

    m1.metric("Full checks", f"{total_checks} / {len(result_df)}")
    m2.metric("Confirmed alerts", int(n_alerts))
    m3.metric("First alert batch", first_alert if first_alert is not None else "—")
    m4.metric("Detection delay", f"{delay} batches" if delay is not None else "—")

    # ── Drift flag chart ─────────────────────────────────────────────────────
    fig_drift = px.bar(
        result_df, x="batch", y="drift_detected",
        color="mode",
        title="Drift flag per batch (1 = flagged)",
        labels={"drift_detected": "Drift detected", "batch": "Batch"},
    )
    if drift_type != "none":
        fig_drift.add_vline(x=drift_start, line_dash="dash", line_color="red",
                            annotation_text="Drift start")
    st.plotly_chart(fig_drift, use_container_width=True)

    # ── MAE chart ────────────────────────────────────────────────────────────
    mae_df = result_df.dropna(subset=["mae"])
    if not mae_df.empty:
        baseline_mae = perf_monitor.baseline_mae
        fig_mae = px.line(mae_df, x="batch", y="mae", title="Batch MAE over time")
        fig_mae.add_hline(y=baseline_mae, line_dash="dot",
                          annotation_text=f"Baseline MAE ({baseline_mae:.4f})")
        fig_mae.add_hline(y=perf_monitor.degradation_threshold, line_dash="dash",
                          line_color="red",
                          annotation_text=f"Degradation threshold ({perf_monitor.degradation_threshold:.4f})")
        if drift_type != "none":
            fig_mae.add_vline(x=drift_start, line_dash="dash", line_color="orange",
                              annotation_text="Drift start")
        st.plotly_chart(fig_mae, use_container_width=True)

    # ── Raw results table ────────────────────────────────────────────────────
    with st.expander("Raw batch-level results"):
        st.dataframe(result_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2 – Investigation
# ──────────────────────────────────────────────────────────────────────────────

def _tab_investigation() -> None:
    st.header("🔎 Investigation")
    st.markdown(
        "Inspect a specific batch in detail — feature drift statistics, "
        "PSI ranking, and side-by-side distribution charts."
    )

    try:
        from src.data_loader import prepare_datasets, NUMERIC_FEATURES, CATEGORICAL_FEATURES
        from src.batch_simulator import simulate_run
        from src.drift_detector import detect_drift
    except ImportError as exc:
        st.error(f"Could not import required modules: {exc}")
        return

    cfg = _load_config()
    col1, col2, col3 = st.columns(3)
    with col1:
        drift_type = st.selectbox("Drift type (investigation)",
            ["none", "shift_mean", "shift_additive", "change_spread",
             "shift_category_mix"], index=1, key="inv_drift_type")
        drift_size = st.selectbox("Drift size", ["small", "medium", "large"], index=1, key="inv_size")
    with col2:
        drift_shape = st.selectbox("Drift shape", ["sudden", "gradual"], index=0, key="inv_shape")
        drift_start = st.slider("Drift start batch", 1, 44, 16, key="inv_start")
    with col3:
        batch_to_inspect = st.slider("Batch to inspect", 0, 47, drift_start + 2, key="inv_batch")

    if not st.button("Load batch", key="inv_run"):
        return

    with st.spinner("Loading …"):
        try:
            ref_df, pool_df, _ = prepare_datasets(cfg)
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            return

        cfg["drift_start_batch"] = drift_start
        sim = simulate_run(pool_df=pool_df, config=cfg, run_id=0,
                           drift_type=drift_type, drift_size=drift_size,
                           drift_shape=drift_shape,
                           random_seed=cfg.get("random_seed", 42))

    idx = min(batch_to_inspect, len(sim.batches) - 1)
    batch_df = sim.batches[idx]

    report = detect_drift(ref_df, batch_df, cfg, batch_id=idx)
    features_info = report.get("features", {})
    ranking = report.get("ranking", [])
    drift_detected = report.get("drift_detected", False)

    # ── Header status ────────────────────────────────────────────────────────
    if drift_detected:
        st.error(f"Batch {idx}: ⚠️ Drift detected — top feature: **{ranking[0] if ranking else '?'}**")
    else:
        st.success(f"Batch {idx}: ✅ No drift detected")

    # ── Feature statistics table ─────────────────────────────────────────────
    rows = []
    for feat in ranking:
        info = features_info.get(feat, {})
        rows.append({
            "Feature": feat,
            "PSI": info.get("psi", float("nan")),
            "KS stat": info.get("ks_stat", "—"),
            "KS p-value": info.get("ks_p", "—"),
            "Chi² p-value": info.get("chi2_p", "—"),
            "Flagged": "⚠️" if info.get("flagged") else "✅",
        })
    st.subheader("Feature statistics (ranked by PSI)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── PSI bar chart ────────────────────────────────────────────────────────
    psi_vals = {f: features_info[f].get("psi", 0) for f in ranking}
    fig_psi = px.bar(
        x=list(psi_vals.keys()), y=list(psi_vals.values()),
        labels={"x": "Feature", "y": "PSI"},
        title=f"PSI values – Batch {idx}",
        color=list(psi_vals.values()),
        color_continuous_scale="RdYlGn_r",
    )
    fig_psi.add_hline(y=cfg.get("psi_flag_threshold", 0.25), line_dash="dash",
                      annotation_text="Flag threshold")
    st.plotly_chart(fig_psi, use_container_width=True)

    # ── Distribution comparison ──────────────────────────────────────────────
    st.subheader("Distribution comparison — Reference vs Batch")
    selected_feat = st.selectbox("Select feature", ranking if ranking else list(features_info.keys()))
    if selected_feat:
        is_numeric = selected_feat in NUMERIC_FEATURES
        if is_numeric:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=ref_df[selected_feat].dropna(),
                                       name="Reference", opacity=0.6, nbinsx=30))
            fig.add_trace(go.Histogram(x=batch_df[selected_feat].dropna(),
                                       name="Batch", opacity=0.6, nbinsx=30))
            fig.update_layout(barmode="overlay",
                              title=f"Distribution: {selected_feat}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            ref_counts = ref_df[selected_feat].value_counts(normalize=True).rename("Reference")
            new_counts = batch_df[selected_feat].value_counts(normalize=True).rename("Batch")
            cmp = pd.concat([ref_counts, new_counts], axis=1).fillna(0).reset_index()
            cmp = cmp.melt(id_vars="index", var_name="Source", value_name="Share")
            fig = px.bar(cmp, x="index", y="Share", color="Source", barmode="group",
                         title=f"Category share: {selected_feat}")
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3 – Decisions & Incidents
# ──────────────────────────────────────────────────────────────────────────────

def _tab_decisions() -> None:
    st.header("📋 Decisions & Incidents")
    st.markdown(
        "Browse the incident log generated by the decision agent. "
        "Each row corresponds to a confirmed alert with the recommended action, "
        "guardrail status, and LLM summary."
    )

    log_path = RESULTS_DIR / "incident_log.csv"
    if not log_path.is_file():
        st.info(
            "No incident log found yet. Run a simulation that produces confirmed "
            "alerts (drift detected for 2 consecutive checks) to generate it."
        )
        return

    df = _load_csv(log_path)
    if df is None or df.empty:
        st.info("The incident log is empty.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Total incidents", len(df))
    if "guardrail_overridden" in df.columns:
        m2.metric("Guardrail overrides", int(df["guardrail_overridden"].sum()))
    if "chosen_action" in df.columns:
        top_action = df["chosen_action"].mode().iloc[0] if not df["chosen_action"].empty else "—"
        m3.metric("Most common action", top_action)

    # ── Action distribution chart ────────────────────────────────────────────
    if "chosen_action" in df.columns:
        action_counts = df["chosen_action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig = px.bar(action_counts, x="action", y="count",
                     title="Incident actions distribution",
                     labels={"action": "Action", "count": "Count"})
        st.plotly_chart(fig, use_container_width=True)

    # ── Full log table ───────────────────────────────────────────────────────
    st.subheader("Incident log")
    display_cols = [c for c in [
        "batch_id", "timestamp", "chosen_action", "default_action",
        "guardrail_overridden", "drift_detected", "performance_status"
    ] if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True)

    # ── Per-incident detail ──────────────────────────────────────────────────
    st.subheader("Incident detail")
    if "batch_id" in df.columns:
        selected_id = st.selectbox("Select batch ID", df["batch_id"].unique().tolist())
        row = df[df["batch_id"] == selected_id].iloc[0]
        st.markdown(f"**Batch ID:** {selected_id}  |  "
                    f"**Action:** `{row.get('chosen_action', '—')}`  |  "
                    f"**Guardrail override:** {bool(row.get('guardrail_overridden', False))}")

        for json_col, label in [("llm_summary", "LLM Summary"), ("drift_summary", "Drift Summary")]:
            if json_col in row.index and pd.notna(row[json_col]):
                st.markdown(f"**{label}**")
                try:
                    st.json(json.loads(row[json_col]))
                except Exception:
                    st.text(row[json_col])


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4 – Experiment Results
# ──────────────────────────────────────────────────────────────────────────────

def _tab_experiment_results() -> None:
    st.header("📊 Experiment Results")
    st.markdown(
        "Browse tables and charts generated by `experiments/run_experiments.py`. "
        "Run the experiments first to populate these results."
    )

    csv_files = sorted([f for f in RESULTS_DIR.glob("experiment_*.csv") if not f.stem.endswith("_summary")])
    if not csv_files:
        st.info(
            "No experiment results found in `results/`. "
            "Run:\n```\npython experiments/run_experiments.py --quick\n```"
        )
        return

    labels = sorted(list({f.stem.split("_")[1].upper() for f in csv_files}))
    selected = st.selectbox("Select experiment", labels)

    csv_path = RESULTS_DIR / f"experiment_{selected.lower()}.csv"
    df = _load_csv(csv_path)
    png_path = RESULTS_DIR / f"experiment_{selected.lower()}.png"

    if df is not None:
        st.subheader(f"Experiment {selected} — data ({len(df)} rows)")
        st.dataframe(df.head(50), use_container_width=True)

        # ── Auto-generate summary stats per experiment ───────────────────────
        if selected == "A" and {"rule", "detection_rate"}.issubset(df.columns):
            summary = df[df.get("has_drift", pd.Series(True, index=df.index)).astype(bool)]
            summary = summary.groupby("rule")[["detection_rate", "false_alarms",
                                               "total_check_time_sec"]].mean().round(4)
            st.subheader("Summary (mean across seeds & scenarios, drift runs only)")
            st.dataframe(summary, use_container_width=True)

        elif selected == "B" and {"strategy", "n_full_checks", "detection_delay"}.issubset(df.columns):
            summary = df[df.get("has_drift", pd.Series(True, index=df.index)).astype(bool)]
            summary = summary.groupby("strategy")[
                ["detection_delay", "n_full_checks", "check_rate_pct",
                 "missed_detection", "false_alarms"]
            ].agg(["mean", "std"]).round(4)
            st.subheader("Summary (mean ± std across seeds & scenarios, drift runs only)")
            st.dataframe(summary, use_container_width=True)

        elif selected == "C" and "precision_at_k" in df.columns:
            summary = df.groupby("scenario")[["precision_at_k"]].agg(["mean", "std"]).round(4)
            st.subheader("Precision@k summary")
            st.dataframe(summary, use_container_width=True)

        elif selected == "D" and "correct_action" in df.columns:
            summary = df.groupby(["scenario", "use_llm"])[
                ["correct_action", "unnecessary_retrain", "guardrail_overridden"]
            ].mean().round(4)
            st.subheader("Decision quality summary")
            st.dataframe(summary, use_container_width=True)

    if png_path.is_file():
        st.subheader("Summary chart")
        st.image(str(png_path), use_container_width=True)
    else:
        st.info(f"No PNG found for Experiment {selected}. It will appear here after running the experiments.")


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DriftGuard Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    st.title("🛡️ DriftGuard — Adaptive Drift Monitoring Dashboard")
    st.caption(
        "Monitors incoming data batches for distribution shift, ranks changed features, "
        "adapts checking frequency, and recommends actions."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔴 Live Simulation",
        "🔎 Investigation",
        "📋 Decisions & Incidents",
        "📊 Experiment Results",
    ])
    with tab1:
        _tab_live_simulation()
    with tab2:
        _tab_investigation()
    with tab3:
        _tab_decisions()
    with tab4:
        _tab_experiment_results()


if __name__ == "__main__":
    main()
