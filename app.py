"""
app.py – DriftGuard Streamlit Dashboard

Four tabs (Section 8 of PROJECT_SPEC.md):
1. Live simulation      – Side-by-side strategy comparison under controlled drift presets
2. Investigation        – Inspection of feature drift stats, ranking, and distributions
3. Decisions & incidents – Incident log, decision rationale, guardrails, and retraining
4. Experiment results   – Empirical tables and figures loaded strictly from results/

UI compliance (PROJECT_SPEC.md Section 8.1):
- No emojis, gradients, glow effects, decorative icons, or marketing text.
- One accent colour (#1f77b4), neutral grey (#7f7f7f) for reference data.
- Red (#d62728) used only for confirmed alerts.
- Labelled threshold and drift-start lines.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

from src.utils import load_config
from src.data_loader import prepare_datasets, NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, TARGET_COLUMN
from src.baseline_model import load_baseline_model, load_baseline_metrics
from src.batch_simulator import simulate_run, apply_drift_to_batch, sample_batch
from src.drift_detector import detect_drift
from src.quality_checks import check_batch_quality
from src.adaptive_monitor import FixedMonitor, AdaptiveMonitor
from src.performance_monitor import PerformanceMonitor


# ──────────────────────────────────────────────────────────────────────────────
# Cached Resource and Data Loaders
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_datasets(config_path: str = "config.yaml") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load reference and pool datasets safely, caching in memory."""
    cfg = load_config(config_path)
    csv_path = Path(cfg.get("data", {}).get("csv_path", "data/E-Commerce.csv"))
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        return None, None
    try:
        ref_df, pool_df, _ = prepare_datasets(cfg)
        return ref_df, pool_df
    except Exception:
        return None, None


@st.cache_resource(show_spinner=False)
def get_cached_baseline_model():
    """Load pre-trained baseline model pipeline into memory once."""
    try:
        return load_baseline_model()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_cached_csv(path_str: str) -> Optional[pd.DataFrame]:
    """Read a CSV safely and cache the resulting DataFrame."""
    p = Path(path_str)
    if not p.is_file():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# UI Guide: Explanations
# ──────────────────────────────────────────────────────────────────────────────

def _render_glossary() -> None:
    """Render plain definitions for key concepts without hype."""
    with st.expander("How to read this page"):
        st.markdown(
            "- **Population Stability Index (PSI):** Measures the divergence between the reference distribution "
            "and incoming batch. Values below 0.10 indicate no change; 0.10 to 0.25 indicate moderate change; "
            "0.25 and above indicate significant shift (flagged threshold).\n"
            "- **Kolmogorov-Smirnov (KS) test:** Two-sample non-parametric statistical test comparing continuous feature distributions. "
            "A p-value below 0.01 indicates statistical significance.\n"
            "- **Full check:** Running the statistical tests across every feature of a batch.\n"
            "- **Confirmed alert:** Requires two flagged checks in a row. A single unflagged check resets the counter.\n"
            "- **Detection delay:** Batches elapsed from the injected drift start to the first confirmed alert."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Live Simulation
# ──────────────────────────────────────────────────────────────────────────────

def _tab_live_simulation() -> None:
    st.subheader("Live simulation")
    st.markdown(
        "Evaluate Fixed(1), Fixed(4), and Adaptive monitoring strategies side-by-side on the same simulated batch stream."
    )

    ref_df, pool_df = get_datasets()
    cached_model = get_cached_baseline_model()

    if ref_df is None or pool_df is None:
        st.error(
            "Dataset not found at `data/E-Commerce.csv`. Please place the dataset in the `data/` directory "
            "per instructions in `data/README.md`."
        )
        return

    if cached_model is None:
        st.error(
            "Baseline model artifact not found at `models/baseline_model.joblib`. "
            "Please train the baseline model by running: `python -m src.baseline_model`"
        )
        return

    cfg = load_config()

    presets = {
        "Medium sudden price drift": {
            "drift_type": "shift_mean",
            "drift_size": "medium",
            "drift_shape": "sudden",
            "affected_features": ["retail_price"],
            "label_shift_factor": None,
            "labels_available": True,
            "seed": 42,
            "desc": "Main comparison case: 40% retail price shift at batch 16.",
        },
        "Large sudden price drift": {
            "drift_type": "shift_mean",
            "drift_size": "large",
            "drift_shape": "sudden",
            "affected_features": ["retail_price"],
            "label_shift_factor": None,
            "labels_available": True,
            "seed": 42,
            "desc": "Pronounced shift: 100% price increase at batch 16.",
        },
        "Small price drift": {
            "drift_type": "shift_mean",
            "drift_size": "small",
            "drift_shape": "sudden",
            "affected_features": ["retail_price"],
            "label_shift_factor": None,
            "labels_available": True,
            "seed": 42,
            "desc": "Subtle shift: 15% price increase, near the statistical detection boundary.",
        },
        "Concept shift": {
            "drift_type": "label_shift",
            "drift_size": "large",
            "drift_shape": "sudden",
            "affected_features": None,
            "label_shift_factor": 1.8,
            "labels_available": True,
            "seed": 42,
            "desc": "Target distribution shifts while input features remain unchanged.",
        },
        "Custom": {
            "desc": "Specify custom drift type, magnitude, and affected columns.",
        },
    }

    col_preset, col_info = st.columns([1, 2])
    with col_preset:
        preset_choice = st.selectbox(
            "Preset scenario",
            list(presets.keys()),
            index=0,
            help="Choose a pre-configured research scenario or select Custom.",
        )

    with col_info:
        st.caption(presets[preset_choice]["desc"])

    # Scenario parameters setup
    drift_start = 16
    n_batches = 48

    if preset_choice == "Custom":
        c1, c2, c3 = st.columns(3)
        with c1:
            drift_type = st.selectbox(
                "Drift type",
                ["none", "shift_mean", "shift_additive", "change_spread", "shift_category_mix", "label_shift"],
                index=1,
                help="Type of statistical perturbation injected.",
            )
            drift_size = st.selectbox("Drift size", ["small", "medium", "large"], index=1)
        with c2:
            drift_shape = st.selectbox("Drift shape", ["sudden", "gradual"], index=0)
            seed = st.number_input("Random seed", value=42, step=1)
        with c3:
            labels_available = st.checkbox("Ground-truth labels available", value=True)
            label_shift_factor = 1.8 if drift_type == "label_shift" else None
            affected_features = ["retail_price"] if drift_type != "label_shift" else None
    else:
        preset_data = presets[preset_choice]
        drift_type = preset_data["drift_type"]
        drift_size = preset_data["drift_size"]
        drift_shape = preset_data["drift_shape"]
        affected_features = preset_data["affected_features"]
        label_shift_factor = preset_data["label_shift_factor"]
        labels_available = preset_data["labels_available"]
        seed = preset_data["seed"]

    st.markdown("Press **Run simulation** to process 48 batches across Fixed(1), Fixed(4), and Adaptive monitors.")
    run_btn = st.button("Run simulation", type="primary")

    if not run_btn:
        st.info("Select a preset scenario above and press Run simulation.")
        return

    # Simulation execution
    sim_cfg = dict(cfg)
    sim_cfg["drift_start_batch"] = drift_start
    sim_cfg["n_batches_per_run"] = n_batches

    prog_bar = st.progress(0.0)
    status_text = st.empty()

    status_text.text("Simulating 48 batch sequence...")
    sim = simulate_run(
        pool_df=pool_df,
        config=sim_cfg,
        run_id=seed,
        drift_type=drift_type,
        drift_size=drift_size,
        drift_shape=drift_shape,
        affected_features=affected_features,
        random_seed=seed,
    )

    mon_f1 = FixedMonitor(k=1, config=sim_cfg)
    mon_f4 = FixedMonitor(k=4, config=sim_cfg)
    mon_ad = AdaptiveMonitor(config=sim_cfg)
    perf_mon = PerformanceMonitor(config=sim_cfg, model=cached_model)

    results_f1: List[Dict[str, Any]] = []
    results_f4: List[Dict[str, Any]] = []
    results_ad: List[Dict[str, Any]] = []
    perf_records: List[Dict[str, Any]] = []
    top_psi_records: List[Dict[str, Any]] = []

    for b_idx, batch_df in enumerate(sim.batches):
        status_text.text(f"Processing batch {b_idx + 1} of {len(sim.batches)}...")
        prog_bar.progress((b_idx + 1) / len(sim.batches))

        # Check quality
        quality = check_batch_quality(batch_df, config=sim_cfg, ref_df=ref_df)
        valid = bool(quality.get("passed", True))

        # If label shift requested
        eval_batch = batch_df
        if label_shift_factor is not None and labels_available and TARGET_COLUMN in batch_df.columns:
            eval_batch = batch_df.copy()
            eval_batch[TARGET_COLUMN] = (eval_batch[TARGET_COLUMN] * label_shift_factor).clip(upper=1.0)

        # Performance evaluation
        perf_rec = perf_mon.evaluate_batch(eval_batch, batch_id=b_idx, labels_available=labels_available)
        perf_records.append({
            "batch": b_idx,
            "mae": perf_rec.batch_mae,
            "status": perf_rec.status,
            "is_degraded": perf_rec.is_degraded,
        })

        if not valid:
            # Invalid batches not counted as drift
            continue

        rec_f1 = mon_f1.process_batch(b_idx, ref_df, batch_df)
        rec_f4 = mon_f4.process_batch(b_idx, ref_df, batch_df)
        rec_ad = mon_ad.process_batch(b_idx, ref_df, batch_df)

        results_f1.append({
            "batch": b_idx,
            "strategy": "Fixed(1)",
            "check_executed": bool(rec_f1.check_executed),
            "flagged": bool(rec_f1.drift_detected) if rec_f1.check_executed else False,
            "confirmed_alert": bool(rec_f1.confirmed_alert) if rec_f1.check_executed else False,
            "top_psi": rec_f1.top_feature_psi,
        })
        results_f4.append({
            "batch": b_idx,
            "strategy": "Fixed(4)",
            "check_executed": bool(rec_f4.check_executed),
            "flagged": bool(rec_f4.drift_detected) if rec_f4.check_executed else False,
            "confirmed_alert": bool(rec_f4.confirmed_alert) if rec_f4.check_executed else False,
            "top_psi": rec_f4.top_feature_psi,
        })
        results_ad.append({
            "batch": b_idx,
            "strategy": "Adaptive",
            "check_executed": bool(rec_ad.check_executed),
            "flagged": bool(rec_ad.drift_detected) if rec_ad.check_executed else False,
            "confirmed_alert": bool(rec_ad.confirmed_alert) if rec_ad.check_executed else False,
            "mode": rec_ad.mode,
            "top_psi": rec_ad.top_feature_psi,
        })

        if rec_f1.check_executed:
            top_psi_records.append({
                "batch": b_idx,
                "top_psi": rec_f1.top_feature_psi,
            })

    prog_bar.empty()
    status_text.empty()

    # Strategy comparison metrics
    df_f1 = pd.DataFrame(results_f1)
    df_f4 = pd.DataFrame(results_f4)
    df_ad = pd.DataFrame(results_ad)

    def _get_metrics(df_strat: pd.DataFrame, strat_name: str) -> Dict[str, Any]:
        n_checks = int(df_strat["check_executed"].sum())
        alerts = df_strat[df_strat["confirmed_alert"]]
        first_alert = int(alerts["batch"].min()) if not alerts.empty else None
        delay = (first_alert - drift_start) if (first_alert is not None and drift_type != "none") else None
        return {
            "strategy": strat_name,
            "checks": n_checks,
            "first_alert": first_alert,
            "delay": delay,
        }

    m_f1 = _get_metrics(df_f1, "Fixed(1)")
    m_f4 = _get_metrics(df_f4, "Fixed(4)")
    m_ad = _get_metrics(df_ad, "Adaptive")

    f1_checks = max(1, m_f1["checks"])
    m_f1["saved_pct"] = 0.0
    m_f4["saved_pct"] = round((1.0 - m_f4["checks"] / f1_checks) * 100.0, 1)
    m_ad["saved_pct"] = round((1.0 - m_ad["checks"] / f1_checks) * 100.0, 1)

    # ── Summary strip ────────────────────────────────────────────────────────
    st.markdown("#### Strategy performance summary")
    cols = st.columns(3)
    for col, m in zip(cols, [m_f1, m_f4, m_ad]):
        with col:
            st.markdown(f"**{m['strategy']}**")
            st.write(f"First confirmed alert: {m['first_alert'] if m['first_alert'] is not None else 'None'}")
            st.write(f"Detection delay: {m['delay'] if m['delay'] is not None else 'N/A'}")
            st.write(f"Full checks: {m['checks']} / 48")
            st.write(f"Checks saved vs Fixed(1): {m['saved_pct']}%")

    # ── Timeline Chart ───────────────────────────────────────────────────────
    st.markdown("#### Monitoring timeline by strategy")
    fig_timeline = go.Figure()

    strategy_order = ["Fixed(1)", "Fixed(4)", "Adaptive"]
    all_strat_data = [df_f1, df_f4, df_ad]

    for strat_name, df_s in zip(strategy_order, all_strat_data):
        # Normal checks
        checks_only = df_s[df_s["check_executed"] & (~df_s["flagged"]) & (~df_s["confirmed_alert"])]
        if not checks_only.empty:
            fig_timeline.add_trace(go.Scatter(
                x=checks_only["batch"],
                y=[strat_name] * len(checks_only),
                mode="markers",
                marker=dict(symbol="circle", size=8, color="#1f77b4"),
                name=f"{strat_name}: Check",
                showlegend=False,
            ))

        # Flagged checks
        flagged = df_s[df_s["check_executed"] & df_s["flagged"] & (~df_s["confirmed_alert"])]
        if not flagged.empty:
            fig_timeline.add_trace(go.Scatter(
                x=flagged["batch"],
                y=[strat_name] * len(flagged),
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="#1f77b4"),
                name=f"{strat_name}: Flagged",
                showlegend=False,
            ))

        # Confirmed alert
        confirmed = df_s[df_s["confirmed_alert"]]
        if not confirmed.empty:
            fig_timeline.add_trace(go.Scatter(
                x=confirmed["batch"],
                y=[strat_name] * len(confirmed),
                mode="markers",
                marker=dict(symbol="star", size=14, color="#d62728"),
                name=f"{strat_name}: Confirmed alert",
                showlegend=False,
            ))

    # Add dummy entries for clean legend
    fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="circle", size=8, color="#1f77b4"), name="Clean check"))
    fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="diamond", size=10, color="#1f77b4"), name="Flagged check"))
    fig_timeline.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="star", size=14, color="#d62728"), name="Confirmed alert"))

    if drift_type != "none":
        fig_timeline.add_vline(x=drift_start, line_dash="dash", line_color="#7f7f7f", annotation_text="Drift start (batch 16)")

    fig_timeline.update_layout(
        xaxis_title="Batch number",
        yaxis_title="Strategy",
        height=280,
        margin=dict(l=40, r=40, t=30, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Top Feature PSI Chart ────────────────────────────────────────────────
    st.markdown("#### Top-feature Population Stability Index (PSI)")
    df_psi = pd.DataFrame(top_psi_records)
    if not df_psi.empty:
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=df_psi["batch"],
            y=df_psi["top_psi"],
            mode="lines+markers",
            name="Top feature PSI",
            line=dict(color="#1f77b4", width=2),
        ))
        fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#d62728", annotation_text="PSI threshold = 0.25")
        if drift_type != "none":
            fig_psi.add_vline(x=drift_start, line_dash="dash", line_color="#7f7f7f", annotation_text="Drift start")

        fig_psi.update_layout(
            xaxis_title="Batch number",
            yaxis_title="Top feature PSI",
            height=260,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    # ── Model Performance (MAE) Chart ────────────────────────────────────────
    st.markdown("#### Model error (Mean Absolute Error)")
    df_perf = pd.DataFrame(perf_records)
    if labels_available and not df_perf["mae"].isna().all():
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Scatter(
            x=df_perf["batch"],
            y=df_perf["mae"],
            mode="lines+markers",
            name="Batch MAE",
            line=dict(color="#1f77b4", width=2),
        ))
        b_mae = perf_mon.baseline_mae
        deg_thresh = perf_mon.degradation_threshold
        fig_mae.add_hline(y=b_mae, line_dash="dot", line_color="#7f7f7f", annotation_text=f"Baseline MAE ({b_mae:.4f})")
        fig_mae.add_hline(y=deg_thresh, line_dash="dash", line_color="#d62728", annotation_text=f"Degradation limit 1.2x ({deg_thresh:.4f})")
        if drift_type != "none":
            fig_mae.add_vline(x=drift_start, line_dash="dash", line_color="#7f7f7f", annotation_text="Drift start")

        fig_mae.update_layout(
            xaxis_title="Batch number",
            yaxis_title="Batch MAE",
            height=260,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    else:
        st.info("Labels unavailable: model error is unverified for this run.")


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Investigation
# ──────────────────────────────────────────────────────────────────────────────

def _tab_investigation() -> None:
    st.subheader("Investigation")
    st.markdown(
        "Examine statistical drift test metrics and distribution changes for any chosen batch."
    )

    ref_df, pool_df = get_datasets()
    if ref_df is None or pool_df is None:
        st.error("Dataset not found. Please place `data/E-Commerce.csv` in the data directory.")
        return

    cfg = load_config()

    col1, col2 = st.columns([1, 2])
    with col1:
        batch_idx = st.slider("Select batch number to inspect", 0, 47, 18, help="Index of the batch to evaluate.")
        drift_scenario = st.selectbox(
            "Drift condition to apply",
            ["Medium price drift", "Large price drift", "Region category drift", "No drift"],
            index=0,
        )

    # Generate the single target batch
    sc_map = {
        "Medium price drift": ("shift_mean", "medium", ["retail_price"]),
        "Large price drift": ("shift_mean", "large", ["retail_price"]),
        "Region category drift": ("shift_category_mix", "large", ["Region"]),
        "No drift": ("none", "medium", None),
    }
    dtype, dsize, dfeats = sc_map[drift_scenario]

    sim = simulate_run(
        pool_df=pool_df,
        config=cfg,
        run_id=42,
        drift_type=dtype,
        drift_size=dsize,
        drift_shape="sudden",
        affected_features=dfeats,
        random_seed=42,
    )

    batch_df = sim.batches[batch_idx]
    report = detect_drift(ref_df, batch_df, cfg, batch_id=batch_idx)

    features_dict = report.get("features", {})
    ranking = report.get("ranking", [])
    drift_flag = report.get("drift_detected", False)

    if drift_flag:
        st.markdown(f"**Batch {batch_idx}: Drift detected.** Top ranked feature: `{ranking[0]}`")
    else:
        st.markdown(f"**Batch {batch_idx}: No drift detected.**")

    # Feature statistics table
    table_rows = []
    for feat in ranking:
        info = features_dict.get(feat, {})
        is_num = feat in NUMERIC_FEATURES
        test_type = "KS test" if is_num else "Chi-square"
        stat_val = info.get("ks_stat" if is_num else "chi2_stat")
        p_val = info.get("ks_p" if is_num else "chi2_p")
        psi_val = info.get("psi", 0.0)
        flagged = "Yes" if info.get("flagged") else "No"

        table_rows.append({
            "Feature": str(feat),
            "Test": str(test_type),
            "Statistic": f"{float(stat_val):.4f}" if stat_val is not None and not np.isnan(stat_val) else "—",
            "p-value": f"{float(p_val):.4e}" if p_val is not None and not np.isnan(p_val) else "—",
            "PSI": f"{float(psi_val):.4f}",
            "Flagged": str(flagged),
        })

    st.markdown("#### Feature drift metrics (ranked by PSI)")
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    # Ranked bar chart
    psi_series = [features_dict[f]["psi"] for f in ranking]
    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=psi_series,
        y=ranking,
        orientation="h",
        marker_color="#1f77b4",
        name="PSI",
    ))
    fig_rank.add_vline(x=0.25, line_dash="dash", line_color="#d62728", annotation_text="PSI threshold = 0.25")
    fig_rank.update_layout(
        title="Features ordered by Population Stability Index (PSI)",
        xaxis_title="PSI",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Distribution comparison
    st.markdown("#### Reference versus incoming batch distribution")
    selected_feature = st.selectbox("Feature to inspect distribution", ranking, index=0)

    if selected_feature in NUMERIC_FEATURES:
        ref_vals = ref_df[selected_feature].dropna()
        batch_vals = batch_df[selected_feature].dropna()
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=ref_vals,
            name="Reference data",
            marker_color="#7f7f7f",
            opacity=0.6,
            nbinsx=25,
            histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=batch_vals,
            name="Incoming batch",
            marker_color="#1f77b4",
            opacity=0.6,
            nbinsx=25,
            histnorm="probability density",
        ))
        fig_dist.update_layout(
            barmode="overlay",
            xaxis_title=selected_feature,
            yaxis_title="Probability density",
            height=320,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    else:
        ref_shares = ref_df[selected_feature].value_counts(normalize=True).rename("Reference")
        batch_shares = batch_df[selected_feature].value_counts(normalize=True).rename("Batch")
        merged = pd.concat([ref_shares, batch_shares], axis=1).fillna(0.0).reset_index()
        merged.columns = ["Category", "Reference", "Incoming"]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=merged["Category"],
            y=merged["Reference"],
            name="Reference data",
            marker_color="#7f7f7f",
        ))
        fig_dist.add_trace(go.Bar(
            x=merged["Category"],
            y=merged["Incoming"],
            name="Incoming batch",
            marker_color="#1f77b4",
        ))
        fig_dist.update_layout(
            barmode="group",
            xaxis_title=selected_feature,
            yaxis_title="Category share",
            height=320,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig_dist, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Decisions and Incidents
# ──────────────────────────────────────────────────────────────────────────────

def _tab_decisions() -> None:
    st.subheader("Decisions and incidents")
    st.markdown(
        "Incident records generated by the decision agent, including recommended actions, "
        "guardrail overrides, and model retraining decisions."
    )

    incidents_csv = RESULTS_DIR / "incidents.csv"
    df_inc = get_cached_csv(str(incidents_csv))

    st.caption("Rule-based summaries are in use (no external LLM API key required).")

    if df_inc is None or df_inc.empty:
        st.info("No recorded incidents found in `results/incidents.csv`.")
        return

    # Incident summary statistics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total recorded incidents", len(df_inc))
    overrides = int(df_inc["guardrail_overridden"].sum()) if "guardrail_overridden" in df_inc.columns else 0
    m2.metric("Guardrail overrides", overrides)
    retrain_calls = int((df_inc["chosen_action"] == "PROPOSE_RETRAIN").sum()) if "chosen_action" in df_inc.columns else 0
    m3.metric("Retraining proposed", retrain_calls)

    # Incident table display
    st.markdown("#### Incident log")
    disp_cols = [c for c in [
        "incident_id", "batch_id", "timestamp", "chosen_action",
        "default_action", "guardrail_overridden", "drift_detected", "performance_status"
    ] if c in df_inc.columns]
    st.dataframe(df_inc[disp_cols] if disp_cols else df_inc, use_container_width=True)

    # Markdown export button
    md_lines = ["# DriftGuard Incident Report", "", f"Total incidents logged: {len(df_inc)}", ""]
    for _, row in df_inc.iterrows():
        md_lines.append(f"### Incident Batch {row.get('batch_id', 'N/A')}")
        md_lines.append(f"- Timestamp: {row.get('timestamp', 'N/A')}")
        md_lines.append(f"- Chosen action: {row.get('chosen_action', 'N/A')}")
        md_lines.append(f"- Default action: {row.get('default_action', 'N/A')}")
        md_lines.append(f"- Guardrail overridden: {row.get('guardrail_overridden', False)}")
        md_lines.append(f"- Drift detected: {row.get('drift_detected', False)}")
        md_lines.append(f"- Performance status: {row.get('performance_status', 'N/A')}")
        md_lines.append("")

    report_content = "\n".join(md_lines)
    st.download_button(
        label="Export incident report",
        data=report_content,
        file_name="driftguard_incident_report.md",
        mime="text/markdown",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: Experiment Results
# ──────────────────────────────────────────────────────────────────────────────

def _tab_experiment_results() -> None:
    st.subheader("Experiment results")
    st.markdown(
        "Empirical evaluation results loaded directly from `results/` produced by research runs."
    )

    manifest_file = RESULTS_DIR / "run_manifest.json"
    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution timestamp (UTC)", manifest.get("timestamp_utc", "—")[:19])
            c2.metric("Git commit", manifest.get("git_commit", "—")[:8])
            c3.metric("Number of seeds", manifest.get("number_of_seeds", "—"))
        except Exception:
            pass

    exp_choice = st.selectbox(
        "Select experiment",
        ["A", "B", "C", "D"],
        index=1,
        help="Choose experiment to inspect data table and summary chart.",
    )

    csv_path = RESULTS_DIR / f"experiment_{exp_choice.lower()}_summary.csv"
    png_path = RESULTS_DIR / f"experiment_{exp_choice.lower()}.png"

    df_summary = get_cached_csv(str(csv_path))

    captions = {
        "A": "Detection delay and false alarms per detector rule and scenario, mean and standard deviation over 30 seeds.",
        "B": "Detection delay and number of full checks per monitoring strategy, mean and standard deviation over 30 seeds.",
        "C": "Feature ranking precision@k and Rank-1 accuracy evaluated over 5 post-drift batches across 30 seeds.",
        "D": "Action match rate and retraining recommendations for the decision agent under streaming evaluation across 30 seeds.",
    }

    if df_summary is not None:
        st.markdown(f"#### Experiment {exp_choice} summary table")
        st.dataframe(df_summary, use_container_width=True)
        st.caption(captions.get(exp_choice, ""))

    if png_path.is_file():
        st.markdown(f"#### Experiment {exp_choice} figure")
        st.image(str(png_path), use_container_width=True)
        st.caption(captions.get(exp_choice, ""))
    else:
        st.info(f"Figure file `{png_path.name}` not found in results directory.")


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Shell
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DriftGuard Monitoring Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("DriftGuard")
    st.caption("Tabular model monitoring, distribution shift detection, and adaptive checking frequency.")

    _render_glossary()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live simulation",
        "Investigation",
        "Decisions and incidents",
        "Experiment results",
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
