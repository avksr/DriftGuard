import os
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def build_manifest():
    shots_file = os.path.join(PROJECT_ROOT, "report_assets", "scripts", "screenshot_records.csv")
    shot_data = {}
    if os.path.exists(shots_file):
        sdf = pd.read_csv(shots_file)
        for _, row in sdf.iterrows():
            shot_data[str(row["figure_number"])] = {
                "date_time": str(row["date_time"]),
                "url": str(row["url"]),
                "seed": str(row["seed"]),
                "preset_or_tab": str(row["preset_or_tab"]),
                "size_px": str(row.get("size_px", "3200 x 8000")),
                "lowest_visible": str(row.get("lowest_visible", ""))
            }

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    git_commit = "edbd0a5"

    manifest_rows = [
        {
            "figure_number": "1.1",
            "file": "figures/fig_1_1_drift_patterns.png",
            "description_or_tab": "Synthetic illustration of three drift patterns: no drift, sudden drift, gradual drift; y-axis: Batch mean (arbitrary units)",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "synthetic illustration",
            "seed": "42",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "48 batches, drift_start=16, ramp_batches=8, baseline_mean=1000.0, shifted_mean=1400.0 (+40%), y_axis='Batch mean (arbitrary units)'"
        },
        {
            "figure_number": "2.1",
            "file": "figures/fig_2_1_architecture.png",
            "description_or_tab": "End-to-end system architecture block diagram and dataflow",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "system specification (PROJECT_SPEC.md & src/ architecture)",
            "seed": "N/A",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "raw=8906x17, ref=4453, pool=4453, batch_size=300, batches=48, train_frac=0.8, val_frac=0.2"
        },
        {
            "figure_number": "2.2",
            "file": "figures/fig_2_2_split.png",
            "description_or_tab": "Data splitting strategy and reference-only artifact isolation schematic",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "data/E-Commerce.csv via src/data_loader.py",
            "seed": "42",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "cleaned=8906, ref_half=4453, pool=4453, train=3562 (80%), val=891 (20%), price_cap=20576.42, top_brands=10, median_rating=3.0"
        },
        {
            "figure_number": "2.3",
            "file": "figures/fig_2_3_injection_effect.png",
            "description_or_tab": "Drift injection comparison: clean batch vs medium sudden price drift histograms",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "data/E-Commerce.csv via src/batch_simulator.py & src/drift_detector.py",
            "seed": "42",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "feature=retail_price, clean_batch_0_psi=0.0326, drift_batch_16_psi=0.2514, price_factor=1.40 (+40%), x_axis_clip_p99=13191.48 INR"
        },
        {
            "figure_number": "2.4",
            "file": "figures/fig_2_4_adaptive_states.png",
            "description_or_tab": "Adaptive monitoring finite state machine transition diagram",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "config.yaml adaptive parameters",
            "seed": "N/A",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "normal_interval=4, alert_interval=1, clean_checks_to_return=3, persistence_required=2, psi_thresh=0.25, p_val_thresh=0.01"
        },
        {
            "figure_number": "2.5",
            "file": "figures/fig_2_5_decision_flow.png",
            "description_or_tab": "Decision layer workflow: rules table, LLM advisor, and guardrail fallback",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "src/decision_agent.py specification",
            "seed": "N/A",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "6-row decision table, 4 actions (DO_NOTHING, INVESTIGATE, REQUEST_LABELS, PROPOSE_RETRAIN)"
        },
        {
            "figure_number": "2.6",
            "file": "figures/fig_2_6_dashboard_layout.png",
            "description_or_tab": "Schematic wireframe of 4 Streamlit dashboard tabs and user controls",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "app.py component architecture",
            "seed": "N/A",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "4 tabs: Live simulation, Investigation, Decisions and incidents, Experiment results"
        },
        {
            "figure_number": "3.1",
            "file": "figures/fig_3_1_experiment_a.png",
            "description_or_tab": "Experiment A result: Detector sensitivity, detection delay, and false-alarm trade-offs",
            "script": "report_assets/scripts/copy_experiment_figures.py (copy of results/experiment_a.png)",
            "data_source": "results/experiment_a.csv",
            "seed": "30 seeds aggregate",
            "git_commit": "fd11597 (run) / " + git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "10 scenarios, 3 detector rules, both_rule false_alarms=0.0, psi_only false_alarms=3.3 at N=150"
        },
        {
            "figure_number": "3.2",
            "file": "figures/fig_3_2_experiment_b.png",
            "description_or_tab": "Experiment B result: Monitoring strategies comparison (Fixed 1 vs Fixed 4 vs Adaptive)",
            "script": "report_assets/scripts/copy_experiment_figures.py (copy of results/experiment_b.png)",
            "data_source": "results/experiment_b.csv",
            "seed": "30 seeds aggregate",
            "git_commit": "62f7561 (run) / " + git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "Check savings: 25% (large), 33.1% (medium), 45.4% (gradual); Fixed_4 missed 10% gradual"
        },
        {
            "figure_number": "3.3",
            "file": "figures/fig_3_3_seed44_trace.png",
            "description_or_tab": "Seed 44 run trace: Top-feature PSI and strategy check/alert event timeline",
            "script": "report_assets/scripts/make_all_figures.py",
            "data_source": "simulated run seed 44 with medium sudden price drift",
            "seed": "44",
            "git_commit": git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "batches 12-30, drift_start=16, psi_thresh=0.25, confirmed alerts: Fixed_1=23, Fixed_4=20, Adaptive=28"
        },
        {
            "figure_number": "3.4",
            "file": "figures/fig_3_4_experiment_c.png",
            "description_or_tab": "Experiment C result: Feature ranking accuracy (Precision@k and Rank-1 accuracy)",
            "script": "report_assets/scripts/copy_experiment_figures.py (copy of results/experiment_c.png)",
            "data_source": "results/experiment_c.csv",
            "seed": "30 seeds aggregate",
            "git_commit": "97885d0 (run) / " + git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "Precision@1: retail_price=1.000, Region=1.000, product_rating=0.000"
        },
        {
            "figure_number": "3.5",
            "file": "figures/fig_3_5_experiment_d.png",
            "description_or_tab": "Experiment D result: Decision agent action match rate and retraining recommendations",
            "script": "report_assets/scripts/copy_experiment_figures.py (copy of results/experiment_d.png)",
            "data_source": "results/experiment_d.csv",
            "seed": "30 seeds aggregate",
            "git_commit": "fd11597 (run) / " + git_commit,
            "date_time_utc": now_utc,
            "key_metrics_or_numbers": "120/120 actions matched (100%), 0 unnecessary retrain proposals"
        },
        {
            "figure_number": "3.6",
            "file": "screenshots/fig_3_6_live_medium_sudden.png",
            "description_or_tab": "Live simulation tab after running 'Medium sudden price drift' preset (tall viewport full-page)",
            "script": "report_assets/scripts/recapture_screenshots_tall.py",
            "data_source": "live running Streamlit dashboard",
            "seed": shot_data.get("3.6", {}).get("seed", "42"),
            "git_commit": git_commit,
            "date_time_utc": shot_data.get("3.6", {}).get("date_time", now_utc),
            "key_metrics_or_numbers": f"URL: {shot_data.get('3.6', {}).get('url', 'http://localhost:8530')}, size={shot_data.get('3.6', {}).get('size_px', '3200 x 8000')}, lowest_visible='{shot_data.get('3.6', {}).get('lowest_visible')}'"
        },
        {
            "figure_number": "3.7",
            "file": "screenshots/fig_3_7_live_small_drift.png",
            "description_or_tab": "Live simulation tab after running 'Small price drift' preset (tall viewport full-page)",
            "script": "report_assets/scripts/recapture_screenshots_tall.py",
            "data_source": "live running Streamlit dashboard",
            "seed": shot_data.get("3.7", {}).get("seed", "42"),
            "git_commit": git_commit,
            "date_time_utc": shot_data.get("3.7", {}).get("date_time", now_utc),
            "key_metrics_or_numbers": f"URL: {shot_data.get('3.7', {}).get('url', 'http://localhost:8530')}, size={shot_data.get('3.7', {}).get('size_px', '3200 x 8000')}, lowest_visible='{shot_data.get('3.7', {}).get('lowest_visible')}'"
        },
        {
            "figure_number": "3.8",
            "file": "screenshots/fig_3_8_investigation.png",
            "description_or_tab": "Investigation tab for Batch 16 medium price drift (tall viewport full-page)",
            "script": "report_assets/scripts/recapture_screenshots_tall.py",
            "data_source": "live running Streamlit dashboard",
            "seed": shot_data.get("3.8", {}).get("seed", "42"),
            "git_commit": git_commit,
            "date_time_utc": shot_data.get("3.8", {}).get("date_time", now_utc),
            "key_metrics_or_numbers": f"URL: {shot_data.get('3.8', {}).get('url', 'http://localhost:8530')}, size={shot_data.get('3.8', {}).get('size_px', '3200 x 8000')}, lowest_visible='{shot_data.get('3.8', {}).get('lowest_visible')}'"
        },
        {
            "figure_number": "3.9",
            "file": "screenshots/fig_3_9_decisions.png",
            "description_or_tab": "Decisions and incidents tab: incident log, summaries, and export buttons (tall viewport full-page)",
            "script": "report_assets/scripts/recapture_screenshots_tall.py",
            "data_source": "live running Streamlit dashboard",
            "seed": shot_data.get("3.9", {}).get("seed", "42"),
            "git_commit": git_commit,
            "date_time_utc": shot_data.get("3.9", {}).get("date_time", now_utc),
            "key_metrics_or_numbers": f"URL: {shot_data.get('3.9', {}).get('url', 'http://localhost:8530')}, size={shot_data.get('3.9', {}).get('size_px', '3200 x 8000')}, lowest_visible='{shot_data.get('3.9', {}).get('lowest_visible')}'"
        },
        {
            "figure_number": "3.10",
            "file": "screenshots/fig_3_10_experiment_results.png",
            "description_or_tab": "Experiment results tab showing Experiment B table, figure, and manifest card (tall viewport full-page)",
            "script": "report_assets/scripts/recapture_screenshots_tall.py",
            "data_source": "live running Streamlit dashboard",
            "seed": shot_data.get("3.10", {}).get("seed", "30 seeds aggregate"),
            "git_commit": git_commit,
            "date_time_utc": shot_data.get("3.10", {}).get("date_time", now_utc),
            "key_metrics_or_numbers": f"URL: {shot_data.get('3.10', {}).get('url', 'http://localhost:8530')}, size={shot_data.get('3.10', {}).get('size_px', '3200 x 8000')}, lowest_visible='{shot_data.get('3.10', {}).get('lowest_visible')}'"
        }
    ]

    out_path = os.path.join(PROJECT_ROOT, "report_assets", "figures_manifest.csv")
    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Updated {out_path} with {len(df)} entries.")

if __name__ == "__main__":
    build_manifest()
