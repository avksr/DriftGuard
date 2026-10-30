"""
Phase 5 Demonstration Script: Fixed vs. Adaptive Monitoring Comparison
Author: DriftMonitor Team
Purpose: Compares Fixed(k=1), Fixed(k=4), and Adaptive monitoring strategies
         on the exact same 48-batch simulation run per PROJECT_SPEC.md Section 5.5, 5.6, 5.9,
         and Section 11.3 Phase 5.
"""

import sys
import os
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_config, prepare_datasets
from src.batch_simulator import simulate_run
from src.adaptive_monitor import compare_strategies
from src.performance_monitor import PerformanceMonitor
from src.incident_logger import IncidentLogger


def run_phase5_comparison():
    print("=" * 85)
    print(" DRIFTMONITOR: PHASE 5 MONITORING STRATEGIES & PERFORMANCE COMPARISON")
    print("=" * 85)

    # 1. Load reference data and clean pool
    config = load_config("config.yaml")
    ref_df, pool_df, _ = prepare_datasets("config.yaml", save_to_disk=False)
    print(f"Loaded Reference data: {len(ref_df):,} rows")
    print(f"Loaded Clean Pool:     {len(pool_df):,} rows")

    # 2. Simulate 48-batch run: sudden medium drift on retail_price at batch 16
    drift_start = 16
    sim_run = simulate_run(
        pool_df=pool_df,
        config=config,
        run_id="demo_phase_5",
        drift_type="shift_mean",
        drift_size="medium",
        drift_shape="sudden",
        drift_start_batch=drift_start,
        affected_features=["retail_price"]
    )

    print("\nSimulation Run Configured:")
    print(f"  Total Batches:         {len(sim_run.batches)}")
    print(f"  Drift Type:            {sim_run.drift_type} ({sim_run.drift_size})")
    print(f"  Drift Shape:           {sim_run.drift_shape}")
    print(f"  Drift Start Batch:     {drift_start} (0-indexed)")
    print(f"  Target Feature(s):     {sim_run.affected_features}")
    print(f"  Persistence Required:  {config.get('persistence_required', 2)} consecutive flagged checks")
    print("-" * 85)

    # 3. Run all strategies on the exact same run
    print("\nExecuting Monitoring Strategies:")
    results = compare_strategies(ref_df, sim_run, config=config)

    # 4. Initialize Performance Monitor & Incident Logger for Demonstration
    perf_monitor = PerformanceMonitor(config=config)
    incident_logger = IncidentLogger(log_dir="results")

    # Evaluate performance across batches
    print("\nEvaluating Model Performance & Logging Incidents for Adaptive Monitor:")
    for b_idx, batch in enumerate(sim_run.batches):
        perf_rec = perf_monitor.evaluate_batch(batch, batch_id=b_idx, labels_available=True)
        adaptive_check = results["Adaptive"].checks_timeline[b_idx]

        # Log incident if confirmed alert triggered or performance degraded
        if adaptive_check.confirmed_alert or perf_rec.is_degraded:
            action = "PROPOSE_RETRAIN" if perf_rec.is_degraded else "INVESTIGATE"
            summary = (
                f"Batch {b_idx}: Confirmed alert={adaptive_check.confirmed_alert}, "
                f"Performance={perf_rec.status} (MAE={perf_rec.batch_mae:.4f})"
            )
            incident_logger.log_incident(
                batch_id=b_idx,
                action=action,
                summary=summary,
                run_id=sim_run.run_id,
                monitoring_mode=adaptive_check.mode,
                drift_detected=bool(adaptive_check.drift_detected),
                top_drifted_features=adaptive_check.ranking[:3],
                top_feature_psi=adaptive_check.top_feature_psi,
                performance_status=perf_rec.status,
                batch_mae=perf_rec.batch_mae
            )

    # 5. Print Comparative Results Table
    fixed1 = results["Fixed_1"]
    fixed4 = results["Fixed_4"]
    adaptive = results["Adaptive"]

    print("\n" + "=" * 85)
    print(" MONITORING STRATEGY COMPARISON RESULTS")
    print("=" * 85)
    header = f"{'Strategy':<14} | {'1st Alert':<10} | {'Delay':<8} | {'Checks':<8} | {'Check %':<9} | {'Time (s)':<10} | {'Savings vs Fixed-1':<18}"
    print(header)
    print("-" * len(header))

    for name, res in [("Fixed(k=1)", fixed1), ("Fixed(k=4)", fixed4), ("Adaptive", adaptive)]:
        alert_str = str(res.first_confirmed_alert_batch) if res.first_confirmed_alert_batch is not None else "None"
        delay_str = f"+{res.detection_delay} b" if res.detection_delay is not None else "Missed"
        checks_str = f"{res.checks_performed}/{res.total_batches}"
        rate_str = f"{res.check_rate_pct:.1f}%"
        time_str = f"{res.total_check_time_sec:.3f}s"
        
        savings = (1.0 - (res.checks_performed / fixed1.checks_performed)) * 100.0 if fixed1.checks_performed > 0 else 0.0
        savings_str = f"{savings:>5.1f}% fewer checks" if name != "Fixed(k=1)" else "Baseline (0%)"

        print(f"{name:<14} | {alert_str:<10} | {delay_str:<8} | {checks_str:<8} | {rate_str:<9} | {time_str:<10} | {savings_str:<18}")

    print("-" * len(header))

    # 6. Mode Transitions for Adaptive Strategy
    print("\nAdaptive Strategy Mode Transitions:")
    if adaptive.mode_transitions:
        for t in adaptive.mode_transitions:
            print(f"  - Batch {t.batch_id:2d}: [{t.from_mode} -> {t.to_mode}] Reason: {t.reason}")
    else:
        print("  - No mode transitions occurred.")

    # 7. Summary of Incidents Logged
    print(f"\nIncidents Logged: {len(incident_logger.get_incidents())} incidents written to memory/disk.")
    csv_path = incident_logger.save_to_csv()
    json_path = incident_logger.save_to_json()
    print(f"Saved incidents log to: {csv_path} and {json_path}")
    print("=" * 85)
    print(" PHASE 5 DEMONSTRATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_phase5_comparison()
