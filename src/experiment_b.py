"""
src/experiment_b.py

Experiment B – Fixed(k=1) vs Fixed(k=4) vs Adaptive monitoring comparison.

Per PROJECT_SPEC.md Section 7 / Experiment B:
  Goal: answer the main research question.
  Strategies: Fixed(k=1), Fixed(k=4), Adaptive — all on the SAME generated runs.
  Metrics:
    - Detection delay   = (first confirmed alert batch) - drift_start_batch
    - Missed-detection rate
    - False alarms per 48-batch no-drift run
    - Number of full checks
    - Total detection computation time

Each call to ``run(cfg)`` executes ONE full scenario (medium sudden drift on
retail_price by default, matching the main research scenario) with ONE seed.
The orchestrator calls run() N times with different seeds.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.adaptive_monitor import compare_strategies


# Scenarios for Experiment B — run the same sim through all three strategies
_SCENARIOS = [
    # No drift — measures false-alarm rate and check overhead
    {"drift_type": "none",       "drift_size": "medium", "drift_shape": "sudden",
     "label": "no_drift"},
    # Small / sudden drift
    {"drift_type": "shift_mean", "drift_size": "small",  "drift_shape": "sudden",
     "label": "small_sudden"},
    # Medium / sudden drift (MAIN scenario)
    {"drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
     "label": "medium_sudden"},
    # Large / sudden drift
    {"drift_type": "shift_mean", "drift_size": "large",  "drift_shape": "sudden",
     "label": "large_sudden"},
    # Medium / gradual drift
    {"drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "gradual",
     "label": "medium_gradual"},
    # Categorical drift
    {"drift_type": "shift_category_mix", "drift_size": "medium", "drift_shape": "sudden",
     "label": "categorical_medium_sudden"},
]


def run(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point called by run_experiments.py once per seed.

    Returns one metric dict per (scenario × strategy) combination.
    """
    seed = cfg.get("_current_seed", cfg.get("random_seed", 42))

    # Load (or reuse cached) reference and pool DataFrames
    ref_df, pool_df, _ = load_experiment_data(cfg)

    records: List[Dict[str, Any]] = []

    for scenario in _SCENARIOS:
        has_drift = scenario["drift_type"] != "none"
        drift_start = int(cfg.get("drift_start_batch", 16))

        sim = simulate_run(
            pool_df=pool_df,
            config=cfg,
            run_id=seed,
            drift_type=scenario["drift_type"],
            drift_size=scenario["drift_size"],
            drift_shape=scenario["drift_shape"],
            random_seed=seed,
        )

        # Run all three strategies on the SAME simulation run
        strategy_results = compare_strategies(ref_df, sim, config=cfg)

        for strategy_name, result in strategy_results.items():
            delay = result.detection_delay  # None if never detected
            records.append({
                "experiment": "B",
                "scenario": scenario["label"],
                "strategy": strategy_name,
                "has_drift": has_drift,
                "drift_type": scenario["drift_type"],
                "drift_size": scenario["drift_size"],
                "drift_shape": scenario["drift_shape"],
                "seed": seed,
                "detection_delay": delay if delay is not None else np.nan,
                "missed_detection": int(result.missed_detection),
                "false_alarms": result.false_alarms,
                "n_full_checks": result.checks_performed,
                "check_rate_pct": round(result.check_rate_pct, 2),
                "total_check_time_sec": round(result.total_check_time_sec, 4),
                "first_alert_batch": result.first_confirmed_alert_batch
                    if result.first_confirmed_alert_batch is not None else np.nan,
            })

    return records
