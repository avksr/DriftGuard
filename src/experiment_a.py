"""
src/experiment_a.py

Experiment A – Compare KS-only, PSI-only, and the combined (both) detector rule.

Per PROJECT_SPEC.md Section 7 / Experiment A:
  Goal: compare detector rules.
  Scenarios: no drift; small/medium/large drift; sudden vs gradual;
             one changed feature vs several; batch sizes 150 / 300 / 600.
  Metrics: detection rate, false-alarm rate, detection delay, computation time.

Each call to ``run(cfg)`` executes ONE full scenario with ONE seed.
The orchestrator (run_experiments.py) calls run() N times with different seeds.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.drift_detector import detect_drift


# ---------------------------------------------------------------------------
# Scenarios to sweep (each is a dict of kwargs for simulate_run)
# ---------------------------------------------------------------------------

_SCENARIOS: List[Dict[str, Any]] = [
    # ── No drift baselines across batch sizes ──────────────────────────────
    {"drift_type": "none",         "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "no_drift", "batch_size_override": 300},
    {"drift_type": "none",         "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "no_drift_bs150", "batch_size_override": 150},
    {"drift_type": "none",         "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "no_drift_bs600", "batch_size_override": 600},

    # ── Single feature, three sizes, sudden ──────────────────────────────
    {"drift_type": "shift_mean",   "drift_size": "small",  "drift_shape": "sudden",
     "scenario_label": "small_sudden_price", "batch_size_override": 300},
    {"drift_type": "shift_mean",   "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "medium_sudden_price", "batch_size_override": 300},
    {"drift_type": "shift_mean",   "drift_size": "large",  "drift_shape": "sudden",
     "scenario_label": "large_sudden_price", "batch_size_override": 300},

    # ── Single feature, gradual ───────────────────────────────────────────
    {"drift_type": "shift_mean",   "drift_size": "medium", "drift_shape": "gradual",
     "scenario_label": "medium_gradual_price", "batch_size_override": 300},

    # ── Categorical drift ─────────────────────────────────────────────────
    {"drift_type": "shift_category_mix", "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "medium_sudden_region", "batch_size_override": 300},

    # ── Batch-size sensitivity ────────────────────────────────────────────
    {"drift_type": "shift_mean",   "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "medium_sudden_bs150", "batch_size_override": 150},
    {"drift_type": "shift_mean",   "drift_size": "medium", "drift_shape": "sudden",
     "scenario_label": "medium_sudden_bs600", "batch_size_override": 600},
]

# Drift rules to compare
_RULES = ["ks_only", "psi_only", "both"]


def _run_scenario(
    ref_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: Dict[str, Any],
    scenario: Dict[str, Any],
    rule: str,
    seed: int,
) -> Dict[str, Any]:
    """Run one scenario under one detector rule and return a metrics dict."""

    batch_size = scenario.get("batch_size_override", cfg.get("batch_size", 300))

    # Override config's drift_rule for this experiment arm
    local_cfg = {**cfg, "drift_rule": rule, "batch_size": batch_size}

    drift_start = int(cfg.get("drift_start_batch", 16))
    has_drift = scenario["drift_type"] != "none"

    sim = simulate_run(
        pool_df=pool_df,
        config=local_cfg,
        run_id=seed,
        drift_type=scenario["drift_type"],
        drift_size=scenario["drift_size"],
        drift_shape=scenario["drift_shape"],
        random_seed=seed,
    )

    consecutive_flagged = 0
    persistence_needed = int(cfg.get("persistence_required", 2))
    first_alert_batch: Optional[int] = None
    false_alarms = 0
    total_time = 0.0

    for b_idx, batch_df in enumerate(sim.batches):
        t0 = time.perf_counter()
        report = detect_drift(ref_df, batch_df, local_cfg)
        total_time += time.perf_counter() - t0

        flagged = report.get("drift_detected", False)
        consecutive_flagged = consecutive_flagged + 1 if flagged else 0
        confirmed = consecutive_flagged >= persistence_needed

        if confirmed and first_alert_batch is None:
            # Confirmed alert
            if not has_drift or b_idx < drift_start:
                # False alarm (no real drift yet)
                false_alarms += 1
                consecutive_flagged = 0  # reset to count distinct alarms
            else:
                first_alert_batch = b_idx

    detection_rate = 1 if (has_drift and first_alert_batch is not None) else 0
    detection_delay: Optional[int] = None
    if has_drift and first_alert_batch is not None:
        detection_delay = max(0, first_alert_batch - drift_start)

    return {
        "experiment": "A",
        "scenario": scenario["scenario_label"],
        "rule": rule,
        "batch_size": batch_size,
        "has_drift": has_drift,
        "drift_type": scenario["drift_type"],
        "drift_size": scenario["drift_size"],
        "drift_shape": scenario["drift_shape"],
        "seed": seed,
        "detection_rate": detection_rate,
        "false_alarms": false_alarms,
        "detection_delay": detection_delay if detection_delay is not None else np.nan,
        "total_check_time_sec": round(total_time, 4),
    }


def run(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point called by run_experiments.py once per seed.

    Returns a list of metric dicts — one row per (scenario × rule) combination.
    """
    seed = cfg.get("_current_seed", cfg.get("random_seed", 42))

    # Load data (uses cached CSVs if present)
    ref_df, pool_df, _ = load_experiment_data(cfg)

    records: List[Dict[str, Any]] = []
    for scenario in _SCENARIOS:
        for rule in _RULES:
            rec = _run_scenario(ref_df, pool_df, cfg, scenario, rule, seed)
            records.append(rec)

    return records
