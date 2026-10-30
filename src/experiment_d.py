"""
src/experiment_d.py

Experiment D – Decision-agent quality.

Per PROJECT_SPEC.md Section 7 / Experiment D:
  Goal: check the agent makes sensible, evidence-based recommendations.

  Four scenarios with a pre-stated expected action:
  +----------------------------------------------------------+----------------------------------------------+-------------------+
  | Scenario                                                 | How it is created                            | Expected action   |
  +----------------------------------------------------------+----------------------------------------------+-------------------+
  | Drift, performance still fine (labels available)        | small shift on a weakly-used feature (rating)| INVESTIGATE       |
  | Drift AND performance degraded                           | large combined drift + label_shift           | PROPOSE_RETRAIN   |
  | Performance worse, no feature drift (concept drift)     | label_shift only (target multiplied)         | INVESTIGATE       |
  | Drift, labels not yet available                          | medium drift, labels hidden                  | REQUEST_LABELS    |
  +----------------------------------------------------------+----------------------------------------------+-------------------+

  Metrics:
    - Share of decisions matching expected action
    - Number of unnecessary PROPOSE_RETRAIN recommendations
    - Number of times guardrail overrode the LLM
    - Number of invalid/fallback LLM replies
    - Whether retraining acceptance rule correctly rejects a bad candidate

  Runs LLM version (use_llm=True) AND rules-only version (use_llm=False) side by side.
  Uses a small number of repeats per scenario (10 by default) as LLM calls are slow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_loader import TARGET_COLUMN
from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.drift_detector import detect_drift
from src.quality_checks import check_batch_quality
from src.performance_monitor import PerformanceMonitor
from src.decision_agent import DecisionAgent


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

_SCENARIOS: List[Dict[str, Any]] = [
    {
        "label": "drift_perf_fine",
        "description": "Drift detected, performance still fine (labels available)",
        "drift_type": "shift_additive",   # small shift on rating – weakly used feature
        "drift_size": "small",
        "drift_shape": "sudden",
        "labels_available": True,
        "label_shift_factor": None,       # no target shift → MAE stays near baseline
        "expected_action": "INVESTIGATE",
    },
    {
        "label": "drift_and_degraded",
        "description": "Drift AND performance degraded",
        "drift_type": "shift_mean",       # large price shift → large MAE jump
        "drift_size": "large",
        "drift_shape": "sudden",
        "labels_available": True,
        "label_shift_factor": 1.8,        # also shift the target to force degradation
        "expected_action": "PROPOSE_RETRAIN",
    },
    {
        "label": "perf_worse_no_drift",
        "description": "Performance worse, no feature drift (concept drift / label_shift only)",
        "drift_type": "label_shift",
        "drift_size": "large",
        "drift_shape": "sudden",
        "labels_available": True,
        "label_shift_factor": None,       # handled by drift_type=label_shift
        "expected_action": "INVESTIGATE",
    },
    {
        "label": "drift_labels_hidden",
        "description": "Drift detected, labels not yet available",
        "drift_type": "shift_mean",
        "drift_size": "medium",
        "drift_shape": "sudden",
        "labels_available": False,        # simulate delayed labels
        "label_shift_factor": None,
        "expected_action": "REQUEST_LABELS",
    },
]


def _run_scenario(
    ref_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: Dict[str, Any],
    scenario: Dict[str, Any],
    use_llm: bool,
    seed: int,
) -> Dict[str, Any]:
    """Execute one scenario for one seed and return a metrics dict."""

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

    # Use the first clearly-drifted batch for the quality/drift/performance check
    target_idx = min(drift_start + 2, len(sim.batches) - 1)
    batch_df = sim.batches[target_idx]

    # Quality check
    quality_status = check_batch_quality(batch_df, config=cfg, ref_df=ref_df)

    # Drift detection
    drift_report = detect_drift(ref_df, batch_df, cfg, batch_id=target_idx)

    # Performance evaluation
    monitor = PerformanceMonitor(config=cfg)

    if scenario["labels_available"]:
        # Apply label shift (concept drift arm) if requested
        if scenario.get("label_shift_factor") is not None:
            eval_batch = batch_df.copy()
            if TARGET_COLUMN in eval_batch.columns:
                eval_batch[TARGET_COLUMN] = (
                    eval_batch[TARGET_COLUMN] * scenario["label_shift_factor"]
                ).clip(upper=1.0)
        else:
            eval_batch = batch_df
        perf_record = monitor.evaluate_batch(
            eval_batch, batch_id=target_idx, labels_available=True
        )
    else:
        perf_record = monitor.evaluate_batch(
            batch_df, batch_id=target_idx, labels_available=False
        )

    perf_dict = {
        "batch_id": target_idx,
        "status": perf_record.status,
        "batch_mae": perf_record.batch_mae,
        "labels_available": scenario["labels_available"],
    }

    # Decision agent
    agent = DecisionAgent(use_llm=use_llm)
    report = agent.decide(
        batch_id=target_idx,
        quality_status=quality_status,
        drift_report=drift_report,
        performance_record=perf_dict,
    )

    expected = scenario["expected_action"]
    correct = report.chosen_action == expected

    return {
        "experiment": "D",
        "scenario": scenario["label"],
        "use_llm": int(use_llm),
        "seed": seed,
        "expected_action": expected,
        "chosen_action": report.chosen_action,
        "default_action": report.default_action,
        "correct_action": int(correct),
        "unnecessary_retrain": int(
            report.chosen_action == "PROPOSE_RETRAIN"
            and expected != "PROPOSE_RETRAIN"
        ),
        "guardrail_overridden": int(report.guardrail_overridden),
        "used_llm_successfully": int(report.used_llm),
        "drift_detected": int(report.drift_detected),
        "perf_status": report.performance_status,
    }


def run(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point called by run_experiments.py once per seed.

    Runs all four scenarios under both LLM=True and LLM=False.
    """
    seed = cfg.get("_current_seed", cfg.get("random_seed", 42))

    ref_df, pool_df, _ = load_experiment_data(cfg)

    records: List[Dict[str, Any]] = []

    for scenario in _SCENARIOS:
        for use_llm in [False, True]:
            rec = _run_scenario(ref_df, pool_df, cfg, scenario, use_llm, seed)
            records.append(rec)

    return records
