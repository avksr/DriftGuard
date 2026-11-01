"""
src/experiment_d.py

Experiment D – Decision-agent quality.

Per PROJECT_SPEC.md Section 7 / Experiment D:
  Goal: check the agent makes sensible, evidence-based recommendations.

  Four scenarios with a pre-stated expected action:
  +----------------------------------------------------------+-------------------+
  | Scenario                                                 | Expected action   |
  +----------------------------------------------------------+-------------------+
  | Drift, performance still fine (labels available)        | INVESTIGATE       |
  | Drift AND performance degraded                           | PROPOSE_RETRAIN   |
  | Performance worse, no feature drift (concept drift)     | INVESTIGATE       |
  | Drift, labels not yet available                          | REQUEST_LABELS    |
  +----------------------------------------------------------+-------------------+

Stream evaluation:
  Feeds batches sequentially from drift_start up to eval_batch_idx (drift_start + 5)
  to PerformanceMonitor so multi-batch consecutive degradation persistence rules
  evaluate authentically under continuous monitoring.
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


# Default scenario configurations matching PROJECT_SPEC.md Section 7
_DEFAULT_SCENARIOS: List[Dict[str, Any]] = [
    {
        "label": "drift_perf_fine",
        "description": "Drift detected, performance still fine (labels available)",
        "drift_type": "shift_category_mix",
        "drift_size": "large",
        "affected_features": ["Region"],
        "labels_available": True,
        "label_shift_factor": None,
        "expected_action": "INVESTIGATE",
    },
    {
        "label": "drift_and_degraded",
        "description": "Drift AND performance degraded",
        "drift_type": "shift_mean",
        "drift_size": "large",
        "affected_features": ["retail_price"],
        "labels_available": True,
        "label_shift_factor": 1.8,
        "expected_action": "PROPOSE_RETRAIN",
    },
    {
        "label": "perf_worse_no_drift",
        "description": "Performance worse, no feature drift (concept drift / label_shift only)",
        "drift_type": "label_shift",
        "drift_size": "large",
        "affected_features": None,
        "labels_available": True,
        "label_shift_factor": 1.8,
        "expected_action": "INVESTIGATE",
    },
    {
        "label": "drift_labels_hidden",
        "description": "Drift detected, labels not yet available",
        "drift_type": "shift_mean",
        "drift_size": "large",
        "affected_features": ["retail_price"],
        "labels_available": False,
        "label_shift_factor": None,
        "expected_action": "REQUEST_LABELS",
    },
]


def _get_scenarios(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieve scenarios from config or fall back to default scenarios."""
    exp_d_cfg = cfg.get("experiment_d", {})
    configured_scenarios = exp_d_cfg.get("scenarios")
    if not configured_scenarios:
        return _DEFAULT_SCENARIOS

    scenarios: List[Dict[str, Any]] = []
    for label, sc_dict in configured_scenarios.items():
        entry = dict(sc_dict)
        entry["label"] = label
        scenarios.append(entry)
    return scenarios


def _run_scenario(
    ref_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    cfg: Dict[str, Any],
    scenario: Dict[str, Any],
    use_llm: bool,
    seed: int,
) -> Dict[str, Any]:
    """Execute one scenario for one seed streaming batches up to evaluation batch."""
    drift_start = int(cfg.get("drift_start_batch", 16))
    eval_offset = int(cfg.get("experiment_d", {}).get("eval_batch_offset", 5))

    sim = simulate_run(
        pool_df=pool_df,
        config=cfg,
        run_id=seed,
        drift_type=scenario["drift_type"],
        drift_size=scenario["drift_size"],
        drift_shape="sudden",
        affected_features=scenario.get("affected_features"),
        random_seed=seed,
    )

    eval_batch_idx = min(drift_start + eval_offset, len(sim.batches) - 1)

    # Stream batches from drift_start to eval_batch_idx to evaluate performance persistence
    monitor = PerformanceMonitor(config=cfg)
    perf_record = None

    for b_idx in range(drift_start, eval_batch_idx + 1):
        b_df = sim.batches[b_idx].copy()
        if scenario.get("label_shift_factor") is not None and scenario["labels_available"]:
            if TARGET_COLUMN in b_df.columns:
                b_df[TARGET_COLUMN] = (
                    b_df[TARGET_COLUMN] * float(scenario["label_shift_factor"])
                ).clip(upper=1.0)
        perf_record = monitor.evaluate_batch(
            b_df, batch_id=b_idx, labels_available=scenario["labels_available"]
        )

    eval_batch_df = sim.batches[eval_batch_idx]

    # Quality check
    quality_status = check_batch_quality(eval_batch_df, config=cfg, ref_df=ref_df)

    # Drift detection
    drift_report = detect_drift(ref_df, eval_batch_df, cfg, batch_id=eval_batch_idx)
    drift_detected = int(drift_report.get("drift_detected", False))
    top_feat = drift_report.get("ranking", [""])[0] if drift_report.get("ranking") else ""
    top_psi = drift_report.get("features", {}).get(top_feat, {}).get("psi", 0.0)

    perf_dict = {
        "batch_id": eval_batch_idx,
        "status": perf_record.status if perf_record else "unverified",
        "batch_mae": perf_record.batch_mae if perf_record else None,
        "labels_available": scenario["labels_available"],
    }

    # Decision agent
    agent = DecisionAgent(use_llm=use_llm)
    report = agent.decide(
        batch_id=eval_batch_idx,
        quality_status=quality_status,
        drift_report=drift_report,
        performance_record=perf_dict,
    )

    expected = scenario["expected_action"]
    correct = report.chosen_action == expected

    # Premise check verification
    sc_label = scenario["label"]
    if sc_label == "drift_perf_fine":
        premise_ok = bool(drift_detected == 1 and report.performance_status == "normal")
    elif sc_label == "drift_and_degraded":
        premise_ok = bool(drift_detected == 1 and report.performance_status == "degraded")
    elif sc_label == "perf_worse_no_drift":
        premise_ok = bool(drift_detected == 0 and report.performance_status == "degraded")
    elif sc_label == "drift_labels_hidden":
        premise_ok = bool(drift_detected == 1 and not scenario["labels_available"])
    else:
        premise_ok = True

    baseline_mae = monitor.baseline_mae
    current_mae = perf_record.batch_mae if perf_record else None
    mae_ratio = (float(current_mae / baseline_mae)) if (baseline_mae and current_mae is not None) else 1.0

    return {
        "experiment": "D",
        "scenario": scenario["label"],
        "use_llm": int(use_llm),
        "seed": seed,
        "expected_action": expected,
        "chosen_action": report.chosen_action,
        "default_action": report.default_action,
        "correct_action": int(correct),
        "premise_ok": int(premise_ok),
        "unnecessary_retrain": int(
            report.chosen_action == "PROPOSE_RETRAIN"
            and expected != "PROPOSE_RETRAIN"
        ),
        "guardrail_overridden": int(report.guardrail_overridden),
        "used_llm_successfully": int(report.used_llm),
        "drift_detected": int(drift_detected),
        "perf_status": report.performance_status,
        "mae_ratio": round(mae_ratio, 4),
        "top_feature_psi": round(float(top_psi), 4),
    }


def run(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point called by pipeline orchestrator once per seed."""
    seed = cfg.get("_current_seed", cfg.get("random_seed", 42))

    ref_df, pool_df, _ = load_experiment_data(cfg)
    scenarios = _get_scenarios(cfg)

    records: List[Dict[str, Any]] = []

    modes = [False] if cfg.get("rules_only", False) else [False, True]
    for scenario in scenarios:
        for use_llm in modes:
            rec = _run_scenario(ref_df, pool_df, cfg, scenario, use_llm, seed)
            records.append(rec)

    return records
