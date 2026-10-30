"""
Smoke test suite for DriftGuard.
Verifies that all modules in src/ and experiments/ can be imported cleanly,
and executes micro-runs of Experiment A, B, C, and D with tiny configurations
to guarantee no import errors, broken signatures, or runtime crashes.
"""

import pytest
import pandas as pd


def test_import_all_src_and_experiment_modules():
    """Verify that every module imports without error."""
    import src.data_loader
    import src.data_audit
    import src.baseline_model
    import src.quality_checks
    import src.drift_detector
    import src.batch_simulator
    import src.adaptive_monitor
    import src.performance_monitor
    import src.decision_agent
    import src.retrainer
    import src.incident_logger
    import src._exp_utils
    import src.experiment_a
    import src.experiment_b
    import src.experiment_c
    import src.experiment_d
    import experiments.run_experiments
    import experiments.run_all

    # Confirm key classes exist
    assert hasattr(src.decision_agent, "DecisionAgent")
    assert hasattr(src.adaptive_monitor, "AdaptiveMonitor")
    assert hasattr(src.adaptive_monitor, "FixedMonitor")
    assert hasattr(src.performance_monitor, "PerformanceMonitor")
    assert hasattr(src.retrainer, "retrain_model")


def test_smoke_micro_experiment_a():
    """Run a fast micro-execution of experiment A with tiny config."""
    from src.drift_detector import load_config
    import src.experiment_a as exp_a

    cfg = load_config("config.yaml")
    cfg = dict(cfg)
    cfg["_current_seed"] = 42
    cfg["n_batches_per_run"] = 4
    cfg["batch_size"] = 50
    cfg["drift_start_batch"] = 2
    cfg["gradual_ramp_batches"] = 2

    # Override scenarios to only 1 quick scenario for smoke testing
    orig_scenarios = exp_a._SCENARIOS
    try:
        exp_a._SCENARIOS = [
            {"drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
             "scenario_label": "smoke_price", "batch_size_override": 50}
        ]
        results = exp_a.run(cfg)
        assert isinstance(results, list)
        assert len(results) == len(exp_a._RULES)
        assert "detection_rate" in results[0]
    finally:
        exp_a._SCENARIOS = orig_scenarios


def test_smoke_micro_experiment_b():
    """Run a fast micro-execution of experiment B with tiny config."""
    from src.drift_detector import load_config
    import src.experiment_b as exp_b

    cfg = load_config("config.yaml")
    cfg = dict(cfg)
    cfg["_current_seed"] = 42
    cfg["n_batches_per_run"] = 4
    cfg["batch_size"] = 50
    cfg["drift_start_batch"] = 2
    cfg["gradual_ramp_batches"] = 2

    orig_scenarios = exp_b._SCENARIOS
    try:
        exp_b._SCENARIOS = [
            {"drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
             "label": "smoke_medium_sudden"}
        ]
        results = exp_b.run(cfg)
        assert isinstance(results, list)
        assert len(results) == 3  # Fixed_1, Fixed_4, Adaptive
        assert "detection_delay" in results[0]
    finally:
        exp_b._SCENARIOS = orig_scenarios


def test_smoke_micro_experiment_c():
    """Run a fast micro-execution of experiment C with tiny config."""
    from src.drift_detector import load_config
    import src.experiment_c as exp_c

    cfg = load_config("config.yaml")
    cfg = dict(cfg)
    cfg["_current_seed"] = 42
    cfg["n_batches_per_run"] = 4
    cfg["batch_size"] = 50
    cfg["drift_start_batch"] = 2
    cfg["gradual_ramp_batches"] = 2

    orig_scenarios = exp_c._SCENARIOS
    try:
        exp_c._SCENARIOS = [
            {"changed": ["retail_price"],
             "drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
             "label": "smoke_1feat_price"}
        ]
        results = exp_c.run(cfg)
        assert isinstance(results, list)
        assert len(results) == 1
        assert "precision_at_k" in results[0]
    finally:
        exp_c._SCENARIOS = orig_scenarios


def test_smoke_micro_experiment_d():
    """Run a fast micro-execution of experiment D (rules-only) with tiny config."""
    from src.drift_detector import load_config
    import src.experiment_d as exp_d

    cfg = load_config("config.yaml")
    cfg = dict(cfg)
    cfg["_current_seed"] = 42
    cfg["n_batches_per_run"] = 4
    cfg["batch_size"] = 50
    cfg["drift_start_batch"] = 2
    cfg["gradual_ramp_batches"] = 2

    orig_scenarios = exp_d._SCENARIOS
    try:
        exp_d._SCENARIOS = [
            {
                "label": "smoke_drift_perf_fine",
                "description": "Drift detected, performance fine",
                "drift_type": "shift_additive",
                "drift_size": "small",
                "drift_shape": "sudden",
                "labels_available": True,
                "label_shift_factor": None,
                "expected_action": "INVESTIGATE",
            }
        ]
        # Only run rules-only (use_llm=False)
        rec = exp_d._run_scenario(
            ref_df=pd.read_csv("data/reference_data.csv").head(100),
            pool_df=pd.read_csv("data/clean_pool.csv").head(100),
            cfg=cfg,
            scenario=exp_d._SCENARIOS[0],
            use_llm=False,
            seed=42
        )
        assert rec["correct_action"] == 1
        assert rec["chosen_action"] == "INVESTIGATE"
    finally:
        exp_d._SCENARIOS = orig_scenarios

