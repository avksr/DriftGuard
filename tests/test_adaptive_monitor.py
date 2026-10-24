"""
Unit tests for adaptive_monitor.py, performance_monitor.py, and incident_logger.py
Verifies monitoring strategies, mode transitions, persistence rules, degradation detection,
and incident logging per PROJECT_SPEC.md Section 5.5, 5.6, 5.9, and Section 9.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.adaptive_monitor import (
    FixedMonitor,
    AdaptiveMonitor,
    compare_strategies,
    CheckRecord
)
from src.performance_monitor import PerformanceMonitor, PerformanceRecord
from src.incident_logger import IncidentLogger, IncidentRecord
from src.data_loader import load_config
from src.batch_simulator import SimulationRun


@pytest.fixture
def mock_config():
    return {
        "persistence_required": 2,
        "adaptive": {
            "normal_interval": 4,
            "alert_interval": 1,
            "clean_checks_to_return": 3
        },
        "performance": {
            "degradation_factor": 1.2,
            "consecutive_batches": 2
        },
        "psi_bins": 10,
        "psi_flag_threshold": 0.25,
        "p_value_threshold": 0.01,
        "drift_rule": "both"
    }


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "retail_price": np.random.uniform(100, 500, size=100),
        "product_rating": np.random.uniform(1, 5, size=100),
        "Region": np.random.choice(["North", "South", "East", "West"], size=100),
        "product_main_category": np.random.choice(["Clothing", "Electronics"], size=100),
        "BrandGroup": np.random.choice(["Brand_A", "Brand_B"], size=100),
        "OrderMonth": np.random.choice([1, 2, 3], size=100),
        "discount_pct": np.random.uniform(0.1, 0.5, size=100)
    })
    return df


# -----------------------------------------------------------------------------
# Tests for FixedMonitor
# -----------------------------------------------------------------------------

def test_fixed_monitor_intervals(mock_config):
    # Fixed(1) should check every batch
    monitor_1 = FixedMonitor(k=1, config=mock_config)
    for b in range(12):
        assert monitor_1.should_check(b) is True
        assert monitor_1.get_current_mode() == "FIXED_1"

    # Fixed(4) should check only multiples of 4
    monitor_4 = FixedMonitor(k=4, config=mock_config)
    for b in range(12):
        expected = (b % 4 == 0)
        assert monitor_4.should_check(b) is expected
        assert monitor_4.get_current_mode() == "FIXED_4"


def test_fixed_monitor_invalid_k():
    with pytest.raises(ValueError):
        FixedMonitor(k=0)


# -----------------------------------------------------------------------------
# Tests for AdaptiveMonitor Transitions & Persistence (Section 9)
# -----------------------------------------------------------------------------

def test_adaptive_monitor_switches_to_alert_after_flagged_check(mock_config, dummy_data):
    monitor = AdaptiveMonitor(config=mock_config)
    assert monitor.get_current_mode() == "NORMAL"

    # Batch 0 is checked (routine normal check)
    with patch("src.adaptive_monitor.detect_drift") as mock_detect:
        # Batch 0 clean
        mock_detect.return_value = {"drift_detected": False, "ranking": []}
        rec0 = monitor.process_batch(0, dummy_data, dummy_data)
        assert rec0.check_executed is True
        assert monitor.get_current_mode() == "NORMAL"

        # Batches 1, 2, 3 skipped
        for b in [1, 2, 3]:
            rec = monitor.process_batch(b, dummy_data, dummy_data)
            assert rec.check_executed is False
            assert monitor.get_current_mode() == "NORMAL"

        # Batch 4: routine check finds drift!
        mock_detect.return_value = {"drift_detected": True, "ranking": ["retail_price"]}
        rec4 = monitor.process_batch(4, dummy_data, dummy_data)
        assert rec4.check_executed is True
        assert rec4.drift_detected is True
        # Must immediately switch to ALERT mode!
        assert monitor.get_current_mode() == "ALERT"
        assert len(monitor.mode_transitions) == 1
        assert monitor.mode_transitions[0].from_mode == "NORMAL"
        assert monitor.mode_transitions[0].to_mode == "ALERT"


def test_adaptive_monitor_returns_to_normal_after_3_clean_checks(mock_config, dummy_data):
    monitor = AdaptiveMonitor(config=mock_config)
    
    with patch("src.adaptive_monitor.detect_drift") as mock_detect:
        # Start: batch 0 flags drift -> enters ALERT
        mock_detect.return_value = {"drift_detected": True, "ranking": ["retail_price"]}
        monitor.process_batch(0, dummy_data, dummy_data)
        assert monitor.get_current_mode() == "ALERT"

        # In ALERT mode, checks happen every batch
        # Clean check 1 at batch 1
        mock_detect.return_value = {"drift_detected": False, "ranking": []}
        monitor.process_batch(1, dummy_data, dummy_data)
        assert monitor.get_current_mode() == "ALERT"

        # Clean check 2 at batch 2
        monitor.process_batch(2, dummy_data, dummy_data)
        assert monitor.get_current_mode() == "ALERT"

        # Clean check 3 at batch 3 -> Should return to NORMAL!
        monitor.process_batch(3, dummy_data, dummy_data)
        assert monitor.get_current_mode() == "NORMAL"
        
        # Verify mode transition history
        assert len(monitor.mode_transitions) == 2
        assert monitor.mode_transitions[1].from_mode == "ALERT"
        assert monitor.mode_transitions[1].to_mode == "NORMAL"


def test_confirmed_alert_requires_two_flagged_checks_in_a_row(mock_config, dummy_data):
    monitor = FixedMonitor(k=1, config=mock_config)

    with patch("src.adaptive_monitor.detect_drift") as mock_detect:
        # Batch 0: Flagged check 1
        mock_detect.return_value = {"drift_detected": True, "ranking": ["retail_price"]}
        rec0 = monitor.process_batch(0, dummy_data, dummy_data)
        assert rec0.drift_detected is True
        assert rec0.consecutive_flagged == 1
        # Persistence required is 2, so NOT confirmed yet!
        assert rec0.confirmed_alert is False
        assert len(monitor.confirmed_alerts) == 0

        # Batch 1: Flagged check 2
        mock_detect.return_value = {"drift_detected": True, "ranking": ["retail_price"]}
        rec1 = monitor.process_batch(1, dummy_data, dummy_data)
        assert rec1.drift_detected is True
        assert rec1.consecutive_flagged == 2
        # Now confirmed alert is triggered!
        assert rec1.confirmed_alert is True
        assert monitor.confirmed_alerts == [1]

        # Batch 2: Clean check -> Resets consecutive counter to 0
        mock_detect.return_value = {"drift_detected": False, "ranking": []}
        rec2 = monitor.process_batch(2, dummy_data, dummy_data)
        assert rec2.drift_detected is False
        assert rec2.consecutive_flagged == 0
        assert rec2.confirmed_alert is False


# -----------------------------------------------------------------------------
# Tests for Strategy Comparison on SimulationRun
# -----------------------------------------------------------------------------

def test_compare_strategies(mock_config, dummy_data):
    # Construct a mini 8-batch simulation run
    batches = [dummy_data.copy() for _ in range(8)]
    gt = [{"is_drifted": False} for _ in range(8)]
    sim_run = SimulationRun(
        run_id="test_run",
        drift_type="none",
        drift_size="none",
        drift_shape="none",
        drift_start_batch=16,
        gradual_ramp_batches=8,
        affected_features=[],
        batches=batches,
        ground_truth_per_batch=gt
    )

    with patch("src.adaptive_monitor.detect_drift") as mock_detect:
        mock_detect.return_value = {"drift_detected": False, "ranking": []}
        results = compare_strategies(dummy_data, sim_run, config=mock_config)

    assert "Fixed_1" in results
    assert "Fixed_4" in results
    assert "Adaptive" in results

    # Fixed_1 must have checked all 8 batches
    assert results["Fixed_1"].checks_performed == 8
    # Fixed_4 must have checked batch 0 and batch 4 (2 checks)
    assert results["Fixed_4"].checks_performed == 2
    # Adaptive in calm mode checked batch 0 and batch 4 (2 checks)
    assert results["Adaptive"].checks_performed == 2


# -----------------------------------------------------------------------------
# Tests for PerformanceMonitor (Section 5.6)
# -----------------------------------------------------------------------------

def test_performance_monitor_labels_unavailable(mock_config, dummy_data):
    perf_mon = PerformanceMonitor(config=mock_config, baseline_mae=0.10)
    
    # Process batch with labels_available=False
    rec = perf_mon.evaluate_batch(dummy_data, batch_id=0, labels_available=False)
    assert rec.labels_available is False
    assert rec.status == "unverified"
    assert rec.batch_mae is None
    assert rec.is_degraded is False


def test_performance_monitor_sustained_degradation(mock_config, dummy_data):
    perf_mon = PerformanceMonitor(config=mock_config, baseline_mae=0.10)
    # Threshold is 0.10 * 1.2 = 0.12

    # Mock predict_batch to produce high error
    with patch("src.performance_monitor.predict_batch") as mock_predict:
        # Batch 0: High error (MAE = 0.20 > 0.12)
        # Dummy target has values around 0.3
        mock_predict.return_value = dummy_data["discount_pct"].to_numpy() + 0.20
        rec0 = perf_mon.evaluate_batch(dummy_data, batch_id=0, labels_available=True)
        assert rec0.batch_mae > 0.12
        assert rec0.consecutive_degraded == 1
        # Requires 2 consecutive batches, so not yet degraded status
        assert rec0.status == "normal"
        assert rec0.is_degraded is False

        # Batch 1: High error again (MAE = 0.20 > 0.12)
        rec1 = perf_mon.evaluate_batch(dummy_data, batch_id=1, labels_available=True)
        assert rec1.consecutive_degraded == 2
        # Now sustained degradation is reached!
        assert rec1.status == "degraded"
        assert rec1.is_degraded is True

        # Batch 2: Good performance (MAE = 0.05 < 0.12)
        mock_predict.return_value = dummy_data["discount_pct"].to_numpy() + 0.05
        rec2 = perf_mon.evaluate_batch(dummy_data, batch_id=2, labels_available=True)
        assert rec2.consecutive_degraded == 0
        assert rec2.status == "normal"
        assert rec2.is_degraded is False


# -----------------------------------------------------------------------------
# Tests for IncidentLogger (Section 5.9)
# -----------------------------------------------------------------------------

def test_incident_logger_lifecycle(tmp_path):
    logger = IncidentLogger(log_dir=str(tmp_path))

    rec = logger.log_incident(
        batch_id=16,
        action="PROPOSE_RETRAIN",
        summary="Drift on retail_price and sustained performance degradation",
        run_id="run_101",
        monitoring_mode="ALERT",
        quality_status="passed",
        drift_detected=True,
        top_drifted_features=["retail_price", "Region"],
        top_feature_psi=0.45,
        performance_status="degraded",
        batch_mae=0.22,
        guardrail_overridden=False
    )

    assert isinstance(rec, IncidentRecord)
    assert len(logger.get_incidents()) == 1

    # Check DataFrame export
    df = logger.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["action"] == "PROPOSE_RETRAIN"
    assert "retail_price" in df.iloc[0]["top_drifted_features"]

    # Test file persistence
    csv_file = logger.save_to_csv()
    json_file = logger.save_to_json()
    assert os.path.exists(csv_file)
    assert os.path.exists(json_file)

    # Test clear
    logger.clear()
    assert len(logger.get_incidents()) == 0
