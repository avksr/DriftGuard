"""
Adaptive and Fixed Monitoring Strategies Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements Fixed(k=1), Fixed(k=4), and Adaptive monitoring strategies
         per PROJECT_SPEC.md Section 5.5.
         Coordinates drift detection on streaming batches, enforces persistence
         rules (two consecutive flagged checks for confirmed alerts), tracks mode
         transitions (NORMAL <-> ALERT), and logs check overhead and latencies.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import time
import os
import yaml
import numpy as np
import pandas as pd

from src.drift_detector import detect_drift, load_config
from src.batch_simulator import SimulationRun


@dataclass
class CheckRecord:
    """Detailed record of a single batch monitoring step."""
    batch_id: int
    mode: str
    check_executed: bool
    drift_detected: Optional[bool] = None
    ranking: List[str] = field(default_factory=list)
    top_feature: Optional[str] = None
    top_feature_psi: float = 0.0
    check_duration_sec: float = 0.0
    consecutive_flagged: int = 0
    confirmed_alert: bool = False


@dataclass
class ModeTransition:
    """Record of an adaptive mode switch."""
    batch_id: int
    from_mode: str
    to_mode: str
    reason: str


@dataclass
class MonitoringResult:
    """Summary of monitoring performance over a complete simulation run."""
    strategy_name: str
    run_id: Union[int, str]
    total_batches: int
    checks_performed: int
    checks_timeline: List[CheckRecord] = field(repr=False)
    mode_transitions: List[ModeTransition] = field(default_factory=list)
    first_confirmed_alert_batch: Optional[int] = None
    detection_delay: Optional[int] = None
    missed_detection: bool = False
    false_alarms: int = 0
    total_check_time_sec: float = 0.0

    @property
    def check_rate_pct(self) -> float:
        """Percentage of batches that underwent full drift checks."""
        if self.total_batches == 0:
            return 0.0
        return (self.checks_performed / self.total_batches) * 100.0


class BaseMonitor:
    """
    Abstract base class for monitoring strategies.
    Enforces identical persistence rules and metric collection across all strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.persistence_required = int(config.get("persistence_required", 2))
        self.consecutive_flagged: int = 0
        self.confirmed_alerts: List[int] = []
        self.checks_performed: int = 0
        self.total_check_time_sec: float = 0.0
        self.checks_timeline: List[CheckRecord] = []
        self.mode_transitions: List[ModeTransition] = []

    def reset(self) -> None:
        """Reset internal monitor state before starting a new run."""
        self.consecutive_flagged = 0
        self.confirmed_alerts = []
        self.checks_performed = 0
        self.total_check_time_sec = 0.0
        self.checks_timeline = []
        self.mode_transitions = []

    def should_check(self, batch_id: int) -> bool:
        """Determine whether the current batch requires a full drift check."""
        raise NotImplementedError

    def get_current_mode(self) -> str:
        """Return the current operational mode ('NORMAL', 'ALERT', 'FIXED')."""
        raise NotImplementedError

    def on_check_completed(self, batch_id: int, drift_detected: bool) -> None:
        """Hook called after a check is executed to update internal state/modes."""
        pass

    def process_batch(
        self,
        batch_id: int,
        ref_df: pd.DataFrame,
        batch_df: pd.DataFrame
    ) -> CheckRecord:
        """
        Process a single incoming batch according to the monitoring policy.
        
        Args:
            batch_id: 0-indexed batch counter.
            ref_df: Reference baseline DataFrame.
            batch_df: Current incoming batch DataFrame.
            
        Returns:
            CheckRecord describing the monitoring decision and detection findings.
        """
        mode = self.get_current_mode()
        do_check = self.should_check(batch_id)

        if not do_check:
            # Check skipped to save compute
            record = CheckRecord(
                batch_id=batch_id,
                mode=mode,
                check_executed=False,
                drift_detected=None,
                ranking=[],
                top_feature=None,
                check_duration_sec=0.0,
                consecutive_flagged=self.consecutive_flagged,
                confirmed_alert=False
            )
            self.checks_timeline.append(record)
            return record

        # Execute full drift detection check (the expensive operation)
        start_time = time.perf_counter()
        report = detect_drift(ref_df, batch_df, config=self.config, batch_id=batch_id)
        duration = time.perf_counter() - start_time

        self.checks_performed += 1
        self.total_check_time_sec += duration

        drift_detected = bool(report.get("drift_detected", False))
        ranking = report.get("ranking", [])
        top_feature = ranking[0] if ranking else None
        top_feature_psi = 0.0
        if top_feature and "features" in report and top_feature in report["features"]:
            top_feature_psi = float(report["features"][top_feature].get("psi", 0.0))

        # Update persistence counter (two consecutive flagged checks trigger confirmed alert)
        if drift_detected:
            self.consecutive_flagged += 1
        else:
            self.consecutive_flagged = 0

        # Confirmed alert triggered when persistence threshold is reached
        is_confirmed = (self.consecutive_flagged >= self.persistence_required)
        if is_confirmed and batch_id not in self.confirmed_alerts:
            self.confirmed_alerts.append(batch_id)

        # Notify strategy subclass of check outcome for mode management
        self.on_check_completed(batch_id, drift_detected)

        record = CheckRecord(
            batch_id=batch_id,
            mode=mode,
            check_executed=True,
            drift_detected=drift_detected,
            ranking=ranking,
            top_feature=top_feature,
            top_feature_psi=top_feature_psi,
            check_duration_sec=duration,
            consecutive_flagged=self.consecutive_flagged,
            confirmed_alert=is_confirmed
        )
        self.checks_timeline.append(record)
        return record

    def run_simulation(
        self,
        ref_df: pd.DataFrame,
        sim_run: SimulationRun
    ) -> MonitoringResult:
        """
        Run this monitoring policy across all batches of a SimulationRun.
        
        Args:
            ref_df: Clean reference dataset.
            sim_run: SimulationRun object containing batches and ground truth.
            
        Returns:
            MonitoringResult with delays, checks count, and alert evaluation.
        """
        self.reset()
        total_batches = len(sim_run.batches)

        for b_idx, batch in enumerate(sim_run.batches):
            self.process_batch(b_idx, ref_df, batch)

        first_alert = self.confirmed_alerts[0] if self.confirmed_alerts else None

        # Calculate detection delay and false alarms against ground truth
        detection_delay: Optional[int] = None
        missed_detection: bool = False
        false_alarms: int = 0

        if sim_run.has_drift:
            drift_start = sim_run.drift_start_batch
            # Find alerts that occurred at or after drift start
            valid_alerts = [a for a in self.confirmed_alerts if a >= drift_start]
            early_alerts = [a for a in self.confirmed_alerts if a < drift_start]
            false_alarms = len(early_alerts)

            if valid_alerts:
                detection_delay = valid_alerts[0] - drift_start
                missed_detection = False
            else:
                detection_delay = None
                missed_detection = True
        else:
            # For no-drift runs, every confirmed alert is a false alarm
            false_alarms = len(self.confirmed_alerts)
            detection_delay = None
            missed_detection = False

        return MonitoringResult(
            strategy_name=self.__class__.__name__,
            run_id=sim_run.run_id,
            total_batches=total_batches,
            checks_performed=self.checks_performed,
            checks_timeline=self.checks_timeline,
            mode_transitions=self.mode_transitions,
            first_confirmed_alert_batch=first_alert,
            detection_delay=detection_delay,
            missed_detection=missed_detection,
            false_alarms=false_alarms,
            total_check_time_sec=self.total_check_time_sec
        )


class FixedMonitor(BaseMonitor):
    """
    Fixed-schedule monitoring policy.
    Checks data strictly every k-th batch (e.g. k=1 for every batch, k=4 for every 4th batch).
    """

    def __init__(self, k: int = 1, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        if k < 1:
            raise ValueError(f"Interval k must be >= 1, got {k}")
        self.k = k

    def should_check(self, batch_id: int) -> bool:
        """Check if batch_id is a multiple of k."""
        return (batch_id % self.k) == 0

    def get_current_mode(self) -> str:
        return f"FIXED_{self.k}"


class AdaptiveMonitor(BaseMonitor):
    """
    Adaptive monitoring policy per Section 3.7 & 5.5:
    - NORMAL mode: checks every normal_interval batches (default 4).
    - ALERT mode: checks every alert_interval batches (default 1).
    - Mode switch NORMAL -> ALERT: triggered immediately upon any flagged drift check.
    - Mode switch ALERT -> NORMAL: triggered after clean_checks_to_return consecutive clean checks (default 3).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config)
        adaptive_cfg = self.config.get("adaptive", {})
        self.normal_interval = int(adaptive_cfg.get("normal_interval", 4))
        self.alert_interval = int(adaptive_cfg.get("alert_interval", 1))
        self.clean_checks_to_return = int(adaptive_cfg.get("clean_checks_to_return", 3))

        self.mode = "NORMAL"
        self.batches_since_last_check = 0
        self.consecutive_clean_in_alert = 0

    def reset(self) -> None:
        super().reset()
        self.mode = "NORMAL"
        self.batches_since_last_check = 0
        self.consecutive_clean_in_alert = 0

    def get_current_mode(self) -> str:
        return self.mode

    def should_check(self, batch_id: int) -> bool:
        """
        Decide whether to execute full drift check based on active mode and elapsed batches.
        Always checks the very first batch (batch 0) to establish reference alignment.
        """
        if batch_id == 0:
            return True

        if self.mode == "ALERT":
            # In ALERT mode: check according to alert_interval (default 1 = every batch)
            if self.alert_interval == 1:
                return True
            return (self.batches_since_last_check >= self.alert_interval)

        # In NORMAL mode: check according to normal_interval (default 4 = every 4th batch)
        return (self.batches_since_last_check >= self.normal_interval)

    def on_check_completed(self, batch_id: int, drift_detected: bool) -> None:
        """
        Update adaptive state machine upon check completion.
        Manages transitions between NORMAL and ALERT modes.
        """
        self.batches_since_last_check = 0

        if self.mode == "NORMAL":
            if drift_detected:
                # Drift signal detected in NORMAL mode: escalate immediately to ALERT mode
                self.mode = "ALERT"
                self.consecutive_clean_in_alert = 0
                transition = ModeTransition(
                    batch_id=batch_id,
                    from_mode="NORMAL",
                    to_mode="ALERT",
                    reason="Drift detected during routine check; escalating to full surveillance"
                )
                self.mode_transitions.append(transition)

        elif self.mode == "ALERT":
            if not drift_detected:
                self.consecutive_clean_in_alert += 1
                if self.consecutive_clean_in_alert >= self.clean_checks_to_return:
                    # System calmed down: return to NORMAL mode
                    self.mode = "NORMAL"
                    self.consecutive_clean_in_alert = 0
                    transition = ModeTransition(
                        batch_id=batch_id,
                        from_mode="ALERT",
                        to_mode="NORMAL",
                        reason=f"Observed {self.clean_checks_to_return} consecutive clean checks; de-escalating"
                    )
                    self.mode_transitions.append(transition)
            else:
                # Still drifting in ALERT mode: reset clean counter
                self.consecutive_clean_in_alert = 0

    def process_batch(
        self,
        batch_id: int,
        ref_df: pd.DataFrame,
        batch_df: pd.DataFrame
    ) -> CheckRecord:
        """
        Increment batches_since_last_check and process the batch.
        """
        self.batches_since_last_check += 1
        return super().process_batch(batch_id, ref_df, batch_df)


# -----------------------------------------------------------------------------
# Multi-Strategy Comparison Utility
# -----------------------------------------------------------------------------

def compare_strategies(
    ref_df: pd.DataFrame,
    sim_run: SimulationRun,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, MonitoringResult]:
    """
    Execute Fixed(k=1), Fixed(k=4), and Adaptive on the exact same SimulationRun.
    
    Args:
        ref_df: Reference baseline DataFrame.
        sim_run: SimulationRun containing batches and ground truth.
        config: Optional configuration dictionary.
        
    Returns:
        Dictionary mapping strategy names to their respective MonitoringResult objects.
    """
    if config is None:
        config = load_config()

    strategies = {
        "Fixed_1": FixedMonitor(k=1, config=config),
        "Fixed_4": FixedMonitor(k=4, config=config),
        "Adaptive": AdaptiveMonitor(config=config)
    }

    results: Dict[str, MonitoringResult] = {}
    for name, monitor in strategies.items():
        results[name] = monitor.run_simulation(ref_df, sim_run)

    return results
