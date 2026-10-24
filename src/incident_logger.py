"""
Incident Logger Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements structured logging for monitoring incidents, confirmed alerts,
         and remediation decisions per PROJECT_SPEC.md Section 5.9.
         Persists records to CSV and JSON formats to support dashboard presentation
         and experimental analysis.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import json
import os
import pandas as pd


@dataclass
class IncidentRecord:
    """Standardized record of an incident or confirmed alert decision."""
    timestamp: str
    run_id: Union[int, str]
    batch_id: int
    monitoring_mode: str
    quality_status: str
    drift_detected: bool
    top_drifted_features: List[str]
    top_feature_psi: float
    performance_status: str
    batch_mae: Optional[float]
    action: str
    summary: str
    guardrail_overridden: bool = False


class IncidentLogger:
    """
    Collects, manages, and persists incident records.
    """

    def __init__(self, log_dir: str = "results"):
        self.log_dir = log_dir
        self.incidents: List[IncidentRecord] = []
        os.makedirs(self.log_dir, exist_ok=True)

    def log_incident(
        self,
        batch_id: int,
        action: str,
        summary: str,
        run_id: Union[int, str] = "default",
        monitoring_mode: str = "NORMAL",
        quality_status: str = "passed",
        drift_detected: bool = False,
        top_drifted_features: Optional[List[str]] = None,
        top_feature_psi: float = 0.0,
        performance_status: str = "normal",
        batch_mae: Optional[float] = None,
        guardrail_overridden: bool = False,
        timestamp: Optional[str] = None
    ) -> IncidentRecord:
        """
        Record a new incident event.
        
        Args:
            batch_id: Current batch index.
            action: Selected system action (e.g., 'INVESTIGATE', 'PROPOSE_RETRAIN').
            summary: Human-readable incident summary or LLM explanation.
            run_id: Identifier of the simulation run.
            monitoring_mode: Active monitoring policy mode ('NORMAL', 'ALERT', 'FIXED').
            quality_status: Data validation status ('passed', 'failed').
            drift_detected: Whether drift detector flagged batch.
            top_drifted_features: Ranked drifted feature names.
            top_feature_psi: PSI of top ranked drifted feature.
            performance_status: 'normal', 'degraded', or 'unverified'.
            batch_mae: Current batch MAE (None if unverified).
            guardrail_overridden: True if guardrail superseded LLM choice.
            timestamp: Optional ISO timestamp override.
            
        Returns:
            Created IncidentRecord.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        if top_drifted_features is None:
            top_drifted_features = []

        record = IncidentRecord(
            timestamp=timestamp,
            run_id=run_id,
            batch_id=batch_id,
            monitoring_mode=monitoring_mode,
            quality_status=quality_status,
            drift_detected=drift_detected,
            top_drifted_features=top_drifted_features,
            top_feature_psi=round(top_feature_psi, 4),
            performance_status=performance_status,
            batch_mae=round(batch_mae, 4) if batch_mae is not None else None,
            action=action,
            summary=summary,
            guardrail_overridden=guardrail_overridden
        )
        self.incidents.append(record)
        return record

    def get_incidents(self) -> List[Dict[str, Any]]:
        """Return all logged incidents as a list of dictionaries."""
        return [asdict(inc) for inc in self.incidents]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all logged incidents to a pandas DataFrame."""
        if not self.incidents:
            return pd.DataFrame(columns=[
                "timestamp", "run_id", "batch_id", "monitoring_mode",
                "quality_status", "drift_detected", "top_drifted_features",
                "top_feature_psi", "performance_status", "batch_mae",
                "action", "summary", "guardrail_overridden"
            ])
        data = []
        for inc in self.incidents:
            d = asdict(inc)
            # Format list of features as comma-separated string for CSV readability
            d["top_drifted_features"] = ", ".join(d["top_drifted_features"])
            data.append(d)
        return pd.DataFrame(data)

    def save_to_csv(self, filepath: Optional[str] = None) -> str:
        """Save incidents to CSV file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "incidents.csv")
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

    def save_to_json(self, filepath: Optional[str] = None) -> str:
        """Save incidents to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, "incidents.json")
        data = self.get_incidents()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return filepath

    def clear(self) -> None:
        """Clear all logged incidents in memory."""
        self.incidents.clear()
