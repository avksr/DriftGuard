"""
Performance Monitor Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements batch-level model performance tracking and degradation detection
         per PROJECT_SPEC.md Section 5.6.
         Calculates Mean Absolute Error (MAE) on incoming labelled batches,
         detects sustained performance degradation against the reference baseline
         (MAE > degradation_factor * baseline_MAE for consecutive batches),
         and gracefully handles unverified batches when ground truth labels are delayed.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import json
import os
import logging
import yaml
import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.data_loader import load_config, TARGET_COLUMN
from src.baseline_model import predict_batch, load_baseline_metrics, load_baseline_model


@dataclass
class PerformanceRecord:
    """Record of performance evaluation for a single batch."""
    batch_id: int
    labels_available: bool
    batch_mae: Optional[float] = None
    batch_rmse: Optional[float] = None
    baseline_mae: float = 0.0
    degradation_threshold: float = 0.0
    ratio_to_baseline: Optional[float] = None
    status: str = "unverified"  # "normal", "degraded", or "unverified"
    consecutive_degraded: int = 0
    is_degraded: bool = False


class PerformanceMonitor:
    """
    Monitors model performance (MAE) across streaming batches.
    
    Attributes:
        baseline_mae: Held-out validation MAE from reference data.
        degradation_factor: Multiplier over baseline MAE considered degraded (e.g., 1.2 = +20% error).
        consecutive_required: Number of consecutive degraded batches required to declare sustained degradation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        baseline_mae: Optional[float] = None,
        model: Optional[Pipeline] = None
    ):
        if config is None:
            config = load_config()
        self.config = config

        perf_cfg = self.config.get("performance", {})
        self.degradation_factor = float(perf_cfg.get("degradation_factor", 1.2))
        self.consecutive_required = int(perf_cfg.get("consecutive_batches", 2))

        # 1. Establish baseline MAE
        if baseline_mae is not None:
            self.baseline_mae = float(baseline_mae)
        else:
            self.baseline_mae = self._resolve_baseline_mae()

        self.degradation_threshold = round(self.baseline_mae * self.degradation_factor, 6)

        # 2. Store or load model
        self.model = model

        # State tracking
        self.consecutive_degraded: int = 0
        self.history: List[PerformanceRecord] = []

    def _resolve_baseline_mae(self) -> float:
        """Attempt to load baseline MAE from saved metrics file, fallback to 0.15."""
        metrics_path = self.config.get("model", {}).get("metrics_path", "models/baseline_metrics.json")
        if os.path.exists(metrics_path):
            try:
                metrics = load_baseline_metrics(metrics_path)
                return float(metrics.get("baseline_val_mae", 0.15))
            except Exception as exc:
                logger.warning(f"Failed to load baseline metrics from {metrics_path}: {exc}")
        return 0.15

    def reset(self) -> None:
        """Reset monitor state for a new simulation run."""
        self.consecutive_degraded = 0
        self.history = []

    def evaluate_batch(
        self,
        batch_df: pd.DataFrame,
        batch_id: int,
        labels_available: bool = True,
        model: Optional[Pipeline] = None
    ) -> PerformanceRecord:
        """
        Evaluate model performance on an incoming batch.
        
        Args:
            batch_df: Incoming batch DataFrame.
            batch_id: Batch index.
            labels_available: Boolean flag simulating whether true labels have arrived.
            model: Optional model pipeline to override default.
            
        Returns:
            PerformanceRecord containing MAE, status ('normal', 'degraded', 'unverified'),
            and consecutive degradation count.
        """
        active_model = model if model is not None else self.model
        if active_model is None and labels_available:
            try:
                model_path = self.config.get("model", {}).get("model_path", "models/baseline_model.joblib")
                active_model = load_baseline_model(model_path)
                self.model = active_model
            except Exception as exc:
                logger.warning(f"Failed to load baseline model from {model_path}: {exc}")

        # Case 1: Labels unavailable (ground truth not yet arrived in production)
        if not labels_available or TARGET_COLUMN not in batch_df.columns:
            record = PerformanceRecord(
                batch_id=batch_id,
                labels_available=False,
                batch_mae=None,
                batch_rmse=None,
                baseline_mae=self.baseline_mae,
                degradation_threshold=self.degradation_threshold,
                ratio_to_baseline=None,
                status="unverified",
                consecutive_degraded=self.consecutive_degraded,
                is_degraded=(self.consecutive_degraded >= self.consecutive_required)
            )
            self.history.append(record)
            return record

        # Case 2: Labels available - compute predictions and MAE
        y_true = batch_df[TARGET_COLUMN].to_numpy(dtype=float)
        
        if active_model is not None:
            y_pred = predict_batch(active_model, batch_df)
        else:
            # Fallback if no model pipeline provided: dummy predictor
            y_pred = np.full_like(y_true, np.mean(y_true))

        batch_mae = float(mean_absolute_error(y_true, y_pred))
        batch_rmse = float(root_mean_squared_error(y_true, y_pred))
        ratio = batch_mae / self.baseline_mae if self.baseline_mae > 0 else 1.0

        # Check single-batch threshold
        if batch_mae > self.degradation_threshold:
            self.consecutive_degraded += 1
        else:
            self.consecutive_degraded = 0

        # Sustained degradation check
        is_sustained_degradation = (self.consecutive_degraded >= self.consecutive_required)
        status = "degraded" if is_sustained_degradation else "normal"

        record = PerformanceRecord(
            batch_id=batch_id,
            labels_available=True,
            batch_mae=round(batch_mae, 4),
            batch_rmse=round(batch_rmse, 4),
            baseline_mae=round(self.baseline_mae, 4),
            degradation_threshold=round(self.degradation_threshold, 4),
            ratio_to_baseline=round(ratio, 4),
            status=status,
            consecutive_degraded=self.consecutive_degraded,
            is_degraded=is_sustained_degradation
        )
        self.history.append(record)
        return record
