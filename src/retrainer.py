"""
Controlled Retraining Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements controlled model retraining with dual-window validation acceptance
         per PROJECT_SPEC.md Section 5.8.
         Retraining is triggered ONLY when proposed by the decision agent.
         A candidate model is accepted if and only if:
         1. Candidate MAE is at least 5% lower than current model MAE on the recent validation window.
         2. Candidate MAE on the reference validation set is no more than 10% worse than before
            (catastrophic forgetting guardrail).
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.data_loader import (
    load_config,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN
)
from src.baseline_model import (
    build_preprocessor,
    predict_batch,
    load_baseline_model
)


@dataclass
class RetrainingResult:
    """Summary of a retraining evaluation and acceptance decision."""
    accepted: bool
    decision: str  # "ACCEPTED" or "REJECTED"
    reason: str
    current_recent_mae: float
    candidate_recent_mae: float
    recent_improvement_pct: float
    current_old_mae: float
    candidate_old_mae: float
    old_data_regression_pct: float
    active_model: Pipeline
    candidate_model: Optional[Pipeline] = None
    n_retrain_samples: int = 0
    n_recent_val_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary to dictionary (excluding model objects)."""
        return {
            "accepted": self.accepted,
            "decision": self.decision,
            "reason": self.reason,
            "current_recent_mae": round(self.current_recent_mae, 4),
            "candidate_recent_mae": round(self.candidate_recent_mae, 4),
            "recent_improvement_pct": round(self.recent_improvement_pct, 2),
            "current_old_mae": round(self.current_old_mae, 4),
            "candidate_old_mae": round(self.candidate_old_mae, 4),
            "old_data_regression_pct": round(self.old_data_regression_pct, 2),
            "n_retrain_samples": self.n_retrain_samples,
            "n_recent_val_samples": self.n_recent_val_samples
        }


# -----------------------------------------------------------------------------
# Acceptance Criteria Evaluation (Pure Logic)
# -----------------------------------------------------------------------------

def evaluate_acceptance(
    current_recent_mae: float,
    candidate_recent_mae: float,
    current_old_mae: float,
    candidate_old_mae: float,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, float, float]:
    """
    Evaluate the dual-window acceptance rule per Section 5.8:
    - Rule 1: candidate MAE is >= min_improvement (default 5%) lower than current MAE on recent data.
    - Rule 2: candidate MAE on old reference data is <= max_old_data_regression (default 10%) worse.
    
    Args:
        current_recent_mae: MAE of current active model on recent validation window.
        candidate_recent_mae: MAE of candidate model on recent validation window.
        current_old_mae: MAE of current active model on old reference validation data.
        candidate_old_mae: MAE of candidate model on old reference validation data.
        config: Optional configuration dictionary.
        
    Returns:
        Tuple of (accepted: bool, reason: str, recent_improvement_pct: float, old_data_regression_pct: float).
    """
    if config is None:
        config = load_config()

    retrain_cfg = config.get("retraining", {})
    min_improvement = float(retrain_cfg.get("min_improvement", 0.05))
    max_regression = float(retrain_cfg.get("max_old_data_regression", 0.10))

    # 1. Calculate relative improvement on recent validation window
    # Improvement = (current - candidate) / current
    if current_recent_mae > 0:
        recent_improvement_fraction = (current_recent_mae - candidate_recent_mae) / current_recent_mae
    else:
        recent_improvement_fraction = 0.0
    recent_improvement_pct = recent_improvement_fraction * 100.0

    # 2. Calculate relative regression on old reference validation data
    # Regression = (candidate - current) / current
    if current_old_mae > 0:
        old_regression_fraction = (candidate_old_mae - current_old_mae) / current_old_mae
    else:
        old_regression_fraction = 0.0
    old_regression_pct = old_regression_fraction * 100.0

    # Check criteria
    passes_recent = (recent_improvement_fraction >= min_improvement)
    passes_old = (old_regression_fraction <= max_regression)

    if passes_recent and passes_old:
        accepted = True
        reason = (
            f"Candidate accepted: improved recent data MAE by {recent_improvement_pct:+.2f}% "
            f"(>= {min_improvement * 100:.1f}%) and old data regression is {old_regression_pct:+.2f}% "
            f"(<= {max_regression * 100:.1f}%)."
        )
    elif not passes_recent:
        accepted = False
        reason = (
            f"Candidate rejected: recent improvement {recent_improvement_pct:+.2f}% did not meet "
            f"the required threshold of {min_improvement * 100:.1f}%."
        )
    else:
        # Failed old data forgetting guardrail
        accepted = False
        reason = (
            f"Candidate rejected: catastrophic forgetting detected on old reference data. "
            f"Regression was {old_regression_pct:+.2f}%, exceeding the allowable limit of {max_regression * 100:.1f}%."
        )

    return accepted, reason, recent_improvement_pct, old_regression_pct


# -----------------------------------------------------------------------------
# Candidate Model Training & Retraining Pipeline
# -----------------------------------------------------------------------------

def train_candidate_model(
    train_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Train a candidate regression model using the identical architecture and settings
    as the baseline model (RandomForestRegressor + ColumnTransformer preprocessor).
    
    Args:
        train_df: Training DataFrame containing required feature columns and target.
        config: Optional configuration dictionary.
        
    Returns:
        Fitted candidate Pipeline.
    """
    if config is None:
        config = load_config()

    model_cfg = config.get("model", {})
    n_estimators = int(model_cfg.get("n_estimators", 100))
    random_state = int(model_cfg.get("random_state", 42))

    X_train = train_df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].astype(str)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)

    candidate = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    candidate.fit(X_train, y_train)
    return candidate


def retrain_model(
    current_model: Pipeline,
    ref_train_df: pd.DataFrame,
    ref_val_df: pd.DataFrame,
    recent_batches: List[pd.DataFrame],
    config: Optional[Dict[str, Any]] = None,
    val_batches_count: int = 2,
    reason: str = "Decision Agent PROPOSE_RETRAIN"
) -> RetrainingResult:
    """
    Execute controlled retraining workflow per Section 5.8:
    1. Splits recent labelled batches into new training data and a recent validation window.
    2. Builds candidate training dataset = reference training data + recent batch training data.
    3. Trains candidate model.
    4. Evaluates current and candidate models on the recent validation window.
    5. Evaluates current and candidate models on the old reference validation data.
    6. Applies acceptance rule; updates active model only if accepted.
    
    Args:
        current_model: The currently deployed model Pipeline.
        ref_train_df: The reference half training data.
        ref_val_df: The held-out reference validation set.
        recent_batches: Chronological list of recent labelled DataFrames.
        config: Configuration dictionary.
        val_batches_count: Number of most recent batches reserved for the validation window.
        reason: Justification string for initiating retraining.
        
    Returns:
        RetrainingResult with evaluation metrics and the resulting active model.
    """
    if config is None:
        config = load_config()

    if not recent_batches:
        raise ValueError("Cannot retrain without recent labelled batches.")

    # 1. Partition recent batches chronologically into training data and validation window
    # Validation window: the most recent val_batches_count batches
    # Training data: all earlier batches
    n_recent = len(recent_batches)
    val_count = min(val_batches_count, max(1, n_recent // 2))

    recent_val_dfs = recent_batches[-val_count:]
    recent_train_dfs = recent_batches[:-val_count]

    recent_val_df = pd.concat(recent_val_dfs, ignore_index=True)

    if recent_train_dfs:
        recent_train_df = pd.concat(recent_train_dfs, ignore_index=True)
        combined_train_df = pd.concat([ref_train_df, recent_train_df], ignore_index=True)
    else:
        # If very few batches exist, take an 80/20 train/val split of available recent batches
        shuffled = recent_val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled) * 0.75)
        recent_train_df = shuffled.iloc[:split_idx]
        recent_val_df = shuffled.iloc[split_idx:]
        combined_train_df = pd.concat([ref_train_df, recent_train_df], ignore_index=True)

    # 2. Train candidate model
    candidate_model = train_candidate_model(combined_train_df, config=config)

    # 3. Evaluate current vs candidate on recent validation window
    y_recent_true = recent_val_df[TARGET_COLUMN].to_numpy(dtype=float)
    current_recent_pred = predict_batch(current_model, recent_val_df)
    candidate_recent_pred = predict_batch(candidate_model, recent_val_df)

    current_recent_mae = float(mean_absolute_error(y_recent_true, current_recent_pred))
    candidate_recent_mae = float(mean_absolute_error(y_recent_true, candidate_recent_pred))

    # 4. Evaluate current vs candidate on old reference validation data
    y_old_true = ref_val_df[TARGET_COLUMN].to_numpy(dtype=float)
    current_old_pred = predict_batch(current_model, ref_val_df)
    candidate_old_pred = predict_batch(candidate_model, ref_val_df)

    current_old_mae = float(mean_absolute_error(y_old_true, current_old_pred))
    candidate_old_mae = float(mean_absolute_error(y_old_true, candidate_old_pred))

    # 5. Apply acceptance rule
    accepted, decision_reason, recent_imp_pct, old_reg_pct = evaluate_acceptance(
        current_recent_mae=current_recent_mae,
        candidate_recent_mae=candidate_recent_mae,
        current_old_mae=current_old_mae,
        candidate_old_mae=candidate_old_mae,
        config=config
    )

    # Active model is updated only if candidate is accepted
    active_model = candidate_model if accepted else current_model
    decision_str = "ACCEPTED" if accepted else "REJECTED"

    return RetrainingResult(
        accepted=accepted,
        decision=decision_str,
        reason=f"[{reason}] {decision_reason}",
        current_recent_mae=current_recent_mae,
        candidate_recent_mae=candidate_recent_mae,
        recent_improvement_pct=recent_imp_pct,
        current_old_mae=current_old_mae,
        candidate_old_mae=candidate_old_mae,
        old_data_regression_pct=old_reg_pct,
        active_model=active_model,
        candidate_model=candidate_model,
        n_retrain_samples=len(combined_train_df),
        n_recent_val_samples=len(recent_val_df)
    )


# -----------------------------------------------------------------------------
# Demo Runner (Section 11.3 Phase 7 Requirement)
# -----------------------------------------------------------------------------

def run_phase7_demo() -> None:
    """
    Demonstrates controlled retraining on real project data with drift injection:
    1. Evaluates candidate retraining when drift causes performance degradation.
    2. Verifies the dual-window acceptance rule (5% recent gain + 10% max old regression).
    """
    from src.data_loader import prepare_datasets
    from src.batch_simulator import simulate_run

    print("=" * 85)
    print(" DRIFTMONITOR: PHASE 7 CONTROLLED RETRAINING DEMONSTRATION")
    print("=" * 85)

    config = load_config("config.yaml")

    # 1. Load reference datasets and baseline model
    ref_df, pool_df, _ = prepare_datasets("config.yaml", save_to_disk=False)
    ref_val_path = config.get("model", {}).get("ref_val_path", "models/reference_val_data.csv")
    model_path = config.get("model", {}).get("model_path", "models/baseline_model.joblib")

    if not os.path.exists(ref_val_path) or not os.path.exists(model_path):
        print("Pre-trained baseline model or reference validation split not found.")
        return

    ref_val_df = pd.read_csv(ref_val_path)
    current_model = load_baseline_model(model_path)

    # Reference training set = all ref_df rows not in ref_val_df
    ref_train_indices = [i for i in ref_df.index if i not in ref_val_df.index]
    ref_train_df = ref_df.loc[ref_train_indices].copy() if ref_train_indices else ref_df.copy()

    print(f"Reference Training Data:     {len(ref_train_df):,} samples")
    print(f"Reference Validation Data:   {len(ref_val_df):,} samples (Held-out)")
    print(f"Active Baseline Model:       {model_path}")

    # 2. Simulate 4 labelled incoming batches under large retail_price drift
    print("\nSimulating 4 incoming batches with severe retail_price drift...")
    sim_run = simulate_run(
        pool_df=pool_df,
        config=config,
        run_id="demo_retrain",
        drift_type="shift_mean",
        drift_size="large",
        drift_shape="sudden",
        drift_start_batch=0,
        affected_features=["retail_price"],
        n_batches=4,
        batch_size=300
    )

    # 3. Execute Controlled Retraining
    print("\nExecuting retrain_model() with dual-window validation...")
    result = retrain_model(
        current_model=current_model,
        ref_train_df=ref_train_df,
        ref_val_df=ref_val_df,
        recent_batches=sim_run.batches,
        config=config,
        val_batches_count=2,
        reason="Agent proposed retraining due to sustained error degradation"
    )

    # 4. Display Results
    print("\n" + "-" * 85)
    print(" RETRAINING EVALUATION SUMMARY")
    print("-" * 85)
    print(f"Decision:                     {result.decision} (accepted={result.accepted})")
    print(f"Reason:                       {result.reason}")
    print(f"Retraining Data Size:         {result.n_retrain_samples:,} samples")
    print(f"Recent Validation Window:     {result.n_recent_val_samples:,} samples")
    print("\nRecent Drifted Window Performance:")
    print(f"  - Current Model MAE:        {result.current_recent_mae:.4f}")
    print(f"  - Candidate Model MAE:      {result.candidate_recent_mae:.4f}")
    print(f"  - Relative Improvement:     {result.recent_improvement_pct:+.2f}% (Threshold: >= +5.0%)")
    print("\nOld Reference Data Performance (Forgetting Guardrail):")
    print(f"  - Current Model MAE:        {result.current_old_mae:.4f}")
    print(f"  - Candidate Model MAE:      {result.candidate_old_mae:.4f}")
    print(f"  - Old Data Regression:      {result.old_data_regression_pct:+.2f}% (Threshold: <= +10.0%)")
    print("-" * 85)
    print(f"Active Model Deployed:        {'Candidate Model' if result.accepted else 'Original Baseline Model'}")
    print("=" * 85)
    print(" PHASE 7 DEMONSTRATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_phase7_demo()
