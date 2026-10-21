"""
Baseline Model Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Trains, evaluates, and persists the baseline tabular regression model
         on the reference half per PROJECT_SPEC.md Section 4.5 & 4.7.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import sys
# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_config,
    prepare_datasets,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for preprocessing:
    - Numeric features: passed through as-is.
    - Categorical features: one-hot encoded with handle_unknown='ignore' so unseen categories
      in future batches are safely zero-encoded without raising exceptions.
      
    Returns:
        Configured ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_pipeline(config: Dict[str, Any]) -> Pipeline:
    """
    Construct model pipeline composed of preprocessor + regressor.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Unfitted scikit-learn Pipeline.
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "RandomForestRegressor")
    n_estimators = model_cfg.get("n_estimators", 100)
    seed = model_cfg.get("random_state", config.get("random_seed", 42))

    if model_type == "GradientBoostingRegressor":
        regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            random_state=seed
        )
    else:
        # Default: RandomForestRegressor per Section 4.5
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def train_baseline_model(
    config_path: str = "config.yaml",
    save_artifacts: bool = True
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """
    Train baseline model on 80% of reference data and evaluate on 20% held-out reference validation data.
    
    Strictly adheres to:
    - Reference-only training (Section 4.7) to prevent leakage.
    - Sanity check against "predict-the-average" dummy baseline (Section 4.5).
    - Persists model artifact and metrics JSON.
    
    Args:
        config_path: Path to config.yaml.
        save_artifacts: Whether to save model and metrics to models/ directory.
        
    Returns:
        Tuple of (trained_pipeline, metrics_dict, val_dataframe).
    """
    config = load_config(config_path)
    seed = config.get("random_seed", 42)
    train_fraction = config.get("data", {}).get("baseline_train_fraction", 0.8)

    # 1. Load reference half (clean pool is never touched here)
    ref_df, _, _ = prepare_datasets(config_path=config_path, save_to_disk=save_artifacts)

    X = ref_df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)
    y = ref_df[TARGET_COLUMN]

    # 2. Split reference half into 80% train / 20% held-out validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True
    )

    # 3. Fit dummy model (predict average target) for sanity check
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_val_preds = dummy.predict(X_val)
    dummy_mae = float(mean_absolute_error(y_val, dummy_val_preds))

    # 4. Fit baseline regressor pipeline
    pipeline = build_model_pipeline(config)
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on train and validation
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)

    train_mae = float(mean_absolute_error(y_train, train_preds))
    val_mae = float(mean_absolute_error(y_val, val_preds))
    val_rmse = float(root_mean_squared_error(y_val, val_preds))

    improvement_pct = float(((dummy_mae - val_mae) / dummy_mae) * 100.0)

    # 6. Sanity check: must beat dummy by at least 10% (Section 4.5)
    if improvement_pct < 10.0:
        raise ValueError(
            f"Baseline model failed sanity check: improvement over dummy is {improvement_pct:.2f}% "
            f"(required >= 10.0%). Model did not learn a strong enough signal."
        )

    metrics = {
        "model_type": config.get("model", {}).get("type", "RandomForestRegressor"),
        "random_seed": seed,
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET_COLUMN,
        "dummy_val_mae": dummy_mae,
        "baseline_train_mae": train_mae,
        "baseline_val_mae": val_mae,
        "baseline_val_rmse": val_rmse,
        "improvement_over_dummy_pct": improvement_pct,
        "sanity_check_passed": True,
    }

    # Held-out validation dataset containing features and true target
    val_df = ref_df.iloc[X_val.index].copy().reset_index(drop=True)

    # 7. Persist artifacts
    if save_artifacts:
        model_cfg = config.get("model", {})
        model_path = model_cfg.get("model_path", "models/baseline_model.joblib")
        metrics_path = model_cfg.get("metrics_path", "models/baseline_metrics.json")
        ref_val_path = model_cfg.get("ref_val_path", "models/reference_val_data.csv")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(pipeline, model_path)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        val_df.to_csv(ref_val_path, index=False)

    return pipeline, metrics, val_df


def load_baseline_model(model_path: str = "models/baseline_model.joblib") -> Pipeline:
    """Load persisted baseline model pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Train it first.")
    return joblib.load(model_path)


def load_baseline_metrics(metrics_path: str = "models/baseline_metrics.json") -> Dict[str, Any]:
    """Load persisted baseline performance metrics."""
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}. Train model first.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_batch(model: Pipeline, batch_df: pd.DataFrame) -> np.ndarray:
    """
    Generate model predictions for a batch DataFrame.
    
    Args:
        model: Trained pipeline.
        batch_df: DataFrame containing the required feature columns.
        
    Returns:
        1D numpy array of predicted values bounded in [0.0, 1.0].
    """
    # Ensure all required features are present
    missing = [c for c in ALL_FEATURES if c not in batch_df.columns]
    if missing:
        raise KeyError(f"Batch DataFrame missing required feature columns: {missing}")

    X = batch_df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str)

    preds = model.predict(X)
    # Clip predictions to valid target bounds [0.0, 1.0]
    return np.clip(preds, 0.0, 1.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate regression predictions.
    
    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        
    Returns:
        Dictionary of MAE and RMSE.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


def main():
    """CLI runner for baseline model training."""
    print("=" * 80)
    print(" DRIFTMONITOR: BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 80)

    pipeline, metrics, val_df = train_baseline_model("config.yaml", save_artifacts=True)

    print("\n1. TRAINING CONFIGURATION:")
    print(f"   Model Type:                {metrics['model_type']}")
    print(f"   Random Seed:               {metrics['random_seed']}")
    print(f"   Training Rows (80% ref):   {metrics['n_train_samples']:,}")
    print(f"   Held-out Val Rows (20%):   {metrics['n_val_samples']:,}")

    print("\n2. PERFORMANCE COMPARISON & SANITY CHECK (Section 4.5):")
    print(f"   Dummy 'Predict Average' MAE:  {metrics['dummy_val_mae']:.4f} ({metrics['dummy_val_mae'] * 100:.2f} percentage points)")
    print(f"   Baseline Model Val MAE:       {metrics['baseline_val_mae']:.4f} ({metrics['baseline_val_mae'] * 100:.2f} percentage points)")
    print(f"   Baseline Model Val RMSE:      {metrics['baseline_val_rmse']:.4f}")
    print(f"   Relative Improvement:         +{metrics['improvement_over_dummy_pct']:.2f}%")
    print(f"   Sanity Check (>= 10% beat):   {'PASS' if metrics['sanity_check_passed'] else 'FAIL'}")

    print("\n3. SAVED ARTIFACTS:")
    print(f"   Model Pipeline:               models/baseline_model.joblib")
    print(f"   Metrics JSON:                 models/baseline_metrics.json")
    print(f"   Held-out Val Data:            models/reference_val_data.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
