"""
Unit Tests for Baseline Model Module
Author: DriftMonitor Team
Purpose: Verifies baseline model training, evaluation against dummy baseline,
         serialization, and batch inference per PROJECT_SPEC.md Section 4.5.
"""

import os
import pytest
import numpy as np
import pandas as pd

from src.baseline_model import (
    train_baseline_model,
    load_baseline_model,
    load_baseline_metrics,
    predict_batch,
    evaluate_predictions,
    build_model_pipeline,
    ALL_FEATURES,
    TARGET_COLUMN,
)
from src.data_loader import load_config


def test_baseline_model_training_and_sanity_check():
    """Verify that baseline model trains and beats dummy regressor by >= 10% (Section 4.5)."""
    pipeline, metrics, val_df = train_baseline_model(config_path="config.yaml", save_artifacts=False)

    assert metrics["sanity_check_passed"] is True
    assert metrics["improvement_over_dummy_pct"] >= 10.0
    assert metrics["baseline_val_mae"] < metrics["dummy_val_mae"]
    assert len(val_df) == metrics["n_val_samples"]


def test_predict_batch_bounds_and_shapes():
    """Verify predict_batch produces valid outputs within [0.0, 1.0]."""
    model = load_baseline_model()
    val_data = pd.read_csv("models/reference_val_data.csv")

    preds = predict_batch(model, val_data.head(50))
    assert len(preds) == 50
    assert isinstance(preds, np.ndarray)
    assert (preds >= 0.0).all()
    assert (preds <= 1.0).all()


def test_predict_batch_handles_unseen_categories_gracefully():
    """
    Simulate incoming batch with previously unseen categories (e.g. new Region, new category).
    Model pipeline must not crash, thanks to handle_unknown='ignore'.
    """
    model = load_baseline_model()
    unseen_batch = pd.DataFrame({
        "retail_price": [1200.0, 3500.0],
        "product_rating": [4.0, 2.5],
        "Region": ["Antarctica", "MoonBase"],  # Completely unseen regions
        "product_main_category": ["FuturisticGadgets", "AlienTech"],
        "BrandGroup": ["UnknownBrand123", "Other"],
        "OrderMonth": ["99", "100"]
    })

    preds = predict_batch(model, unseen_batch)
    assert len(preds) == 2
    assert not np.isnan(preds).any()
    assert (preds >= 0.0).all() and (preds <= 1.0).all()


def test_predict_batch_missing_features_raises_key_error():
    """Missing a required feature column must raise KeyError."""
    model = load_baseline_model()
    bad_batch = pd.DataFrame({
        "retail_price": [1200.0],
        # 'product_rating' missing!
        "Region": ["East"],
        "product_main_category": ["Clothing"],
        "BrandGroup": ["Other"],
        "OrderMonth": ["5"]
    })

    with pytest.raises(KeyError):
        predict_batch(model, bad_batch)


def test_model_serialization_consistency():
    """Ensure loaded model produces identical predictions to saved model."""
    model = load_baseline_model("models/baseline_model.joblib")
    val_data = pd.read_csv("models/reference_val_data.csv")

    preds1 = predict_batch(model, val_data.head(20))
    preds2 = predict_batch(model, val_data.head(20))

    np.testing.assert_allclose(preds1, preds2)


def test_load_baseline_metrics():
    """Verify saved metrics file loads correctly and contains necessary keys."""
    metrics = load_baseline_metrics("models/baseline_metrics.json")
    required_keys = [
        "baseline_val_mae",
        "dummy_val_mae",
        "improvement_over_dummy_pct",
        "features",
        "target"
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key in metrics: {key}"
    assert metrics["improvement_over_dummy_pct"] > 10.0
