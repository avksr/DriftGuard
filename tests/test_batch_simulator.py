"""
Unit tests for batch_simulator.py
Tests synthetic drift injection functions, batch sampling, and ground truth generation
per PROJECT_SPEC.md Section 5.2.
"""

import pytest
import numpy as np
import pandas as pd

from src.batch_simulator import (
    shift_mean,
    shift_additive,
    change_spread,
    shift_category_mix,
    label_shift,
    sample_batch,
    simulate_run,
    SimulationRun
)
from src.data_loader import load_config


@pytest.fixture
def sample_numeric_series():
    np.random.seed(42)
    return pd.Series(np.random.uniform(50.0, 500.0, size=200), name="retail_price")


@pytest.fixture
def sample_rating_series():
    np.random.seed(42)
    return pd.Series(np.random.uniform(2.5, 4.8, size=200), name="product_rating")


@pytest.fixture
def sample_categorical_series():
    np.random.seed(42)
    categories = ["North", "South", "East", "West"]
    probs = [0.5, 0.2, 0.2, 0.1]
    return pd.Series(np.random.choice(categories, size=300, p=probs), name="Region")


@pytest.fixture
def sample_pool_df():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "retail_price": np.random.uniform(100.0, 1000.0, size=n),
        "product_rating": np.random.uniform(1.0, 5.0, size=n),
        "Region": np.random.choice(["North", "South", "East", "West"], size=n),
        "product_main_category": np.random.choice(["Clothing", "Electronics", "Home"], size=n),
        "BrandGroup": np.random.choice(["Brand_A", "Brand_B", "Other"], size=n),
        "OrderMonth": np.random.choice([1, 2, 3, 4, 5, 6], size=n),
        "discount_pct": np.random.uniform(0.1, 0.7, size=n)
    })


def test_shift_mean(sample_numeric_series):
    original_mean = sample_numeric_series.mean()
    shifted = shift_mean(sample_numeric_series, factor=1.5)
    
    assert len(shifted) == len(sample_numeric_series)
    assert shifted.min() >= 0.01
    assert pytest.approx(shifted.mean(), rel=1e-3) == original_mean * 1.5


def test_shift_additive(sample_rating_series):
    shifted = shift_additive(sample_rating_series, delta=-1.0, min_val=1.0, max_val=5.0)
    
    assert len(shifted) == len(sample_rating_series)
    assert shifted.min() >= 1.0
    assert shifted.max() <= 5.0
    # Values not at boundary should be exactly original - 1.0
    unclipped_mask = sample_rating_series - 1.0 >= 1.0
    np.testing.assert_allclose(shifted[unclipped_mask], (sample_rating_series - 1.0)[unclipped_mask])


def test_change_spread(sample_numeric_series):
    # Test on a series where expansion doesn't hit lower clipping bound
    safe_series = pd.Series([100.0, 150.0, 200.0, 250.0, 300.0])
    spread_out = change_spread(safe_series, factor=1.2)
    assert pytest.approx(spread_out.mean(), abs=1e-3) == safe_series.mean()
    assert spread_out.var() > safe_series.var()

    # Test clipping guarantee on large factor
    extreme_spread = change_spread(sample_numeric_series, factor=3.0)
    assert extreme_spread.min() >= 0.01


def test_shift_category_mix(sample_categorical_series):
    original_counts = sample_categorical_series.value_counts(normalize=True)
    top_cat = original_counts.index[0]
    original_top_share = original_counts.iloc[0]
    
    shifted = shift_category_mix(sample_categorical_series, drop=0.25, random_state=42)
    new_counts = shifted.value_counts(normalize=True)
    new_top_share = new_counts.get(top_cat, 0.0)
    
    assert len(shifted) == len(sample_categorical_series)
    assert new_top_share < original_top_share
    assert pytest.approx(new_top_share, abs=0.05) == (original_top_share - 0.25)


def test_label_shift(sample_numeric_series):
    target = pd.Series([0.2, 0.5, 0.8, 0.9])
    shifted = label_shift(target, factor=1.5)
    
    assert len(shifted) == len(target)
    assert shifted.max() <= 1.0
    assert shifted.min() >= 0.0
    assert pytest.approx(shifted.iloc[0], rel=1e-3) == 0.3
    assert shifted.iloc[3] == 1.0  # 0.9 * 1.5 = 1.35 clipped to 1.0


def test_sample_batch(sample_pool_df):
    batch = sample_batch(sample_pool_df, batch_size=300, random_state=123)
    assert len(batch) == 300
    assert list(batch.columns) == list(sample_pool_df.columns)
    
    # Reproducibility
    batch_again = sample_batch(sample_pool_df, batch_size=300, random_state=123)
    pd.testing.assert_frame_equal(batch, batch_again)


def test_simulate_run_sudden_drift(sample_pool_df):
    config = load_config("config.yaml")
    run = simulate_run(
        pool_df=sample_pool_df,
        config=config,
        run_id="test_run_sudden",
        drift_type="shift_mean",
        drift_size="medium",
        drift_shape="sudden",
        drift_start_batch=10,
        affected_features=["retail_price"]
    )
    
    assert isinstance(run, SimulationRun)
    assert run.has_drift is True
    assert len(run.batches) == config["n_batches_per_run"]
    assert len(run.ground_truth_per_batch) == config["n_batches_per_run"]
    
    # Batches before start batch (0 to 9) must not have drift
    for i in range(10):
        gt = run.ground_truth_per_batch[i]
        assert gt["is_drifted"] is False
        assert gt["multiplier"] == 0.0
        assert gt["affected_features"] == []
        
    # Batches from 10 onwards must have full drift
    for i in range(10, len(run.batches)):
        gt = run.ground_truth_per_batch[i]
        assert gt["is_drifted"] is True
        assert gt["multiplier"] == 1.0
        assert gt["affected_features"] == ["retail_price"]


def test_simulate_run_gradual_drift(sample_pool_df):
    config = load_config("config.yaml")
    ramp = 8
    start = 12
    run = simulate_run(
        pool_df=sample_pool_df,
        config=config,
        run_id="test_run_gradual",
        drift_type="shift_mean",
        drift_size="large",
        drift_shape="gradual",
        drift_start_batch=start,
        gradual_ramp_batches=ramp,
        affected_features=["retail_price"]
    )
    
    # Before start: 0.0
    assert run.ground_truth_per_batch[start - 1]["multiplier"] == 0.0
    # First drifted batch has fraction 1/ramp
    assert pytest.approx(run.ground_truth_per_batch[start]["multiplier"]) == 1.0 / ramp
    # Ramp middle has increasing multiplier
    assert run.ground_truth_per_batch[start + 2]["multiplier"] > run.ground_truth_per_batch[start]["multiplier"]
    # At start + ramp - 1, multiplier reaches 1.0
    assert pytest.approx(run.ground_truth_per_batch[start + ramp - 1]["multiplier"]) == 1.0
    # After ramp, multiplier stays 1.0
    assert pytest.approx(run.ground_truth_per_batch[start + ramp + 2]["multiplier"]) == 1.0


def test_simulate_run_no_drift(sample_pool_df):
    config = load_config("config.yaml")
    run = simulate_run(
        pool_df=sample_pool_df,
        config=config,
        run_id="test_run_none",
        drift_type="none",
        affected_features=[]
    )
    assert run.has_drift is False
    for gt in run.ground_truth_per_batch:
        assert gt["is_drifted"] is False
        assert gt["multiplier"] == 0.0
