"""
Unit Tests for Drift Detection & Quality Checks Modules
Author: DriftMonitor Team
Purpose: Validates statistical drift detection (KS test, PSI, Chi-square, feature ranking)
         and batch quality validation per PROJECT_SPEC.md Sections 5.3, 5.4, and 9.
"""

import pytest
import numpy as np
import pandas as pd
from src.drift_detector import (
    ks_test,
    psi_numeric,
    psi_categorical,
    chi_square_categorical,
    detect_drift
)
from src.quality_checks import check_batch_quality


@pytest.fixture
def sample_numeric_data():
    """Provides reference and shifted numeric samples."""
    np.random.seed(42)
    ref = np.random.normal(loc=100.0, scale=15.0, size=500)
    shifted = np.random.normal(loc=160.0, scale=15.0, size=300)
    return ref, shifted


@pytest.fixture
def sample_categorical_data():
    """Provides reference and shifted categorical samples."""
    ref = ["North"] * 200 + ["South"] * 150 + ["East"] * 100 + ["West"] * 50
    shifted = ["North"] * 30 + ["South"] * 50 + ["East"] * 120 + ["West"] * 300
    return pd.Series(ref), pd.Series(shifted)


# -----------------------------------------------------------------------------
# 1. KS-Test Tests (Numeric)
# -----------------------------------------------------------------------------

def test_ks_test_identical_data():
    """Identical samples must yield KS statistic near 0 and large p-value (>= 0.05)."""
    np.random.seed(42)
    data = np.random.normal(loc=50.0, scale=10.0, size=300)
    stat, p_val = ks_test(data, data)
    assert stat == 0.0
    assert p_val == 1.0


def test_ks_test_strongly_shifted(sample_numeric_data):
    """Strongly shifted distribution must yield high KS statistic and p-value < 0.01."""
    ref, shifted = sample_numeric_data
    stat, p_val = ks_test(ref, shifted)
    assert stat > 0.5
    assert p_val < 0.001


def test_ks_test_empty_or_nan():
    """Handling empty arrays or arrays with NaNs gracefully."""
    ref = [np.nan, 10.0, 20.0, np.nan, 30.0]
    new = [10.0, 20.0, 30.0]
    stat, p_val = ks_test(ref, new)
    assert stat == 0.0
    assert p_val == 1.0

    stat_empty, p_val_empty = ks_test([], [1.0, 2.0])
    assert stat_empty == 0.0
    assert p_val_empty == 1.0


# -----------------------------------------------------------------------------
# 2. Numeric PSI Tests
# -----------------------------------------------------------------------------

def test_psi_numeric_identical_data():
    """Identical numeric data must yield PSI close to 0 (< 0.01)."""
    np.random.seed(42)
    data = np.random.normal(loc=100.0, scale=20.0, size=400)
    psi = psi_numeric(data, data, n_bins=10)
    assert psi < 0.01


def test_psi_numeric_strongly_shifted(sample_numeric_data):
    """Strongly shifted numeric data must yield PSI > 0.25 (major drift)."""
    ref, shifted = sample_numeric_data
    psi = psi_numeric(ref, shifted, n_bins=10)
    assert psi >= 0.25


def test_psi_numeric_duplicate_quantile_edges():
    """Quantiles with duplicate edges (e.g. repeated ratings) must not crash."""
    # Highly repetitive data where quantiles produce identical bin boundaries
    ref = [4.0] * 300 + [5.0] * 100 + [3.0] * 50 + [1.0] * 10
    new = [4.0] * 200 + [5.0] * 100 + [2.0] * 50
    psi = psi_numeric(ref, new, n_bins=10)
    assert isinstance(psi, float)
    assert psi >= 0.0


def test_psi_numeric_empty_bin_handling():
    """Empty bins in new batch must be smoothed with epsilon and not cause log(0) crash."""
    ref = np.linspace(0, 100, 500)
    # New batch concentrated entirely in a narrow slice [45, 55]
    new = np.random.uniform(45, 55, size=200)
    psi = psi_numeric(ref, new, n_bins=10)
    assert isinstance(psi, float)
    assert not np.isnan(psi)
    assert not np.isinf(psi)
    assert psi > 0.25


# -----------------------------------------------------------------------------
# 3. Categorical PSI & Chi-Square Tests
# -----------------------------------------------------------------------------

def test_psi_categorical_identical():
    """Identical categorical distribution must yield PSI close to 0."""
    series = pd.Series(["North"] * 100 + ["South"] * 50 + ["East"] * 50)
    psi = psi_categorical(series, series)
    assert psi < 0.01


def test_psi_categorical_strongly_shifted(sample_categorical_data):
    """Shifted categorical proportions must yield PSI >= 0.25."""
    ref, shifted = sample_categorical_data
    psi = psi_categorical(ref, shifted)
    assert psi >= 0.25


def test_psi_categorical_unseen_category():
    """New category in batch absent from reference must not crash."""
    ref = pd.Series(["A"] * 100 + ["B"] * 100)
    new = pd.Series(["A"] * 80 + ["B"] * 80 + ["BrandNewCategory"] * 40)
    psi = psi_categorical(ref, new)
    assert isinstance(psi, float)
    assert psi > 0.0


def test_chi_square_categorical():
    """Chi-square p-value must be large (>=0.05) when identical, and small (<0.01) when shifted."""
    ref = pd.Series(["North"] * 100 + ["South"] * 100)
    new_same = pd.Series(["North"] * 50 + ["South"] * 50)
    p_same = chi_square_categorical(ref, new_same)
    assert p_same >= 0.05

    new_shifted = pd.Series(["North"] * 10 + ["South"] * 90)
    p_shifted = chi_square_categorical(ref, new_shifted)
    assert p_shifted < 0.01


# -----------------------------------------------------------------------------
# 4. Full Drift Detection Pipeline Tests (detect_drift)
# -----------------------------------------------------------------------------

def test_detect_drift_no_drift():
    """Running detect_drift on two samples from the same distribution should detect no drift."""
    np.random.seed(42)
    n = 400
    ref_df = pd.DataFrame({
        "retail_price": np.random.normal(500, 50, n),
        "product_rating": np.random.choice([3.0, 4.0, 5.0], n),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "product_main_category": np.random.choice(["Electronics", "Clothing", "Books"], n),
        "BrandGroup": np.random.choice(["BrandA", "BrandB", "Other"], n),
        "OrderMonth": np.random.choice(["1", "2", "3"], n),
    })
    # Batch sampled from same distribution
    batch_df = ref_df.sample(n=200, random_state=123).reset_index(drop=True)

    result = detect_drift(ref_df, batch_df, batch_id=1)

    assert result["batch_id"] == 1
    assert "retail_price" in result["features"]
    assert "Region" in result["features"]
    assert len(result["ranking"]) == 6
    assert result["drift_detected"] is False


def test_detect_drift_ranking_puts_changed_feature_first():
    """Injected severe shift on retail_price must flag retail_price and rank it #1."""
    np.random.seed(42)
    n = 400
    ref_df = pd.DataFrame({
        "retail_price": np.random.normal(500, 50, n),
        "product_rating": np.random.choice([3.0, 4.0, 5.0], n),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "product_main_category": np.random.choice(["Electronics", "Clothing", "Books"], n),
        "BrandGroup": np.random.choice(["BrandA", "BrandB", "Other"], n),
        "OrderMonth": np.random.choice(["1", "2", "3"], n),
    })

    batch_df = ref_df.sample(n=200, random_state=123).copy().reset_index(drop=True)
    # Severe drift injected into retail_price (multiply by 3.0)
    batch_df["retail_price"] = batch_df["retail_price"] * 3.0

    result = detect_drift(ref_df, batch_df, batch_id=42)

    assert result["drift_detected"] is True
    assert result["features"]["retail_price"]["flagged"] is True
    assert result["features"]["retail_price"]["psi"] >= 0.25
    assert result["features"]["retail_price"]["ks_p"] < 0.01
    # Changed feature must be ranked #1
    assert result["ranking"][0] == "retail_price"


def test_detect_drift_configurable_rules():
    """Test switching drift_rule between 'both', 'ks_only', and 'psi_only'."""
    ref_df = pd.DataFrame({"retail_price": np.linspace(10, 100, 300)})
    batch_df = pd.DataFrame({"retail_price": np.linspace(20, 110, 150)})

    # 'both' requires psi >= 0.25 AND p < 0.01
    cfg_both = {"drift_rule": "both", "psi_flag_threshold": 0.25, "p_value_threshold": 0.01}
    res_both = detect_drift(ref_df, batch_df, config=cfg_both)
    
    # 'ks_only' requires only p < 0.01
    cfg_ks = {"drift_rule": "ks_only", "p_value_threshold": 0.01}
    res_ks = detect_drift(ref_df, batch_df, config=cfg_ks)

    assert "retail_price" in res_both["features"]
    assert "retail_price" in res_ks["features"]


# -----------------------------------------------------------------------------
# 5. Quality Checks Tests (check_batch_quality)
# -----------------------------------------------------------------------------

def test_quality_checks_valid_batch():
    """A clean, valid batch should pass all quality checks."""
    df = pd.DataFrame({
        "retail_price": [100.0, 200.0, 300.0],
        "product_rating": [4.0, 4.5, 3.8],
        "Region": ["North", "South", "East"],
        "product_main_category": ["Electronics", "Clothing", "Books"],
        "BrandGroup": ["BrandA", "BrandB", "Other"],
        "OrderMonth": ["1", "2", "3"]
    })
    res = check_batch_quality(df)
    assert res["passed"] is True
    assert len(res["issues"]) == 0


def test_quality_checks_missing_columns():
    """Batch missing required features must fail quality check."""
    df = pd.DataFrame({"retail_price": [100.0, 200.0]})
    res = check_batch_quality(df)
    assert res["passed"] is False
    assert any("Missing expected column" in issue for issue in res["issues"])


def test_quality_checks_non_positive_price():
    """Batch containing non-positive retail_price must fail quality check."""
    df = pd.DataFrame({
        "retail_price": [100.0, -50.0, 0.0],
        "product_rating": [4.0, 4.0, 4.0],
        "Region": ["North", "South", "East"],
        "product_main_category": ["Electronics", "Clothing", "Books"],
        "BrandGroup": ["BrandA", "BrandB", "Other"],
        "OrderMonth": ["1", "2", "3"]
    })
    res = check_batch_quality(df)
    assert res["passed"] is False
    assert any("retail_price <= 0" in issue for issue in res["issues"])


def test_quality_checks_discount_greater_than_retail():
    """Batch where discounted_price > retail_price must fail quality check."""
    df = pd.DataFrame({
        "retail_price": [100.0, 200.0],
        "discounted_price": [150.0, 180.0],  # 150 > 100 invalid
        "product_rating": [4.0, 4.0],
        "Region": ["North", "South"],
        "product_main_category": ["Electronics", "Clothing"],
        "BrandGroup": ["BrandA", "BrandB"],
        "OrderMonth": ["1", "2"]
    })
    res = check_batch_quality(df)
    assert res["passed"] is False
    assert any("discounted_price > retail_price" in issue for issue in res["issues"])


def test_quality_checks_high_missing_rate():
    """Batch with missing value share > 10% must fail quality check."""
    df = pd.DataFrame({
        "retail_price": [100.0, np.nan, np.nan, 200.0],  # 50% missing
        "product_rating": [4.0, 4.0, 4.0, 4.0],
        "Region": ["North", "South", "East", "West"],
        "product_main_category": ["Electronics", "Clothing", "Books", "Other"],
        "BrandGroup": ["BrandA", "BrandB", "Other", "Other"],
        "OrderMonth": ["1", "2", "3", "4"]
    })
    res = check_batch_quality(df)
    assert res["passed"] is False
    assert any("missing-value rate" in issue for issue in res["issues"])


def test_quality_checks_unseen_categories():
    """Batch with excessive unseen category values must fail quality check."""
    ref_df = pd.DataFrame({
        "retail_price": [100.0, 200.0],
        "product_rating": [4.0, 4.0],
        "Region": ["North", "South"],
        "product_main_category": ["Electronics", "Clothing"],
        "BrandGroup": ["BrandA", "BrandB"],
        "OrderMonth": ["1", "2"]
    })
    # Batch has 100% unseen Region ('Moon')
    batch_df = pd.DataFrame({
        "retail_price": [100.0, 200.0],
        "product_rating": [4.0, 4.0],
        "Region": ["Moon", "Mars"],
        "product_main_category": ["Electronics", "Clothing"],
        "BrandGroup": ["BrandA", "BrandB"],
        "OrderMonth": ["1", "2"]
    })
    res = check_batch_quality(batch_df, ref_df=ref_df)
    assert res["passed"] is False
    assert any("unseen category" in issue for issue in res["issues"])
