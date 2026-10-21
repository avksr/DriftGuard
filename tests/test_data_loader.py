"""
Unit Tests for Data Loader & Preprocessing
Author: DriftMonitor Team
Purpose: Verifies data cleaning rules, reference/clean-pool split, and absence of data leakage
         per PROJECT_SPEC.md Sections 4.3 to 4.7 and Section 9.
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    clean_raw_data,
    split_reference_and_pool,
    compute_reference_artifacts,
    apply_reference_transformations,
    prepare_datasets,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    TARGET_COLUMN,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Creates a synthetic DataFrame with various data anomalies to test cleaning rules."""
    return pd.DataFrame({
        "Customer_uniq_id": ["id_1", "id_2", "id_3", "id_4", "id_5", "id_6", "id_7", "id_1"],
        "Region": ["East ", " West", "North", "South ", "East", "West", "North", "East "],
        "Order_Date": [
            "24-10-2021", "15-05-2021", "01-12-2020", "20-07-2021",
            "10-08-2021", "invalid_date", "12-09-2021", "24-10-2021"
        ],
        "retail_price": [1000.0, 500.0, -10.0, 2000.0, 1500.0, 800.0, 300.0, 1000.0],
        "discounted_price": [800.0, 600.0, 5.0, 1000.0, 1500.0, 400.0, 0.0, 800.0],  # 600>500 invalid, 0.0 non-positive
        "product_rating": [4.5, 3.0, 2.0, 5.0, 1.0, 4.0, 3.5, 4.5],
        "Brand": ["BrandA ", "BrandB", "BrandC", "BrandA", "BrandD", "BrandE", "BrandF", "BrandA "],
        "product_main_category": ["Clothing ", "Automotive", "Computers", "Clothing", "Mobiles", "Clothing", "Clothing", "Clothing "],
        "product_specifications": ["spec1", "spec2", "spec3", "spec4", "spec5", "spec6", "spec7", "spec1"],
        "description": ["desc1", "desc2", "desc3", "desc4", "desc5", "desc6", "desc7", "desc1"]
    })


def test_cleaning_drops_long_text_and_unused_columns(sample_raw_data):
    """Verify long-text columns like description and specifications are dropped."""
    cleaned_df, _ = clean_raw_data(sample_raw_data)
    assert "product_specifications" not in cleaned_df.columns
    assert "description" not in cleaned_df.columns


def test_cleaning_removes_exact_duplicates(sample_raw_data):
    """Row 7 is an exact duplicate of row 0; it must be dropped."""
    cleaned_df, drop_stats = clean_raw_data(sample_raw_data)
    assert drop_stats["dropped_duplicates"] >= 1


def test_cleaning_removes_non_positive_and_invalid_prices(sample_raw_data):
    """
    Rows with retail_price <= 0 (id_3), discounted_price <= 0 (id_7),
    or discounted_price > retail_price (id_2) must be removed.
    """
    cleaned_df, drop_stats = clean_raw_data(sample_raw_data)
    assert (cleaned_df["retail_price"] > 0).all()
    # Check that id_2 (discounted 600 > retail 500) and id_3 (retail -10) were eliminated
    assert "id_2" not in cleaned_df["Customer_uniq_id"].values
    assert "id_3" not in cleaned_df["Customer_uniq_id"].values
    assert "id_7" not in cleaned_df["Customer_uniq_id"].values


def test_cleaning_removes_unparseable_dates(sample_raw_data):
    """Rows with unparseable Order_Date (id_6) must be dropped."""
    cleaned_df, drop_stats = clean_raw_data(sample_raw_data)
    assert "id_6" not in cleaned_df["Customer_uniq_id"].values


def test_cleaning_strips_whitespace():
    """Verify string columns have whitespace stripped."""
    df_raw = pd.DataFrame({
        "Customer_uniq_id": ["id_1"],
        "Region": ["  East  "],
        "Order_Date": ["24-10-2021"],
        "retail_price": [1000.0],
        "discounted_price": [800.0],
        "product_rating": [4.0],
        "Brand": [" BrandX "],
        "product_main_category": ["  Clothing  "]
    })
    cleaned_df, _ = clean_raw_data(df_raw)
    assert cleaned_df["Region"].iloc[0] == "East"
    assert cleaned_df["product_main_category"].iloc[0] == "Clothing"
    assert cleaned_df["Brand"].iloc[0] == "BrandX"


def test_target_bounded_between_0_and_1():
    """Target discount_pct must strictly satisfy 0.0 <= discount_pct <= 1.0."""
    df_raw = pd.DataFrame({
        "Customer_uniq_id": ["id_1", "id_2"],
        "Region": ["East", "West"],
        "Order_Date": ["24-10-2021", "25-10-2021"],
        "retail_price": [1000.0, 500.0],
        "discounted_price": [1000.0, 250.0],  # 0% and 50% discount
        "product_rating": [4.0, 3.0],
        "Brand": ["BrandA", "BrandB"],
        "product_main_category": ["Clothing", "Automotive"]
    })
    cleaned_df, _ = clean_raw_data(df_raw)
    assert (cleaned_df["discount_pct"] >= 0.0).all()
    assert (cleaned_df["discount_pct"] <= 1.0).all()
    assert cleaned_df["discount_pct"].iloc[0] == 0.0
    assert cleaned_df["discount_pct"].iloc[1] == 0.5


def test_reference_pool_split_disjoint_and_reproducible():
    """Reference and clean pool must be disjoint and reproducible with seed."""
    df = pd.DataFrame({
        "row_id": range(100),
        "val": range(100)
    })
    ref1, pool1 = split_reference_and_pool(df, ref_fraction=0.5, random_seed=42)
    ref2, pool2 = split_reference_and_pool(df, ref_fraction=0.5, random_seed=42)

    assert len(ref1) == 50
    assert len(pool1) == 50
    # Disjointness
    assert set(ref1["row_id"]).isdisjoint(set(pool1["row_id"]))
    # Determinism
    pd.testing.assert_frame_equal(ref1, ref2)
    pd.testing.assert_frame_equal(pool1, pool2)


def test_no_data_leakage_in_reference_transformations():
    """
    Ensure price cap and top brands are derived strictly from reference data,
    and unknown brands in pool are assigned to 'Other'.
    """
    ref_df = pd.DataFrame({
        "retail_price": [100.0] * 995 + [10000.0] * 5,  # 99.5th percentile ~100.0
        "product_rating": [3.0] * 1000,
        "Brand": ["Brand1"] * 500 + ["Brand2"] * 500,
        "Region": ["East"] * 1000,
        "product_main_category": ["Clothing"] * 1000,
        "OrderMonth": ["1"] * 1000,
        "discount_pct": [0.2] * 1000
    })
    artifacts = compute_reference_artifacts(ref_df, top_brands=2)
    assert "Brand1" in artifacts["top_brands"]
    assert "Brand2" in artifacts["top_brands"]
    assert len(artifacts["top_brands"]) == 2

    # Now create pool data with an unseen brand 'Brand99' and an extreme price 50000.0
    pool_df = pd.DataFrame({
        "retail_price": [50000.0],
        "product_rating": [4.0],
        "Brand": ["Brand99"],
        "Region": ["West"],
        "product_main_category": ["Automotive"],
        "OrderMonth": ["2"],
        "discount_pct": [0.1]
    })
    transformed_pool = apply_reference_transformations(pool_df, artifacts)
    # Brand99 must be mapped to 'Other'
    assert transformed_pool["BrandGroup"].iloc[0] == "Other"
    # Price must be capped at reference price_cap
    assert transformed_pool["retail_price"].iloc[0] <= artifacts["price_cap"]


def test_prepare_datasets_on_real_dataset():
    """End-to-end integration test verifying real CSV produces expected reference and clean pool."""
    ref_df, pool_df, artifacts = prepare_datasets(config_path="config.yaml", save_to_disk=False)

    assert len(ref_df) > 0
    assert len(pool_df) > 0
    assert len(ref_df) == len(pool_df)  # 50/50 split

    # Check all feature columns exist
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in ref_df.columns, f"Column '{col}' missing from reference DataFrame"
        assert col in pool_df.columns, f"Column '{col}' missing from clean pool DataFrame"

    # Check target bounds
    assert (ref_df[TARGET_COLUMN] >= 0.0).all()
    assert (ref_df[TARGET_COLUMN] <= 1.0).all()
    assert (pool_df[TARGET_COLUMN] >= 0.0).all()
    assert (pool_df[TARGET_COLUMN] <= 1.0).all()

    # Check BrandGroup has top brands + Other
    assert "Other" in ref_df["BrandGroup"].unique()
    assert len(artifacts["top_brands"]) == 10
