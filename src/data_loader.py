"""
Data Loader & Preprocessing Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements data cleaning, feature engineering, and reference/clean-pool splitting
         per PROJECT_SPEC.md Sections 4.3 to 4.7, strictly avoiding data leakage.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional
import yaml
import pandas as pd
import numpy as np


# Canonical feature definitions per Section 4.4
NUMERIC_FEATURES: List[str] = ["retail_price", "product_rating"]
CATEGORICAL_FEATURES: List[str] = ["Region", "product_main_category", "BrandGroup", "OrderMonth"]
ALL_FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "discount_pct"


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml.
        
    Returns:
        Dictionary of configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(csv_path: str = "data/E-Commerce.csv") -> pd.DataFrame:
    """
    Load raw CSV data handling character encoding fallbacks.
    
    Args:
        csv_path: Path to the raw CSV file.
        
    Returns:
        Raw pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="latin-1")


def clean_raw_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply initial cleaning rules per Section 4.3 that do not depend on reference statistics:
    1. Drop long text columns and unneeded metadata columns.
    2. Strip whitespace from string columns (audit revealed trailing spaces like 'East ').
    3. Drop exact duplicate rows.
    4. Drop rows with missing essential fields.
    5. Drop rows with retail_price <= 0 or discounted_price <= 0.
    6. Drop rows where discounted_price > retail_price.
    7. Parse Order_Date (DD-MM-YYYY) and derive OrderMonth.
    8. Compute target: discount_pct = (retail_price - discounted_price) / retail_price.
    
    Args:
        raw_df: Raw imported DataFrame.
        
    Returns:
        Tuple of (cleaned DataFrame, dictionary of drop counts for reporting).
    """
    initial_rows = len(raw_df)
    drop_stats = {"initial_rows": initial_rows}
    df = raw_df.copy()

    # 1. Drop unused text and auxiliary columns
    cols_to_drop = [
        "product_specifications",
        "description",
        "product_category_filter",
        "product_category_subfilter",
        "product_unique ID",
        "product_name",
        "product_subcategory",
        "Expected_Delivery_Date",
        "Delivered_Date"
    ]
    existing_drop_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # 2. Strip whitespace from text columns
    for col in ["Region", "product_main_category", "Brand"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3. Drop exact duplicates
    n_before_dupes = len(df)
    df = df.drop_duplicates()
    drop_stats["dropped_duplicates"] = n_before_dupes - len(df)

    # 4. Filter missing required values
    required_cols = ["retail_price", "discounted_price", "Region", "product_main_category", "Order_Date"]
    n_before_na = len(df)
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    drop_stats["dropped_missing"] = n_before_na - len(df)

    # 5. Filter invalid prices (retail_price <= 0 or discounted_price <= 0)
    n_before_prices = len(df)
    df = df[(df["retail_price"] > 0) & (df["discounted_price"] > 0)]
    drop_stats["dropped_non_positive_prices"] = n_before_prices - len(df)

    # 6. Filter rows where discounted_price > retail_price
    n_before_inconsistent = len(df)
    df = df[df["discounted_price"] <= df["retail_price"]]
    drop_stats["dropped_discount_gt_retail"] = n_before_inconsistent - len(df)

    # 7. Parse dates and derive OrderMonth
    # Using format='%d-%m-%Y' as verified during Week-1 audit
    parsed_dates = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    n_before_dates = len(df)
    df = df[parsed_dates.notna()].copy()
    drop_stats["dropped_unparseable_dates"] = n_before_dates - len(df)
    df["Order_Date"] = parsed_dates[parsed_dates.notna()]
    df["OrderMonth"] = df["Order_Date"].dt.month.astype(str)

    # 8. Compute target: discount_pct
    # discount_pct represents fraction of discount, bounded in [0.0, 1.0]
    df["discount_pct"] = (df["retail_price"] - df["discounted_price"]) / df["retail_price"]

    # Drop discounted_price to prevent leakage into modeling features (Section 4.4)
    if "discounted_price" in df.columns:
        df = df.drop(columns=["discounted_price"])

    drop_stats["final_rows"] = len(df)
    return df, drop_stats


def split_reference_and_pool(
    df: pd.DataFrame,
    ref_fraction: float = 0.5,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the cleaned dataset randomly into reference half and clean pool (Section 4.6).
    Random splitting ensures the clean pool has zero baseline drift relative to reference.
    
    Args:
        df: Cleaned DataFrame.
        ref_fraction: Proportion of data allocated to reference (default 0.5).
        random_seed: Fixed random seed for reproducibility.
        
    Returns:
        Tuple of (reference_df, clean_pool_df).
    """
    np.random.seed(random_seed)
    n_total = len(df)
    shuffled_indices = np.random.permutation(n_total)
    ref_size = int(n_total * ref_fraction)

    ref_indices = shuffled_indices[:ref_size]
    pool_indices = shuffled_indices[ref_size:]

    reference_df = df.iloc[ref_indices].copy().reset_index(drop=True)
    clean_pool_df = df.iloc[pool_indices].copy().reset_index(drop=True)

    return reference_df, clean_pool_df


def compute_reference_artifacts(
    reference_df: pd.DataFrame,
    top_brands: int = 10
) -> Dict[str, Any]:
    """
    Compute reference statistics ONLY from reference data to prevent leakage (Section 4.7):
    - Price cap: 99.5th percentile of retail_price.
    - Top brands list: top N most frequent brands in reference.
    - Rating median: median product_rating in reference for imputing any missing values.
    
    Args:
        reference_df: Reference DataFrame.
        top_brands: Number of frequent brands to retain.
        
    Returns:
        Dictionary of reference artifacts.
    """
    price_cap = float(reference_df["retail_price"].quantile(0.995))
    top_brands_list = reference_df["Brand"].value_counts().head(top_brands).index.tolist()
    rating_median = float(reference_df["product_rating"].median())

    return {
        "price_cap": price_cap,
        "top_brands": top_brands_list,
        "rating_median": rating_median,
        "reference_row_count": len(reference_df)
    }


def apply_reference_transformations(
    df: pd.DataFrame,
    ref_artifacts: Dict[str, Any]
) -> pd.DataFrame:
    """
    Apply reference-derived transformations to a DataFrame (reference or incoming batch):
    - Cap retail_price at reference 99.5th percentile.
    - Map Brand to BrandGroup (top brands keep name, others become 'Other').
    - Impute any missing product_rating with reference median.
    
    Args:
        df: Input DataFrame.
        ref_artifacts: Reference statistics dictionary.
        
    Returns:
        Transformed DataFrame.
    """
    transformed = df.copy()

    # 1. Cap retail_price
    price_cap = ref_artifacts["price_cap"]
    transformed["retail_price"] = transformed["retail_price"].clip(upper=price_cap)

    # 2. Impute product_rating if missing
    rating_median = ref_artifacts["rating_median"]
    transformed["product_rating"] = transformed["product_rating"].fillna(rating_median)

    # 3. Create BrandGroup
    top_brands = set(ref_artifacts["top_brands"])
    transformed["BrandGroup"] = transformed["Brand"].apply(
        lambda b: b if b in top_brands else "Other"
    )

    # Keep only defined features + target + optional tracking id
    cols_to_keep = [col for col in ALL_FEATURES + [TARGET_COLUMN] if col in transformed.columns]
    if "Customer_uniq_id" in transformed.columns:
        cols_to_keep.append("Customer_uniq_id")
    if "Order_Date" in transformed.columns:
        cols_to_keep.append("Order_Date")

    return transformed[cols_to_keep]


def prepare_datasets(
    config_path: str = "config.yaml",
    save_to_disk: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Complete end-to-end data pipeline:
    1. Loads configuration and raw CSV.
    2. Cleans raw data and constructs the target.
    3. Splits data into reference half and clean pool (no leakage).
    4. Computes reference artifacts (price cap, top brands, median rating).
    5. Applies transformations using reference artifacts to both halves.
    6. Optionally saves cleaned datasets and metadata to the data/ folder.
    
    Args:
        config_path: Path to config.yaml.
        save_to_disk: Whether to save processed CSVs to disk.
        
    Returns:
        Tuple of (final_reference_df, final_clean_pool_df, ref_artifacts).
    """
    config = load_config(config_path)
    csv_path = config["data"]["csv_path"]
    ref_fraction = config["data"].get("reference_fraction", 0.5)
    top_brands = config["data"].get("top_brands", 10)
    seed = config.get("random_seed", 42)

    # Step 1: Load and initial clean
    raw_df = load_raw_data(csv_path)
    cleaned_df, drop_stats = clean_raw_data(raw_df)

    # Step 2: Split 50/50 without leakage
    ref_df_raw, pool_df_raw = split_reference_and_pool(
        cleaned_df,
        ref_fraction=ref_fraction,
        random_seed=seed
    )

    # Step 3: Compute reference artifacts strictly on reference
    ref_artifacts = compute_reference_artifacts(ref_df_raw, top_brands=top_brands)
    ref_artifacts["drop_stats"] = drop_stats

    # Step 4: Apply reference artifacts to both datasets
    reference_df = apply_reference_transformations(ref_df_raw, ref_artifacts)
    clean_pool_df = apply_reference_transformations(pool_df_raw, ref_artifacts)

    # Step 5: Save to disk for reproducibility across all later modules
    if save_to_disk:
        os.makedirs("data", exist_ok=True)
        reference_df.to_csv("data/reference_data.csv", index=False)
        clean_pool_df.to_csv("data/clean_pool.csv", index=False)
        combined_df = pd.concat([
            reference_df.assign(dataset_split="reference"),
            clean_pool_df.assign(dataset_split="clean_pool")
        ], ignore_index=True)
        combined_df.to_csv("data/cleaned_data.csv", index=False)
        with open("data/reference_artifacts.json", "w", encoding="utf-8") as f:
            json.dump(ref_artifacts, f, indent=2)

    return reference_df, clean_pool_df, ref_artifacts


def print_data_summary() -> None:
    """
    Prints dataset size before and after cleaning, and summary of each feature and target.
    """
    print("=" * 80)
    print(" DRIFTMONITOR: DATA PREPARATION & FEATURE SUMMARY")
    print("=" * 80)

    config = load_config("config.yaml")
    raw_df = load_raw_data(config["data"]["csv_path"])
    ref_df, pool_df, artifacts = prepare_datasets("config.yaml", save_to_disk=True)

    print(f"\n1. DATASET SIZES:")
    print(f"   Raw Dataset Rows:          {len(raw_df):,}")
    print(f"   Cleaned Reference Rows:    {len(ref_df):,} ({len(ref_df) / len(raw_df) * 100:.1f}%)")
    print(f"   Cleaned Pool Rows:         {len(pool_df):,} ({len(pool_df) / len(raw_df) * 100:.1f}%)")
    print(f"   Total Cleaned Rows:        {len(ref_df) + len(pool_df):,}")

    drop_stats = artifacts.get("drop_stats", {})
    print(f"\n2. DROP / CLEANING STATS:")
    for k, v in drop_stats.items():
        print(f"   - {k:30s}: {v}")

    print(f"\n3. REFERENCE ARTIFACTS (Computed strictly from Reference Data):")
    print(f"   - 99.5th Retail Price Cap:  {artifacts['price_cap']:.2f}")
    print(f"   - Median Product Rating:    {artifacts['rating_median']:.2f}")
    print(f"   - Top {len(artifacts['top_brands'])} Brands:             {artifacts['top_brands']}")

    print(f"\n4. NUMERIC FEATURES SUMMARY (Reference Data):")
    for feat in NUMERIC_FEATURES:
        series = ref_df[feat]
        print(f"   Feature: '{feat}'")
        print(f"     Min: {series.min():.2f} | Max: {series.max():.2f} | Mean: {series.mean():.2f} | Std: {series.std():.2f}")

    print(f"\n5. CATEGORICAL FEATURES SUMMARY (Reference Data):")
    for feat in CATEGORICAL_FEATURES:
        series = ref_df[feat]
        n_unique = series.nunique()
        top3 = series.value_counts().head(3).to_dict()
        top3_str = ", ".join([f"{k}: {v}" for k, v in top3.items()])
        print(f"   Feature: '{feat:22s}' | Unique: {n_unique:2d} | Top 3: {top3_str}")

    print(f"\n6. TARGET VARIABLE SUMMARY ('{TARGET_COLUMN}'):")
    target = ref_df[TARGET_COLUMN]
    print(f"   Min:    {target.min():.4f}")
    print(f"   Max:    {target.max():.4f}")
    print(f"   Mean:   {target.mean():.4f} ({target.mean() * 100:.1f}%)")
    print(f"   Median: {target.median():.4f} ({target.median() * 100:.1f}%)")
    print(f"   Std:    {target.std():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    print_data_summary()
