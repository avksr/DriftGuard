import sys
import os
import shutil
import subprocess
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config
from src.data_loader import (
    load_raw_data,
    clean_raw_data,
    prepare_datasets,
    split_reference_and_pool,
    compute_reference_artifacts,
    apply_reference_transformations,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from sklearn.model_selection import train_test_split

def run_cleaning_counts():
    config = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    csv_path = os.path.join(PROJECT_ROOT, config["data"]["csv_path"])
    raw_df = load_raw_data(csv_path)
    
    raw_rows, raw_cols = raw_df.shape
    print(f"Raw shape: {raw_rows} rows, {raw_cols} columns")

    # Step-by-step replay matching clean_raw_data
    df = raw_df.copy()
    
    # 1. Drop unused columns & strip whitespace
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
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    for col in ["Region", "product_main_category", "Brand"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    records = []

    # Rule 1: Drop exact duplicates
    r1_before = len(df)
    df = df.drop_duplicates()
    r1_after = len(df)
    records.append({
        "rule_name": "Drop exact duplicates",
        "rows_before": r1_before,
        "rows_after": r1_after,
        "rows_removed": r1_before - r1_after
    })

    # Rule 2: Drop missing essential fields
    r2_before = len(df)
    required_cols = ["retail_price", "discounted_price", "Region", "product_main_category", "Order_Date"]
    df = df.dropna(subset=[c for c in required_cols if c in df.columns])
    r2_after = len(df)
    records.append({
        "rule_name": "Drop missing essential fields",
        "rows_before": r2_before,
        "rows_after": r2_after,
        "rows_removed": r2_before - r2_after
    })

    # Rule 3: Drop non-positive prices
    r3_before = len(df)
    df = df[(df["retail_price"] > 0) & (df["discounted_price"] > 0)]
    r3_after = len(df)
    records.append({
        "rule_name": "Drop non-positive prices",
        "rows_before": r3_before,
        "rows_after": r3_after,
        "rows_removed": r3_before - r3_after
    })

    # Rule 4: Drop discounted_price > retail_price
    r4_before = len(df)
    df = df[df["discounted_price"] <= df["retail_price"]]
    r4_after = len(df)
    records.append({
        "rule_name": "Drop discounted price greater than retail price",
        "rows_before": r4_before,
        "rows_after": r4_after,
        "rows_removed": r4_before - r4_after
    })

    # Rule 5: Drop unparseable dates
    r5_before = len(df)
    parsed_dates = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
    df = df[parsed_dates.notna()].copy()
    r5_after = len(df)
    records.append({
        "rule_name": "Drop unparseable dates",
        "rows_before": r5_before,
        "rows_after": r5_after,
        "rows_removed": r5_before - r5_after
    })
    
    df["Order_Date"] = parsed_dates[parsed_dates.notna()]
    df["OrderMonth"] = df["Order_Date"].dt.month.astype(str)
    df["discount_pct"] = (df["retail_price"] - df["discounted_price"]) / df["retail_price"]
    if "discounted_price" in df.columns:
        df = df.drop(columns=["discounted_price"])

    cleaned_rows = len(df)
    print(f"Step-by-step cleaned rows: {cleaned_rows}")

    # Compare against pipeline prepare_datasets (with save_to_disk=False)
    ref_df, pool_df, artifacts = prepare_datasets(config, save_to_disk=False)
    pipeline_total_rows = len(ref_df) + len(pool_df)
    print(f"Pipeline reference rows: {len(ref_df)}, pool rows: {len(pool_df)}, total: {pipeline_total_rows}")

    assert cleaned_rows == pipeline_total_rows, f"Mismatch: step-by-step={cleaned_rows} vs pipeline={pipeline_total_rows}"
    print("Assertion passed: Step-by-step cleaning row count matches pipeline prepare_datasets!")

    # Save cleaning_counts.csv
    counts_df = pd.DataFrame(records)
    out_counts_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "cleaning_counts.csv")
    counts_df.to_csv(out_counts_path, index=False, encoding="utf-8")
    print(f"Saved {out_counts_path}")
    print(counts_df.to_string(index=False))

    # Reference train / val split
    train_fraction = config.get("data", {}).get("baseline_train_fraction", 0.8)
    seed = config.get("random_seed", 42)
    X = ref_df[ALL_FEATURES].copy()
    y = ref_df[TARGET_COLUMN]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True
    )
    ref_train_size = len(X_train)
    ref_val_size = len(X_val)

    # Distinct values of categorical features in reference_df and transformed pool
    cat_distinct = {}
    for cat in ["Region", "product_main_category", "BrandGroup", "OrderMonth"]:
        combined_cat = pd.concat([ref_df[cat], pool_df[cat]])
        cat_distinct[cat] = int(combined_cat.nunique())

    # Date range of Order_Date
    min_date = df["Order_Date"].min().strftime("%Y-%m-%d")
    max_date = df["Order_Date"].max().strftime("%Y-%m-%d")
    date_range_str = f"{min_date} to {max_date}"

    facts = [
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "raw_columns", "value": raw_cols},
        {"metric": "cleaned_rows", "value": cleaned_rows},
        {"metric": "reference_half_size", "value": len(ref_df)},
        {"metric": "reference_train_size_80pct", "value": ref_train_size},
        {"metric": "reference_val_size_20pct", "value": ref_val_size},
        {"metric": "clean_pool_size", "value": len(pool_df)},
        {"metric": "distinct_Region", "value": cat_distinct["Region"]},
        {"metric": "distinct_product_main_category", "value": cat_distinct["product_main_category"]},
        {"metric": "distinct_BrandGroup", "value": cat_distinct["BrandGroup"]},
        {"metric": "distinct_OrderMonth", "value": cat_distinct["OrderMonth"]},
        {"metric": "order_date_range", "value": date_range_str}
    ]
    facts_df = pd.DataFrame(facts)
    out_facts_path = os.path.join(PROJECT_ROOT, "report_assets", "data", "dataset_facts.csv")
    facts_df.to_csv(out_facts_path, index=False, encoding="utf-8")
    print(f"Saved {out_facts_path}")
    print(facts_df.to_string(index=False))

    # Also copy baseline_metrics.json, config.yaml, and results/run_manifest.json
    data_dir = os.path.join(PROJECT_ROOT, "report_assets", "data")
    shutil.copy2(os.path.join(PROJECT_ROOT, "models", "baseline_metrics.json"), os.path.join(data_dir, "baseline_metrics.json"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "config.yaml"), os.path.join(data_dir, "config.yaml"))
    shutil.copy2(os.path.join(PROJECT_ROOT, "results", "run_manifest.json"), os.path.join(data_dir, "run_manifest.json"))
    print("Copied baseline_metrics.json, config.yaml, and run_manifest.json to report_assets/data/")

if __name__ == "__main__":
    run_cleaning_counts()

