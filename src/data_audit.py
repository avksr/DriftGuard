"""
Data Audit Script for DriftMonitor
Author: DriftMonitor Team
Purpose: Conduct comprehensive Week-1 audit on the raw e-commerce CSV dataset per PROJECT_SPEC.md Section 4.2.
"""

import os
import sys
from typing import Dict, Any, Tuple
import yaml
import pandas as pd
import numpy as np


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration file safely."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_data_audit(csv_path: str = "data/E-Commerce.csv") -> Dict[str, Any]:
    """
    Run full audit per PROJECT_SPEC.md Section 4.2:
    1. Shape, column names, data types.
    2. Missing values per column (count and percent).
    3. Number of duplicate rows.
    4. For retail_price and discounted_price: format, min, max, mean, zero/negative count, discounted > retail.
    5. For Order_Date, Delivered_Date: format, min/max, unparseable count.
    6. Unique values for Region, product_main_category, product_subcategory, Brand (top 10).
    7. Distribution of product_rating.
    8. Distribution of candidate targets: discount_pct and delivery_days.
    """
    print("=" * 80)
    print(f" DRIFTMONITOR DATA AUDIT: {csv_path}")
    print("=" * 80)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    # Load raw data with fallback encodings
    try:
        raw_df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("[WARN] UTF-8 decoding failed, falling back to latin-1 encoding.")
        raw_df = pd.read_csv(csv_path, encoding="latin-1")

    # 1. Shape, Column Names, Data Types
    n_rows, n_cols = raw_df.shape
    print("\n--- 1. SHAPE, COLUMNS & RAW DATA TYPES ---")
    print(f"Total Rows:    {n_rows:,}")
    print(f"Total Columns: {n_cols}")
    print("\nColumns and raw dtypes:")
    for idx, (col, dtype) in enumerate(raw_df.dtypes.items(), start=1):
        print(f"  {idx:2d}. {col:30s} : {str(dtype):10s} (non-null: {raw_df[col].count():,})")

    # 2. Missing Values per Column
    print("\n--- 2. MISSING VALUES PER COLUMN ---")
    missing_counts = raw_df.isnull().sum()
    missing_pcts = (missing_counts / n_rows) * 100.0
    for col in raw_df.columns:
        cnt = missing_counts[col]
        pct = missing_pcts[col]
        flag = " [!] >20% MISSING" if pct > 20.0 else ""
        print(f"  {col:30s} : {cnt:5d} missing ({pct:5.2f}%){flag}")

    # 3. Duplicate Rows
    n_exact_dupes = raw_df.duplicated().sum()
    n_id_dupes = raw_df.duplicated(subset=["Customer_uniq_id"]).sum() if "Customer_uniq_id" in raw_df.columns else 0
    print("\n--- 3. DUPLICATE ROWS ---")
    print(f"Exact duplicate rows across all columns: {n_exact_dupes}")
    print(f"Duplicate Customer_uniq_id rows:         {n_id_dupes}")

    # 4. Pricing Columns Analysis: retail_price & discounted_price
    print("\n--- 4. PRICING COLUMNS ANALYSIS ---")
    price_cols = ["retail_price", "discounted_price"]
    clean_prices = {}

    for col in price_cols:
        if col not in raw_df.columns:
            print(f"  [ERROR] Expected price column '{col}' missing from dataset!")
            continue

        series = raw_df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        print(f"\n  Analyzing '{col}':")
        print(f"    Raw type:              {'Numeric' if is_numeric else 'Text / Object'}")

        # Coerce to numeric
        converted = pd.to_numeric(series.astype(str).str.replace(",", "").str.replace("₹", "").str.strip(), errors="coerce")
        n_unparseable = converted.isna().sum() - series.isna().sum()
        clean_prices[col] = converted

        valid_vals = converted.dropna()
        n_zeros = (valid_vals == 0).sum()
        n_negatives = (valid_vals < 0).sum()

        print(f"    Unparseable values:    {n_unparseable}")
        print(f"    Valid numeric values:  {len(valid_vals):,}")
        print(f"    Zero values:           {n_zeros}")
        print(f"    Negative values:       {n_negatives}")
        if len(valid_vals) > 0:
            print(f"    Min:                   {valid_vals.min():.2f}")
            print(f"    Max:                   {valid_vals.max():.2f}")
            print(f"    Mean:                  {valid_vals.mean():.2f}")
            print(f"    Median:                {valid_vals.median():.2f}")
            print(f"    Std Dev:               {valid_vals.std():.2f}")

    if "retail_price" in clean_prices and "discounted_price" in clean_prices:
        rp = clean_prices["retail_price"]
        dp = clean_prices["discounted_price"]
        both_valid = rp.notna() & dp.notna()
        n_invalid_relation = (both_valid & (dp > rp)).sum()
        n_equal = (both_valid & (dp == rp)).sum()
        print(f"\n  Price consistency checks:")
        print(f"    Rows where discounted_price > retail_price: {n_invalid_relation} ({n_invalid_relation / both_valid.sum() * 100:.2f}%)")
        print(f"    Rows where discounted_price == retail_price: {n_equal} ({n_equal / both_valid.sum() * 100:.2f}%)")

    # 5. Date Columns Analysis: Order_Date & Delivered_Date
    print("\n--- 5. DATE COLUMNS ANALYSIS ---")
    date_cols = ["Order_Date", "Expected_Delivery_Date", "Delivered_Date"]
    parsed_dates = {}

    for col in date_cols:
        if col not in raw_df.columns:
            print(f"  [WARN] Expected date column '{col}' missing from dataset!")
            continue

        raw_series = raw_df[col]
        # Check sample formats
        non_null_samples = raw_series.dropna().head(3).tolist()
        print(f"\n  Analyzing '{col}':")
        print(f"    Sample values:         {non_null_samples}")

        # Attempt parsing with dayfirst=True
        parsed = pd.to_datetime(raw_series, format="%d-%m-%Y", errors="coerce")
        if parsed.isna().sum() == len(raw_series):
            # Fallback to general parser
            parsed = pd.to_datetime(raw_series, errors="coerce", dayfirst=True)

        n_unparseable = parsed.isna().sum() - raw_series.isna().sum()
        parsed_dates[col] = parsed
        valid_dates = parsed.dropna()

        print(f"    Unparseable values:    {n_unparseable}")
        print(f"    Valid parsed dates:    {len(valid_dates):,}")
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            span_days = (max_date - min_date).days
            print(f"    Earliest Date:         {min_date.strftime('%Y-%m-%d')}")
            print(f"    Latest Date:           {max_date.strftime('%Y-%m-%d')}")
            print(f"    Date Span:             {span_days} days (~{span_days / 30.4:.1f} months)")

    # 6. Categorical Columns: Region, product_main_category, product_subcategory, Brand
    print("\n--- 6. CATEGORICAL COLUMNS ANALYSIS ---")
    cat_cols = ["Region", "product_main_category", "product_subcategory", "Brand"]
    for col in cat_cols:
        if col not in raw_df.columns:
            print(f"  [WARN] Categorical column '{col}' missing!")
            continue

        raw_cat = raw_df[col]
        # Inspect string whitespace issues
        cleaned_cat = raw_cat.astype(str).str.strip()
        n_unique_raw = raw_cat.nunique(dropna=True)
        n_unique_clean = cleaned_cat.nunique()
        print(f"\n  Column '{col}':")
        print(f"    Raw unique values:     {n_unique_raw}")
        print(f"    Cleaned unique values: {n_unique_clean} (after strip whitespace)")
        print(f"    Value counts (top 5):")
        top5 = cleaned_cat.value_counts().head(5)
        for val, count in top5.items():
            print(f"      - {val:30s}: {count:5d} ({count / n_rows * 100:.1f}%)")

        if col == "Brand":
            print(f"    Top 10 Brands:")
            top10 = cleaned_cat.value_counts().head(10)
            for b_idx, (brand, count) in enumerate(top10.items(), start=1):
                print(f"      {b_idx:2d}. {brand:25s}: {count:4d} ({count / n_rows * 100:.2f}%)")

    # 7. Distribution of product_rating
    print("\n--- 7. PRODUCT RATING DISTRIBUTION ---")
    if "product_rating" in raw_df.columns:
        pr_series = raw_df["product_rating"]
        pr_numeric = pd.to_numeric(pr_series, errors="coerce")
        n_missing_pr = pr_numeric.isna().sum()
        valid_pr = pr_numeric.dropna()
        print(f"  Total rows:              {len(pr_series)}")
        print(f"  Missing / NaN ratings:   {n_missing_pr} ({n_missing_pr / n_rows * 100:.2f}%)")
        print(f"  Valid ratings:           {len(valid_pr):,}")
        if len(valid_pr) > 0:
            print(f"  Min rating:              {valid_pr.min()}")
            print(f"  Max rating:              {valid_pr.max()}")
            print(f"  Mean rating:             {valid_pr.mean():.2f}")
            print(f"  Median rating:           {valid_pr.median():.2f}")
            print(f"  Distinct rating values:  {sorted(valid_pr.unique().tolist())}")
            print(f"  Value distribution:")
            for val, cnt in valid_pr.value_counts().sort_index().items():
                print(f"    Rating {val}: {cnt:5d} ({cnt / len(valid_pr) * 100:.1f}%)")
    else:
        print("  [ERROR] 'product_rating' column missing!")

    # 8. Candidate Targets Distribution: discount_pct and delivery_days
    print("\n--- 8. CANDIDATE TARGETS DISTRIBUTION ---")
    # Main target: discount_pct = (retail_price - discounted_price) / retail_price
    if "retail_price" in clean_prices and "discounted_price" in clean_prices:
        rp = clean_prices["retail_price"]
        dp = clean_prices["discounted_price"]
        valid_mask = (rp > 0) & (dp > 0) & (dp <= rp)
        discount_pct = (rp[valid_mask] - dp[valid_mask]) / rp[valid_mask]

        print("  Main Target: discount_pct = (retail_price - discounted_price) / retail_price")
        print(f"    Eligible rows (rp > 0, dp > 0, dp <= rp): {len(discount_pct):,} / {n_rows:,} ({len(discount_pct) / n_rows * 100:.1f}%)")
        print(f"    Min discount:          {discount_pct.min():.4f}")
        print(f"    Max discount:          {discount_pct.max():.4f}")
        print(f"    Mean discount:         {discount_pct.mean():.4f} ({discount_pct.mean() * 100:.1f}%)")
        print(f"    Median discount:       {discount_pct.median():.4f} ({discount_pct.median() * 100:.1f}%)")
        print(f"    Std Dev discount:      {discount_pct.std():.4f}")
        print(f"    25th Percentile:       {discount_pct.quantile(0.25):.4f}")
        print(f"    75th Percentile:       {discount_pct.quantile(0.75):.4f}")
        print(f"    Zero discount rows:    {(discount_pct == 0).sum()} ({(discount_pct == 0).sum() / len(discount_pct) * 100:.2f}%)")

    # Fallback target: delivery_days = Delivered_Date - Order_Date
    if "Order_Date" in parsed_dates and "Delivered_Date" in parsed_dates:
        od = parsed_dates["Order_Date"]
        dd = parsed_dates["Delivered_Date"]
        date_mask = od.notna() & dd.notna()
        delivery_days = (dd[date_mask] - od[date_mask]).dt.days
        valid_delivery = delivery_days[delivery_days >= 0]

        print("\n  Fallback Target: delivery_days = Delivered_Date - Order_Date")
        print(f"    Eligible rows with both dates parsed:      {date_mask.sum():,} / {n_rows:,}")
        print(f"    Negative delivery days (anomalies):        {(delivery_days < 0).sum()}")
        print(f"    Valid non-negative delivery days:          {len(valid_delivery):,}")
        if len(valid_delivery) > 0:
            print(f"    Min delivery days:     {valid_delivery.min()}")
            print(f"    Max delivery days:     {valid_delivery.max()}")
            print(f"    Mean delivery days:    {valid_delivery.mean():.2f}")
            print(f"    Median delivery days:  {valid_delivery.median():.2f}")
            print(f"    Std Dev delivery days: {valid_delivery.std():.2f}")
            print(f"    Value counts summary (top 5):")
            for days, count in valid_delivery.value_counts().head(5).items():
                print(f"      {days:2d} days: {count:5d} ({count / len(valid_delivery) * 100:.1f}%)")

    # Check Decision Rules per Section 4.2
    print("\n--- DECISION RULES EVALUATION (Section 4.2) ---")
    # Rule 1: Missing column or >20% missing
    print("  [Rule 1] Features with >20% missing values:")
    high_missing = [c for c, p in missing_pcts.items() if p > 20.0]
    if high_missing:
        print(f"    Found: {high_missing}")
    else:
        print("    None. All core candidate features have well under 20% missing values.")

    # Rule 2: Order_Date span
    if "Order_Date" in parsed_dates and parsed_dates["Order_Date"].notna().sum() > 0:
        od_valid = parsed_dates["Order_Date"].dropna()
        span_days = (od_valid.max() - od_valid.min()).days
        print(f"  [Rule 2] Order_Date span: {span_days} days (~{span_days / 30.4:.1f} months).")
        if span_days >= 60:
            print("    PASS: Order_Date covers >= 2 months.")
        else:
            print("    FAIL: Order_Date covers less than 2 months.")

    # Rule 3: Usable features count
    print("  [Rule 3] Usable features available: >= 5 features (retail_price, product_rating, Region, product_main_category, BrandGroup, OrderMonth).")

    # Rule 4: discount_pct standard deviation
    if "discount_pct" in locals():
        print(f"  [Rule 4] discount_pct standard deviation: {discount_pct.std():.4f}")
        if discount_pct.std() > 0.05:
            print("    PASS: discount_pct shows healthy variance (std > 0.05). Proceed with discount_pct as primary target!")
        else:
            print("    WARN: discount_pct is nearly constant. Recommend fallback.")

    print("\n" + "=" * 80)
    print(" END OF AUDIT REPORT")
    print("=" * 80)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(raw_df.columns),
        "missing_counts": missing_counts.to_dict(),
        "n_exact_dupes": int(n_exact_dupes),
    }


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/E-Commerce.csv"
    run_data_audit(csv_file)
