"""
Data Quality Checks Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements batch quality validation per PROJECT_SPEC.md Section 5.3.
         Checks schema, data types, missing rates, price validity, and unseen category ratios.
         Batches failing quality checks are marked invalid to prevent corrupted data
         from being mistaken for statistical drift.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import yaml
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml.
        
    Returns:
        Dictionary containing configuration settings.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def check_batch_quality(
    batch_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    ref_df: Optional[pd.DataFrame] = None,
    ref_categories: Optional[Dict[str, List[Any]]] = None
) -> Dict[str, Any]:
    """
    Validate a single incoming batch against quality criteria per Section 5.3:
    1. Expected columns exist.
    2. Data types are valid (numeric features have numeric dtypes).
    3. Missing-value share is strictly below the configured threshold.
    4. No non-positive prices (retail_price must be > 0).
    5. Price / discount consistency (discounted_price <= retail_price, or discount_pct in [0, 1]).
    6. Proportion of unseen category labels is within the allowable tolerance.

    Args:
        batch_df: The incoming batch DataFrame to evaluate.
        config: Optional configuration dictionary. If None, loaded from config.yaml.
        ref_df: Optional reference DataFrame used to extract known categories if ref_categories is not supplied.
        ref_categories: Optional precomputed dictionary of known categories per column.

    Returns:
        Dictionary with keys:
            - 'passed': bool, True if all checks passed, False otherwise.
            - 'issues': List[str], human-readable descriptions of failed criteria.
            - 'batch_size': int, number of rows in the batch.
            - 'checks_detail': Dict[str, bool], pass/fail flag for each specific check.
    """
    if config is None:
        config = load_config()

    quality_cfg = config.get("quality", {})
    max_missing_fraction = quality_cfg.get("max_missing_fraction", 0.10)
    unseen_tolerance = quality_cfg.get("unseen_category_tolerance", 0.05)
    expected_cols = quality_cfg.get("expected_features", [
        "retail_price", "product_rating", "Region", "product_main_category", "BrandGroup", "OrderMonth"
    ])

    issues: List[str] = []
    checks_detail: Dict[str, bool] = {
        "expected_columns": True,
        "data_types": True,
        "missing_values": True,
        "price_validity": True,
        "discount_validity": True,
        "unseen_categories": True
    }

    # Edge Case: Check for empty batch
    if batch_df is None or len(batch_df) == 0:
        return {
            "passed": False,
            "issues": ["Batch is empty (0 rows)"],
            "batch_size": 0,
            "checks_detail": {k: False for k in checks_detail}
        }

    # 1. Expected columns check
    missing_columns = [col for col in expected_cols if col not in batch_df.columns]
    if missing_columns:
        issues.append(f"Missing expected column(s): {missing_columns}")
        checks_detail["expected_columns"] = False

    # 2. Data types check for numeric features
    numeric_expected = [c for c in ["retail_price", "product_rating", "discount_pct", "discounted_price"] if c in batch_df.columns]
    for col in numeric_expected:
        if not pd.api.types.is_numeric_dtype(batch_df[col].dtype):
            issues.append(f"Column '{col}' has invalid data type '{batch_df[col].dtype}'; expected numeric.")
            checks_detail["data_types"] = False

    # 3. Missing-value share check
    for col in batch_df.columns:
        missing_rate = float(batch_df[col].isna().mean())
        if missing_rate > max_missing_fraction:
            issues.append(
                f"Column '{col}' missing-value rate {missing_rate:.1%} exceeds limit {max_missing_fraction:.1%}."
            )
            checks_detail["missing_values"] = False

    # 4. Price validity (retail_price > 0)
    if "retail_price" in batch_df.columns and pd.api.types.is_numeric_dtype(batch_df["retail_price"].dtype):
        non_positive_count = int((batch_df["retail_price"] <= 0).sum())
        if non_positive_count > 0:
            issues.append(f"Found {non_positive_count} rows with retail_price <= 0.")
            checks_detail["price_validity"] = False

    # 5. Discount validity (discounted_price <= retail_price or discount_pct in [0, 1])
    if "discounted_price" in batch_df.columns and "retail_price" in batch_df.columns:
        if pd.api.types.is_numeric_dtype(batch_df["discounted_price"].dtype) and pd.api.types.is_numeric_dtype(batch_df["retail_price"].dtype):
            inconsistent_count = int((batch_df["discounted_price"] > batch_df["retail_price"]).sum())
            if inconsistent_count > 0:
                issues.append(f"Found {inconsistent_count} rows where discounted_price > retail_price.")
                checks_detail["discount_validity"] = False

    if "discount_pct" in batch_df.columns and pd.api.types.is_numeric_dtype(batch_df["discount_pct"].dtype):
        out_of_bounds = int(((batch_df["discount_pct"] < 0.0) | (batch_df["discount_pct"] > 1.0)).sum())
        if out_of_bounds > 0:
            issues.append(f"Found {out_of_bounds} rows with discount_pct outside valid range [0.0, 1.0].")
            checks_detail["discount_validity"] = False

    # 6. Unseen category labels check
    # Build known categories from ref_categories or ref_df if available
    known_cats: Dict[str, set] = {}
    if ref_categories:
        known_cats = {k: set(v) for k, v in ref_categories.items()}
    elif ref_df is not None:
        cat_cols = [c for c in ["Region", "product_main_category", "BrandGroup", "OrderMonth"] if c in ref_df.columns]
        for c in cat_cols:
            known_cats[c] = set(ref_df[c].dropna().unique())

    if known_cats:
        for col, known_set in known_cats.items():
            if col in batch_df.columns:
                batch_vals = batch_df[col].dropna()
                if len(batch_vals) > 0:
                    unseen_mask = ~batch_vals.isin(known_set)
                    unseen_rate = float(unseen_mask.mean())
                    if unseen_rate > unseen_tolerance:
                        issues.append(
                            f"Column '{col}' has {unseen_rate:.1%} unseen category rows, "
                            f"exceeding tolerance of {unseen_tolerance:.1%}."
                        )
                        checks_detail["unseen_categories"] = False

    passed = len(issues) == 0
    return {
        "passed": passed,
        "issues": issues,
        "batch_size": len(batch_df),
        "checks_detail": checks_detail
    }
