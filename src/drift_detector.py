"""
Drift Detection Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements statistical data drift detection per PROJECT_SPEC.md Section 5.4.
         Computes two-sample Kolmogorov-Smirnov (KS) test for numeric features,
         Population Stability Index (PSI) for numeric and categorical features,
         and Chi-Square contingency test for categorical features.
         Ranks features by drift magnitude and determines batch-level drift flags.
         Uses only SciPy and NumPy without any third-party drift libraries.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml.
        
    Returns:
        Dictionary of configuration parameters.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ks_test(
    ref_values: Union[np.ndarray, pd.Series, List[float]],
    new_values: Union[np.ndarray, pd.Series, List[float]]
) -> Tuple[float, float]:
    """
    Compute two-sample Kolmogorov-Smirnov test for numeric values (Section 3.5 & 5.4).
    Measures the maximum distance between cumulative distributions and calculates p-value.
    
    Args:
        ref_values: Reference numeric sample.
        new_values: New incoming batch numeric sample.
        
    Returns:
        Tuple of (ks_statistic, p_value), where statistic in [0.0, 1.0].
    """
    ref_arr = np.asarray(ref_values, dtype=float)
    new_arr = np.asarray(new_values, dtype=float)

    # Filter out NaNs
    ref_clean = ref_arr[~np.isnan(ref_arr)]
    new_clean = new_arr[~np.isnan(new_arr)]

    if len(ref_clean) == 0 or len(new_clean) == 0:
        return 0.0, 1.0

    res = ks_2samp(ref_clean, new_clean)
    return float(res.statistic), float(res.pvalue)


def psi_numeric(
    ref_values: Union[np.ndarray, pd.Series, List[float]],
    new_values: Union[np.ndarray, pd.Series, List[float]],
    n_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Compute Population Stability Index (PSI) for numeric values (Section 3.6 & 5.4).
    
    Bin edges are computed strictly from reference quantiles (never from the new batch).
    Duplicate quantile edges are resolved via deduplication.
    Out-of-range values in the new batch are handled by unbounded outer edges [-inf, inf].
    Empty bins receive a small epsilon smoothing (1e-4) to prevent log(0) and division by zero.
    
    Formula:
        PSI = sum_i ((new_share_i - ref_share_i) * ln(new_share_i / ref_share_i))
    
    Args:
        ref_values: Reference numeric sample.
        new_values: New incoming batch numeric sample.
        n_bins: Target number of quantile bins (default 10).
        epsilon: Smoothing offset added to avoid division by zero / log(0).
        
    Returns:
        Non-negative float representing the PSI score.
    """
    ref_arr = np.asarray(ref_values, dtype=float)
    new_arr = np.asarray(new_values, dtype=float)

    ref_clean = ref_arr[~np.isnan(ref_arr)]
    new_clean = new_arr[~np.isnan(new_arr)]

    if len(ref_clean) == 0 or len(new_clean) == 0:
        return 0.0

    # 1. Determine bin edges strictly from reference quantiles
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw_edges = np.quantile(ref_clean, quantiles)
    unique_edges = np.unique(raw_edges)

    # Handle degenerate case where all reference values are identical
    if len(unique_edges) < 2:
        val = unique_edges[0]
        # Bins: (-inf, val - eps], (val - eps, val + eps], (val + eps, inf)
        unique_edges = np.array([-np.inf, val - 1e-5, val + 1e-5, np.inf])
    else:
        # Extend boundaries to infinity to capture all possible batch values
        unique_edges[0] = -np.inf
        unique_edges[-1] = np.inf

    # 2. Bin both reference and new values into the same bins
    ref_binned = pd.cut(ref_clean, bins=unique_edges, include_lowest=True)
    new_binned = pd.cut(new_clean, bins=unique_edges, include_lowest=True)

    # 3. Compute bin counts and shares
    ref_counts = pd.Series(ref_binned).value_counts(sort=False).to_numpy(dtype=float)
    new_counts = pd.Series(new_binned).value_counts(sort=False).to_numpy(dtype=float)

    ref_share = ref_counts / len(ref_clean)
    new_share = new_counts / len(new_clean)

    # 4. Apply epsilon smoothing and renormalize
    ref_share = np.maximum(ref_share, epsilon)
    new_share = np.maximum(new_share, epsilon)
    ref_share = ref_share / np.sum(ref_share)
    new_share = new_share / np.sum(new_share)

    # 5. Compute PSI
    psi_terms = (new_share - ref_share) * np.log(new_share / ref_share)
    psi = float(np.sum(psi_terms))

    return max(0.0, psi)


def psi_categorical(
    ref_series: Union[pd.Series, np.ndarray, List[Any]],
    new_series: Union[pd.Series, np.ndarray, List[Any]],
    epsilon: float = 1e-4
) -> float:
    """
    Compute Population Stability Index (PSI) for categorical features (Section 3.6 & 5.4).
    Each distinct category represents a bin.
    
    Args:
        ref_series: Reference categorical sample.
        new_series: New incoming batch categorical sample.
        epsilon: Smoothing offset to prevent log(0).
        
    Returns:
        Non-negative float representing categorical PSI score.
    """
    ref_s = pd.Series(ref_series).dropna().astype(str)
    new_s = pd.Series(new_series).dropna().astype(str)

    if len(ref_s) == 0 or len(new_s) == 0:
        return 0.0

    all_categories = sorted(list(set(ref_s.unique()).union(set(new_s.unique()))))

    ref_counts = ref_s.value_counts()
    new_counts = new_s.value_counts()

    ref_share = np.array([ref_counts.get(cat, 0) / len(ref_s) for cat in all_categories], dtype=float)
    new_share = np.array([new_counts.get(cat, 0) / len(new_s) for cat in all_categories], dtype=float)

    ref_share = np.maximum(ref_share, epsilon)
    new_share = np.maximum(new_share, epsilon)
    ref_share = ref_share / np.sum(ref_share)
    new_share = new_share / np.sum(new_share)

    psi_terms = (new_share - ref_share) * np.log(new_share / ref_share)
    psi = float(np.sum(psi_terms))

    return max(0.0, psi)


def chi_square_categorical(
    ref_series: Union[pd.Series, np.ndarray, List[Any]],
    new_series: Union[pd.Series, np.ndarray, List[Any]]
) -> float:
    """
    Compute Chi-Square test p-value for categorical distributions (Section 5.4).
    Acts as the categorical counterpart of the KS p-value.
    
    Constructs a 2 x K contingency table of category counts for reference and batch.
    
    Args:
        ref_series: Reference categorical values.
        new_series: New batch categorical values.
        
    Returns:
        p-value in [0.0, 1.0].
    """
    ref_s = pd.Series(ref_series).dropna().astype(str)
    new_s = pd.Series(new_series).dropna().astype(str)

    if len(ref_s) == 0 or len(new_s) == 0:
        return 1.0

    all_categories = sorted(list(set(ref_s.unique()).union(set(new_s.unique()))))

    # If only one category exists in total, no divergence can occur
    if len(all_categories) <= 1:
        return 1.0

    ref_counts = ref_s.value_counts()
    new_counts = new_s.value_counts()

    table = [
        [int(ref_counts.get(cat, 0)) for cat in all_categories],
        [int(new_counts.get(cat, 0)) for cat in all_categories]
    ]

    res = chi2_contingency(table)
    return float(res.pvalue)


def detect_drift(
    ref_df: pd.DataFrame,
    batch_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    batch_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run full drift detection pipeline on an incoming batch against reference data (Section 5.4).
    
    Calculates KS and PSI for numeric features, and Chi-Square and PSI for categorical features.
    Flags features according to configured rule ('both', 'ks_only', 'psi_only').
    Ranks features by PSI score (highest first), breaking ties with KS / Chi-square significance.
    Determines overall batch-level drift flag.
    
    Args:
        ref_df: Baseline reference DataFrame.
        batch_df: Incoming batch DataFrame.
        config: Optional configuration dictionary. Loaded from config.yaml if None.
        batch_id: Optional identifier for the current batch.
        
    Returns:
        Structured dictionary matching PROJECT_SPEC.md Section 5.4:
        {
            "batch_id": int or None,
            "features": {
                "retail_price": {"ks_stat": 0.31, "ks_p": 0.0001, "psi": 0.42, "flagged": True},
                "Region": {"chi2_p": 0.40, "psi": 0.03, "flagged": False}
            },
            "ranking": ["retail_price", "product_main_category", ...],
            "drift_detected": bool
        }
    """
    if config is None:
        config = load_config()

    psi_bins = int(config.get("psi_bins", 10))
    psi_flag_threshold = float(config.get("psi_flag_threshold", 0.25))
    p_value_threshold = float(config.get("p_value_threshold", 0.01))
    drift_rule = str(config.get("drift_rule", "both")).lower()

    # Canonical feature definitions from Section 4.4
    default_numeric = ["retail_price", "product_rating"]
    default_categorical = ["Region", "product_main_category", "BrandGroup", "OrderMonth"]

    # Identify features available in both reference and batch
    common_cols = [c for c in ref_df.columns if c in batch_df.columns]
    
    # Exclude non-feature columns
    excluded_cols = {"discount_pct", "discounted_price", "Customer_uniq_id", "Order_Date", "delivery_days", "dataset_split"}
    feature_candidates = [c for c in common_cols if c not in excluded_cols]

    features_report: Dict[str, Dict[str, Any]] = {}

    for col in feature_candidates:
        is_numeric = col in default_numeric or pd.api.types.is_numeric_dtype(ref_df[col].dtype)
        
        if is_numeric:
            stat, p_val = ks_test(ref_df[col], batch_df[col])
            psi = psi_numeric(ref_df[col], batch_df[col], n_bins=psi_bins)

            # Apply flagging rule
            if drift_rule == "both":
                flagged = (psi >= psi_flag_threshold) and (p_val < p_value_threshold)
            elif drift_rule == "ks_only":
                flagged = (p_val < p_value_threshold)
            elif drift_rule == "psi_only":
                flagged = (psi >= psi_flag_threshold)
            else:
                flagged = (psi >= psi_flag_threshold) and (p_val < p_value_threshold)

            features_report[col] = {
                "ks_stat": round(stat, 4),
                "ks_p": round(p_val, 4),
                "psi": round(psi, 4),
                "flagged": bool(flagged)
            }
        else:
            p_val = chi_square_categorical(ref_df[col], batch_df[col])
            psi = psi_categorical(ref_df[col], batch_df[col])

            # Apply flagging rule
            if drift_rule == "both":
                flagged = (psi >= psi_flag_threshold) and (p_val < p_value_threshold)
            elif drift_rule == "ks_only":
                flagged = (p_val < p_value_threshold)
            elif drift_rule == "psi_only":
                flagged = (psi >= psi_flag_threshold)
            else:
                flagged = (psi >= psi_flag_threshold) and (p_val < p_value_threshold)

            features_report[col] = {
                "chi2_p": round(p_val, 4),
                "psi": round(psi, 4),
                "flagged": bool(flagged)
            }

    # Ranking: sort by PSI descending; break ties with KS stat (or 1 - chi2_p for categorical)
    def ranking_key(feat_name: str) -> Tuple[float, float]:
        f_info = features_report[feat_name]
        psi_val = f_info["psi"]
        tie_break = f_info.get("ks_stat", 1.0 - f_info.get("chi2_p", 1.0))
        return (psi_val, tie_break)

    ranking = sorted(features_report.keys(), key=ranking_key, reverse=True)

    # Batch-level drift: True if at least one feature is flagged
    drift_detected = any(f_info["flagged"] for f_info in features_report.values())

    return {
        "batch_id": batch_id,
        "features": features_report,
        "ranking": ranking,
        "drift_detected": drift_detected
    }
