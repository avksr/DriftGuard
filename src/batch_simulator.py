"""
Batch Simulator & Drift Injection Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements batch generation and synthetic drift injection per PROJECT_SPEC.md Section 5.2.
         Samples batches of fixed size from the clean pool, injects controlled drift
         (sudden or gradual, across defined sizes and types), and records exact ground truth
         to allow rigorous evaluation of detection delay and ranking precision.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import os
import yaml
import numpy as np
import pandas as pd

from src.data_loader import (
    load_config,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)


@dataclass
class SimulationRun:
    """
    Container storing the full sequence of simulated batches along with exact ground truth.
    """
    run_id: Union[int, str]
    drift_type: str
    drift_size: str
    drift_shape: str
    drift_start_batch: int
    gradual_ramp_batches: int
    affected_features: List[str]
    batches: List[pd.DataFrame] = field(repr=False)
    ground_truth_per_batch: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_drift(self) -> bool:
        """Return True if drift is scheduled in this simulation run."""
        return self.drift_type != "none" and len(self.affected_features) > 0


# -----------------------------------------------------------------------------
# Primitive Drift Injection Functions (Section 5.2)
# -----------------------------------------------------------------------------

def shift_mean(series: pd.Series, factor: float) -> pd.Series:
    """
    Multiplies a numeric feature by a scalar factor (e.g., retail_price * 1.4).
    Guarantees non-negative values for positive inputs.
    
    Args:
        series: Numeric pandas Series.
        factor: Multiplicative factor.
        
    Returns:
        Transformed Series.
    """
    shifted = series.astype(float) * factor
    # For price or strictly positive quantities, ensure values remain > 0
    return shifted.clip(lower=0.01)


def shift_additive(
    series: pd.Series,
    delta: float,
    min_val: float = 1.0,
    max_val: float = 5.0
) -> pd.Series:
    """
    Adds a constant shift to a numeric feature and bounds it within a valid interval.
    Example: product_rating - 0.5, kept inside [1.0, 5.0].
    
    Args:
        series: Numeric pandas Series.
        delta: Additive delta (can be negative).
        min_val: Lower bound constraint.
        max_val: Upper bound constraint.
        
    Returns:
        Transformed Series.
    """
    shifted = series.astype(float) + delta
    return shifted.clip(lower=min_val, upper=max_val)


def change_spread(series: pd.Series, factor: float) -> pd.Series:
    """
    Stretches or compresses numeric values around their mean.
    
    Formula:
        new_val = mean + (val - mean) * factor
        
    Args:
        series: Numeric pandas Series.
        factor: Spread multiplier (> 1 stretches variance, < 1 narrows variance).
        
    Returns:
        Transformed Series.
    """
    mean_val = float(series.mean())
    shifted = mean_val + (series.astype(float) - mean_val) * factor
    return shifted.clip(lower=0.01)


def shift_category_mix(
    series: pd.Series,
    drop: float,
    random_state: Optional[int] = None
) -> pd.Series:
    """
    Reassigns a share of rows from the most frequent category to other categories.
    For example: lower the most common Region's share by 25 percentage points.
    
    Args:
        series: Categorical pandas Series.
        drop: Fraction of total rows (or proportion of mode) to reassign away from mode.
        random_state: Optional seed for reproducible reassignment.
        
    Returns:
        Transformed categorical Series.
    """
    if len(series) == 0:
        return series.copy()

    rng = np.random.RandomState(random_state)
    s = series.astype(str).copy()
    counts = s.value_counts()
    
    if len(counts) <= 1:
        # Cannot shift mix if there is only one category
        return s

    mode_cat = counts.index[0]
    other_cats = counts.index[1:].tolist()

    mode_indices = s[s == mode_cat].index.to_numpy()
    
    # Calculate number of rows to reassign
    # drop represents the percentage-point drop relative to batch size
    n_drop = int(len(s) * drop)
    # Cannot drop more rows than exist for the mode category (keep at least 1)
    n_to_reassign = min(max(0, n_drop), max(0, len(mode_indices) - 1))

    if n_to_reassign > 0:
        chosen_indices = rng.choice(mode_indices, size=n_to_reassign, replace=False)
        new_assignments = rng.choice(other_cats, size=n_to_reassign)
        s.loc[chosen_indices] = new_assignments

    return s


def label_shift(series: pd.Series, factor: float) -> pd.Series:
    """
    Multiplies the target discount_pct by a scalar factor and caps it within [0.0, 1.0].
    Simulates concept drift where input features remain constant but outcome relations shift.
    
    Args:
        series: Numeric target Series (discount_pct).
        factor: Multiplier applied to target.
        
    Returns:
        Transformed target Series.
    """
    shifted = series.astype(float) * factor
    return shifted.clip(lower=0.0, upper=1.0)


# -----------------------------------------------------------------------------
# Batch Generator & Drift Pipeline
# -----------------------------------------------------------------------------

def apply_drift_to_batch(
    batch_df: pd.DataFrame,
    drift_type: str,
    affected_features: List[str],
    multiplier: float,
    config: Dict[str, Any],
    drift_size: str = "medium",
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Inject configured drift into a single batch based on the effective multiplier.
    
    Args:
        batch_df: Uncorrupted batch DataFrame.
        drift_type: One of 'shift_mean', 'shift_additive', 'change_spread',
                    'shift_category_mix', 'label_shift', or 'none'.
        affected_features: List of feature names to modify.
        multiplier: Real number in [0.0, 1.0] representing ramp progress.
        config: Configuration dictionary with drift_sizes parameters.
        drift_size: One of 'small', 'medium', 'large'.
        random_state: Seed for stochastic drift operations.
        
    Returns:
        Drifted batch DataFrame.
    """
    if multiplier <= 0.0 or drift_type == "none" or not affected_features:
        return batch_df.copy()

    df = batch_df.copy()
    drift_params = config.get("drift_sizes", {}).get(drift_size, {
        "price_factor": 1.40,
        "category_share_drop": 0.25,
        "rating_shift": -0.5
    })

    for feat in affected_features:
        if feat not in df.columns:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[feat].dtype)

        if not is_numeric:
            # Categorical feature drift
            target_drop = float(drift_params.get("category_share_drop", 0.25))
            effective_drop = target_drop * multiplier
            df[feat] = shift_category_mix(df[feat], drop=effective_drop, random_state=random_state)
            continue

        if drift_type == "shift_mean":
            target_factor = float(drift_params.get("price_factor", 1.40))
            # Linear interpolation of factor from 1.0 to target_factor
            effective_factor = 1.0 + (target_factor - 1.0) * multiplier
            df[feat] = shift_mean(df[feat], effective_factor)

        elif drift_type == "shift_additive":
            target_delta = float(drift_params.get("rating_shift", -0.5))
            effective_delta = target_delta * multiplier
            df[feat] = shift_additive(df[feat], effective_delta, min_val=1.0, max_val=5.0)

        elif drift_type == "change_spread":
            target_factor = float(drift_params.get("price_factor", 1.40))
            effective_factor = 1.0 + (target_factor - 1.0) * multiplier
            df[feat] = change_spread(df[feat], effective_factor)

        elif drift_type == "shift_category_mix":
            target_drop = float(drift_params.get("category_share_drop", 0.25))
            effective_drop = target_drop * multiplier
            df[feat] = shift_category_mix(df[feat], drop=effective_drop, random_state=random_state)

        elif drift_type == "label_shift":
            target_factor = float(drift_params.get("price_factor", 1.40))
            effective_factor = 1.0 + (target_factor - 1.0) * multiplier
            df[feat] = label_shift(df[feat], effective_factor)

    return df


def sample_batch(
    pool_df: pd.DataFrame,
    batch_size: int = 300,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Sample a single batch without replacement from the clean pool.
    
    Args:
        pool_df: Clean pool DataFrame.
        batch_size: Number of rows in the batch.
        random_state: Seed for reproducible sampling.
        
    Returns:
        Sampled batch DataFrame with reset index.
    """
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(
        len(pool_df),
        size=min(batch_size, len(pool_df)),
        replace=False
    )
    return pool_df.iloc[sample_indices].copy().reset_index(drop=True)


def simulate_run(
    pool_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    run_id: Union[int, str] = 1,
    drift_type: str = "none",
    drift_size: str = "medium",
    drift_shape: str = "sudden",
    drift_start_batch: Optional[int] = None,
    gradual_ramp_batches: Optional[int] = None,
    affected_features: Optional[List[str]] = None,
    n_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> SimulationRun:
    """
    Generate an entire simulation run consisting of sequential batches with synthetic drift.
    
    Batches are drawn from pool_df without replacement within each batch.
    Saves comprehensive ground truth for each batch (start batch, affected features,
    multiplier, drift type) to evaluate monitoring policies and detectors.
    
    Args:
        pool_df: Clean pool DataFrame from data_loader.
        config: Optional configuration dictionary.
        run_id: Unique identifier for the run.
        drift_type: Drift type ('shift_mean', 'shift_additive', 'change_spread',
                    'shift_category_mix', 'label_shift', 'none').
        drift_size: 'small', 'medium', or 'large'.
        drift_shape: 'sudden' or 'gradual'.
        drift_start_batch: Batch index where drift begins (0-indexed). Defaults to config value.
        gradual_ramp_batches: Number of batches to ramp up drift. Defaults to config value.
        affected_features: List of column names to mutate.
        n_batches: Total number of batches in run (default 48).
        batch_size: Rows per batch (default 300).
        random_seed: Seed for batch sampling and perturbation.
        
    Returns:
        SimulationRun object containing batches and ground truth metadata.
    """
    if config is None:
        config = load_config()

    if n_batches is None:
        n_batches = int(config.get("n_batches_per_run", 48))
    if batch_size is None:
        batch_size = int(config.get("batch_size", 300))
    if drift_start_batch is None:
        drift_start_batch = int(config.get("drift_start_batch", 16))
    if gradual_ramp_batches is None:
        gradual_ramp_batches = int(config.get("gradual_ramp_batches", 8))
    if random_seed is None:
        random_seed = int(config.get("random_seed", 42))

    if affected_features is None:
        if drift_type == "label_shift":
            affected_features = [TARGET_COLUMN]
        elif drift_type in ("shift_mean", "change_spread"):
            affected_features = ["retail_price"]
        elif drift_type == "shift_additive":
            affected_features = ["product_rating"]
        elif drift_type == "shift_category_mix":
            affected_features = ["Region"]
        else:
            affected_features = []

    batches: List[pd.DataFrame] = []
    ground_truth_per_batch: List[Dict[str, Any]] = []

    rng = np.random.RandomState(random_seed)

    for b in range(n_batches):
        # 1. Sample clean batch without replacement within the batch
        batch_seed = int(rng.randint(0, 10_000_000))
        clean_batch = sample_batch(pool_df, batch_size=batch_size, random_state=batch_seed)

        # 2. Determine drift multiplier based on shape and start batch
        if drift_type == "none" or b < drift_start_batch:
            multiplier = 0.0
        else:
            if drift_shape == "sudden":
                multiplier = 1.0
            elif drift_shape == "gradual":
                # Linear ramp from 0 to 1 over gradual_ramp_batches
                step = b - drift_start_batch + 1
                multiplier = min(1.0, max(0.0, step / float(gradual_ramp_batches)))
            else:
                multiplier = 1.0

        # 3. Apply drift if multiplier > 0
        is_drifted = multiplier > 0.0
        drifted_batch = apply_drift_to_batch(
            batch_df=clean_batch,
            drift_type=drift_type,
            affected_features=affected_features,
            multiplier=multiplier,
            config=config,
            drift_size=drift_size,
            random_state=batch_seed + 1
        )

        batches.append(drifted_batch)
        ground_truth_per_batch.append({
            "batch_id": b,
            "is_drifted": is_drifted,
            "multiplier": round(multiplier, 4),
            "drift_type": drift_type if is_drifted else "none",
            "drift_size": drift_size if is_drifted else "none",
            "drift_shape": drift_shape if is_drifted else "none",
            "affected_features": affected_features if is_drifted else [],
            "batch_size": len(drifted_batch)
        })

    return SimulationRun(
        run_id=run_id,
        drift_type=drift_type,
        drift_size=drift_size,
        drift_shape=drift_shape,
        drift_start_batch=drift_start_batch,
        gradual_ramp_batches=gradual_ramp_batches,
        affected_features=affected_features,
        batches=batches,
        ground_truth_per_batch=ground_truth_per_batch
    )


# -----------------------------------------------------------------------------
# Demo Script (Section 11.3 Phase 4 Requirement)
# -----------------------------------------------------------------------------

def run_phase4_demo() -> None:
    """
    Demonstration per Section 11.3 Phase 4:
    Generates one 48-batch run with sudden medium drift on retail_price starting
    at batch 16, and prints the detector's report for batches 15 to 18.
    """
    from src.data_loader import prepare_datasets
    from src.drift_detector import detect_drift

    print("=" * 80)
    print(" DRIFTMONITOR: PHASE 4 BATCH SIMULATOR & DRIFT INJECTION DEMO")
    print("=" * 80)

    # 1. Load reference and clean pool
    config = load_config("config.yaml")
    ref_df, pool_df, _ = prepare_datasets("config.yaml", save_to_disk=False)

    print(f"Loaded Reference data: {len(ref_df):,} rows")
    print(f"Loaded Clean Pool:     {len(pool_df):,} rows")

    # 2. Simulate 48-batch run with sudden medium drift on retail_price starting at batch 16
    sim_run = simulate_run(
        pool_df=pool_df,
        config=config,
        run_id="demo_phase_4",
        drift_type="shift_mean",
        drift_size="medium",
        drift_shape="sudden",
        drift_start_batch=16,
        affected_features=["retail_price"]
    )

    print("\nSimulation Run Configured:")
    print(f"  Total Batches:       {len(sim_run.batches)}")
    print(f"  Drift Type:          {sim_run.drift_type}")
    print(f"  Drift Size:          {sim_run.drift_size}")
    print(f"  Drift Shape:         {sim_run.drift_shape}")
    print(f"  Drift Start Batch:   {sim_run.drift_start_batch} (0-indexed)")
    print(f"  Target Feature(s):   {sim_run.affected_features}")
    print("-" * 80)

    # 3. Print detector reports for batches 15 to 18
    print("\nDETECTOR REPORTS FOR BATCHES 15 TO 18:\n")
    for b_idx in range(15, 19):
        batch = sim_run.batches[b_idx]
        gt = sim_run.ground_truth_per_batch[b_idx]
        
        report = detect_drift(ref_df, batch, config=config, batch_id=b_idx)

        print(f"--- BATCH {b_idx} ---")
        print(f"Ground Truth: Drifted={gt['is_drifted']}, Multiplier={gt['multiplier']}, Features={gt['affected_features']}")
        print(f"Detector:     Drift Detected={report['drift_detected']}")
        print(f"Top Ranking:  {report['ranking']}")
        print("Feature Details:")
        for feat, details in report["features"].items():
            flag_str = "FLAGGED" if details["flagged"] else "OK"
            stat_str = f"ks_stat={details.get('ks_stat', '-'):<6} p={details.get('ks_p', details.get('chi2_p', '-')):<8}"
            print(f"  - {feat:<22} | PSI={details['psi']:<6.4f} | {stat_str} | Status={flag_str}")
        print()

    print("=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_phase4_demo()
