"""
src/_exp_utils.py

Shared utilities for experiment modules A-D.
Loads the pre-processed reference/pool DataFrames from the cached CSVs
(written by data_loader.prepare_datasets) to avoid re-running the full
data pipeline on every experiment repeat.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


_CACHED_DATA: Dict[str, Any] = {}  # module-level cache so data is loaded once per process


def load_experiment_data(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Return (ref_df, pool_df, ref_artifacts) using cached CSVs when available.

    Falls back to running the full pipeline if the CSVs are missing.
    The results are cached in-process so subsequent calls within the same
    experiment run are instant.
    """
    global _CACHED_DATA
    if "ref_df" in _CACHED_DATA:
        return _CACHED_DATA["ref_df"], _CACHED_DATA["pool_df"], _CACHED_DATA["artifacts"]

    ref_path = Path("data/reference_data.csv")
    pool_path = Path("data/clean_pool.csv")
    artifacts_path = Path("data/reference_artifacts.json")

    if ref_path.is_file() and pool_path.is_file():
        ref_df = pd.read_csv(ref_path)
        pool_df = pd.read_csv(pool_path)
        artifacts: Dict[str, Any] = {}
        if artifacts_path.is_file():
            with artifacts_path.open() as f:
                artifacts = json.load(f)
    else:
        # Full pipeline fallback (slower)
        from src.data_loader import prepare_datasets
        ref_df, pool_df, artifacts = prepare_datasets("config.yaml", save_to_disk=True)

    _CACHED_DATA["ref_df"] = ref_df
    _CACHED_DATA["pool_df"] = pool_df
    _CACHED_DATA["artifacts"] = artifacts
    return ref_df, pool_df, artifacts
