"""
src/experiment_c.py

Experiment C – Feature ranking accuracy (Precision@k).

Per PROJECT_SPEC.md Section 7 / Experiment C:
  Goal: can the ranking find the columns we deliberately changed?
  Scenarios: change 1 feature; change 2 features; change 3 features.
  Metrics:
    - Precision@k  = (# of truly-changed features in top-k ranked features) / k
      where k = number of truly changed features
    - Whether the changed feature is ranked #1 in single-feature cases

The experiment runs across drift sizes and shapes to show how well
the PSI-based ranking generalises.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.data_loader import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.drift_detector import detect_drift


# Scenarios: each entry specifies which features are deliberately drifted
# and how. We use the drift_start_batch from config, then look at the first
# batch that actually received drift (batch index == drift_start_batch).

_SCENARIOS: List[Dict[str, Any]] = [
    # ── Single feature ────────────────────────────────────────────────────
    {"changed": ["retail_price"],
     "drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
     "label": "1feat_price_medium_sudden"},
    {"changed": ["retail_price"],
     "drift_type": "shift_mean", "drift_size": "large", "drift_shape": "sudden",
     "label": "1feat_price_large_sudden"},
    {"changed": ["product_rating"],
     "drift_type": "shift_additive", "drift_size": "medium", "drift_shape": "sudden",
     "label": "1feat_rating_medium_sudden"},
    {"changed": ["Region"],
     "drift_type": "shift_category_mix", "drift_size": "medium", "drift_shape": "sudden",
     "label": "1feat_region_medium_sudden"},

    # ── Two features ──────────────────────────────────────────────────────
    {"changed": ["retail_price", "product_rating"],
     "drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
     "label": "2feat_price_rating_medium_sudden"},

    # ── Three features ────────────────────────────────────────────────────
    {"changed": ["retail_price", "product_rating", "Region"],
     "drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "sudden",
     "label": "3feat_price_rating_region_medium_sudden"},
]


def _precision_at_k(ranked: List[str], truly_changed: List[str]) -> float:
    """Return the fraction of truly-changed features found in the top-k ranked list.

    k is the number of truly changed features.
    """
    k = len(truly_changed)
    if k == 0:
        return float("nan")
    top_k = ranked[:k]
    hits = sum(1 for f in top_k if f in truly_changed)
    return hits / k


def _rank_one_correct(ranked: List[str], truly_changed: List[str]) -> bool:
    """Return True if the single changed feature is ranked #1."""
    if len(truly_changed) != 1:
        return False
    return len(ranked) > 0 and ranked[0] == truly_changed[0]


def run(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Entry point called by run_experiments.py once per seed."""
    seed = cfg.get("_current_seed", cfg.get("random_seed", 42))

    ref_df, pool_df, _ = load_experiment_data(cfg)
    drift_start = int(cfg.get("drift_start_batch", 16))

    records: List[Dict[str, Any]] = []

    for scenario in _SCENARIOS:
        truly_changed: List[str] = scenario["changed"]
        n_changed = len(truly_changed)

        # For multi-feature drift we apply drift_type to the first changed feature
        # (shift_mean / shift_additive / shift_category_mix per feature type).
        sim = simulate_run(
            pool_df=pool_df,
            config=cfg,
            run_id=seed,
            drift_type=scenario["drift_type"],
            drift_size=scenario["drift_size"],
            drift_shape=scenario["drift_shape"],
            affected_features=truly_changed,
            random_seed=seed,
        )

        # Evaluate ranking on the FIRST batch that should be fully drifted
        target_batch_idx = drift_start
        if target_batch_idx >= len(sim.batches):
            target_batch_idx = len(sim.batches) - 1

        batch_df = sim.batches[target_batch_idx]
        report = detect_drift(ref_df, batch_df, cfg, batch_id=target_batch_idx)

        ranked: List[str] = report.get("ranking", [])
        prec_at_k = _precision_at_k(ranked, truly_changed)
        rank1_correct = _rank_one_correct(ranked, truly_changed) if n_changed == 1 else None

        records.append({
            "experiment": "C",
            "scenario": scenario["label"],
            "n_changed_features": n_changed,
            "changed_features": ",".join(truly_changed),
            "drift_type": scenario["drift_type"],
            "drift_size": scenario["drift_size"],
            "drift_shape": scenario["drift_shape"],
            "seed": seed,
            "precision_at_k": round(prec_at_k, 4) if not np.isnan(prec_at_k) else np.nan,
            "rank1_correct": int(rank1_correct) if rank1_correct is not None else np.nan,
            "top_ranked_feature": ranked[0] if ranked else "",
            "full_ranking": ",".join(ranked[:6]),  # top-6 for inspection
        })

    return records
