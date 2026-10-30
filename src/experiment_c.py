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

Evaluates ranking across several post-drift batches (5 batches):
  - Sudden drift: first 5 batches from drift start
  - Gradual drift: first 5 batches after the ramp finishes
Reports precision@k as mean and standard deviation over these batches.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from src._exp_utils import load_experiment_data
from src.batch_simulator import simulate_run
from src.drift_detector import detect_drift


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

    # ── Single feature gradual ────────────────────────────────────────────
    {"changed": ["retail_price"],
     "drift_type": "shift_mean", "drift_size": "medium", "drift_shape": "gradual",
     "label": "1feat_price_medium_gradual"},

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
    """Return the fraction of truly-changed features found in the top-k ranked list."""
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
    gradual_ramp = int(cfg.get("gradual_ramp_batches", 8))

    records: List[Dict[str, Any]] = []

    for scenario in _SCENARIOS:
        truly_changed: List[str] = scenario["changed"]
        n_changed = len(truly_changed)

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

        # For sudden drift: first 5 batches from drift_start
        # For gradual drift: first 5 batches after ramp finishes
        if scenario["drift_shape"] == "gradual":
            start_eval_idx = drift_start + gradual_ramp
        else:
            start_eval_idx = drift_start

        total_b = len(sim.batches)
        eval_indices = [
            b for b in range(start_eval_idx, start_eval_idx + 5) if b < total_b
        ]
        if not eval_indices:
            eval_indices = [max(0, total_b - 1)]

        batch_precs: List[float] = []
        batch_r1s: List[float] = []
        first_ranked = None

        for target_idx in eval_indices:
            batch_df = sim.batches[target_idx]
            report = detect_drift(ref_df, batch_df, cfg, batch_id=target_idx)
            ranked = report.get("ranking", [])
            if first_ranked is None:
                first_ranked = ranked
            prec = _precision_at_k(ranked, truly_changed)
            batch_precs.append(prec)
            if n_changed == 1:
                batch_r1s.append(1.0 if _rank_one_correct(ranked, truly_changed) else 0.0)

        mean_prec = float(np.mean(batch_precs))
        std_prec = float(np.std(batch_precs)) if len(batch_precs) > 1 else 0.0
        mean_r1 = float(np.mean(batch_r1s)) if batch_r1s else np.nan

        records.append({
            "experiment": "C",
            "scenario": scenario["label"],
            "n_changed_features": n_changed,
            "changed_features": ",".join(truly_changed),
            "drift_type": scenario["drift_type"],
            "drift_size": scenario["drift_size"],
            "drift_shape": scenario["drift_shape"],
            "seed": seed,
            "precision_at_k": round(mean_prec, 4) if not np.isnan(mean_prec) else np.nan,
            "precision_at_k_std": round(std_prec, 4),
            "rank1_correct": round(mean_r1, 4) if not np.isnan(mean_r1) else np.nan,
            "n_eval_batches": len(eval_indices),
            "top_ranked_feature": first_ranked[0] if first_ranked else "",
            "full_ranking": ",".join(first_ranked[:6]) if first_ranked else "",
        })

    return records
