"""
Legacy interactive phase demos (Phase 6 and Phase 7).
Moved out of library source modules to maintain clean production boundaries.
"""

import os
import sys
import json
from unittest.mock import patch
import pandas as pd

from src.utils import load_config
from src.decision_agent import DecisionAgent
from src.data_loader import prepare_datasets
from src.batch_simulator import simulate_run
from src.baseline_model import load_baseline_model
from src.retrainer import retrain_model


def run_phase6_demo() -> None:
    """
    Demonstrates the Hybrid Decision Agent under key incident scenarios:
    1. Incident with No API Key (Template fallback)
    2. Incident with Guardrail Override (LLM picks unauthorized action)
    3. Severe Drift + Degraded Performance (Recommending PROPOSE_RETRAIN)
    """
    print("=" * 85)
    print(" DRIFTMONITOR: PHASE 6 HYBRID DECISION AGENT DEMONSTRATION")
    print("=" * 85)

    agent = DecisionAgent(use_llm=True)

    # Scenario 1: Drift detected, labels unavailable (pure template fallback, no API key)
    print("\n--- SCENARIO 1: Drift Detected & Labels Delayed (No API Key Fallback) ---")
    quality1 = {"passed": True, "issues": []}
    drift1 = {
        "drift_detected": True,
        "ranking": ["retail_price", "product_main_category"],
        "features": {
            "retail_price": {"psi": 0.3845, "ks_stat": 0.28, "ks_p": 0.0001, "flagged": True},
            "product_main_category": {"psi": 0.08, "chi2_p": 0.15, "flagged": False}
        }
    }
    perf1 = {"status": "unverified", "batch_mae": None, "baseline_mae": 0.1432}

    rep1 = agent.decide(batch_id=16, quality_status=quality1, drift_report=drift1, performance_record=perf1)
    print(f"Situation:            Drift confirmed, labels not yet arrived")
    print(f"Allowed Actions:      {rep1.allowed_actions}")
    print(f"Default Action:       {rep1.default_action}")
    print(f"Chosen Action:        {rep1.chosen_action}")
    print(f"Used LLM:             {rep1.used_llm}")
    print(f"Guardrail Overridden: {rep1.guardrail_overridden}")
    print(f"Summary:              {rep1.summary}")
    print(f"Observed Facts:       {rep1.observed_facts}")

    # Scenario 2: Drift detected + performance degraded (Propose Retraining)
    print("\n--- SCENARIO 2: Drift Detected & Sustained Degradation (PROPOSE_RETRAIN) ---")
    drift2 = {
        "drift_detected": True,
        "ranking": ["retail_price"],
        "features": {"retail_price": {"psi": 0.4210, "ks_stat": 0.32, "ks_p": 0.0, "flagged": True}}
    }
    perf2 = {"status": "degraded", "batch_mae": 0.2245, "baseline_mae": 0.1432}

    rep2 = agent.decide(batch_id=18, quality_status=quality1, drift_report=drift2, performance_record=perf2)
    print(f"Situation:            Drift confirmed & consecutive degraded batches")
    print(f"Allowed Actions:      {rep2.allowed_actions}")
    print(f"Chosen Action:        {rep2.chosen_action}")
    print(f"Summary:              {rep2.summary}")
    print(f"Follow-up Checks:     {rep2.follow_up_checks}")

    # Scenario 3: Guardrail intervention when an LLM proposes a forbidden action
    print("\n--- SCENARIO 3: Guardrail Intervention (Disallowed Action Blocked) ---")
    fake_llm_output = json.dumps({
        "summary": "Everything looks fine, no action needed.",
        "observed_facts": ["Price changed"],
        "possible_explanations": ["Normal fluctuation"],
        "recommended_action": "CONTINUE",  # FORBIDDEN!
        "follow_up_checks": []
    })
    
    with patch("src.decision_agent.call_llm", return_value=fake_llm_output):
        rep3 = agent.decide(batch_id=19, quality_status=quality1, drift_report=drift2, performance_record=perf2)

    print(f"LLM Proposed Action:  'CONTINUE' (Forbidden when degraded)")
    print(f"Guardrail Overridden: {rep3.guardrail_overridden}")
    print(f"Enforced Action:      {rep3.chosen_action}")
    print(f"Summary:              {rep3.summary}")

    print("\n" + "=" * 85)
    print(" PHASE 6 DEMONSTRATION COMPLETE")
    print("=" * 85)


def run_phase7_demo() -> None:
    """
    Demonstrates controlled retraining on real project data with drift injection.
    """
    print("=" * 85)
    print(" DRIFTMONITOR: PHASE 7 CONTROLLED RETRAINING DEMONSTRATION")
    print("=" * 85)

    config = load_config("config.yaml")

    ref_df, pool_df, _ = prepare_datasets("config.yaml", save_to_disk=False)
    ref_val_path = config.get("model", {}).get("ref_val_path", "models/reference_val_data.csv")
    model_path = config.get("model", {}).get("model_path", "models/baseline_model.joblib")

    if not os.path.exists(ref_val_path) or not os.path.exists(model_path):
        print("Pre-trained baseline model or reference validation split not found.")
        return

    ref_val_df = pd.read_csv(ref_val_path)
    current_model = load_baseline_model(model_path)

    from sklearn.model_selection import train_test_split
    train_fraction = config.get("data", {}).get("baseline_train_fraction", 0.8)
    seed = config.get("random_seed", 42)
    ref_train_df, _ = train_test_split(
        ref_df,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True
    )

    print(f"Reference Training Data:     {len(ref_train_df):,} samples")
    print(f"Reference Validation Data:   {len(ref_val_df):,} samples (Held-out)")
    print(f"Active Baseline Model:       {model_path}")

    sim_run = simulate_run(
        pool_df=pool_df,
        config=config,
        run_id="demo_retrain",
        drift_type="shift_mean",
        drift_size="large",
        drift_shape="sudden",
        drift_start_batch=0,
        affected_features=["retail_price"],
        n_batches=4,
        batch_size=300
    )

    result = retrain_model(
        current_model=current_model,
        ref_train_df=ref_train_df,
        ref_val_df=ref_val_df,
        recent_batches=sim_run.batches,
        config=config,
        val_batches_count=2,
        reason="Agent proposed retraining due to sustained error degradation"
    )

    print("\n" + "-" * 85)
    print(" RETRAINING EVALUATION SUMMARY")
    print("-" * 85)
    print(f"Decision:                     {result.decision} (accepted={result.accepted})")
    print(f"Reason:                       {result.reason}")
    print(f"Retraining Data Size:         {result.n_retrain_samples:,} samples")
    print(f"Recent Validation Window:     {result.n_recent_val_samples:,} samples")
    print("\nRecent Drifted Window Performance:")
    print(f"  - Current Model MAE:        {result.current_recent_mae:.4f}")
    print(f"  - Candidate Model MAE:      {result.candidate_recent_mae:.4f}")
    print(f"  - Relative Improvement:     {result.recent_improvement_pct:+.2f}% (Threshold: >= +5.0%)")
    print("\nOld Reference Data Performance (Forgetting Guardrail):")
    print(f"  - Current Model MAE:        {result.current_old_mae:.4f}")
    print(f"  - Candidate Model MAE:      {result.candidate_old_mae:.4f}")
    print(f"  - Old Data Regression:      {result.old_data_regression_pct:+.2f}% (Threshold: <= +10.0%)")
    print("-" * 85)
    print(f"Active Model Deployed:        {'Candidate Model' if result.accepted else 'Original Baseline Model'}")
    print("=" * 85)
    print(" PHASE 7 DEMONSTRATION COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_phase6_demo()
    run_phase7_demo()

