"""
Unit tests for retrainer.py
Verifies candidate model training and the dual-window acceptance rule
per PROJECT_SPEC.md Section 5.8 and Section 9.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.retrainer import (
    evaluate_acceptance,
    train_candidate_model,
    retrain_model,
    RetrainingResult
)
from src.baseline_model import build_preprocessor


@pytest.fixture
def test_config():
    return {
        "model": {
            "type": "RandomForestRegressor",
            "n_estimators": 10,
            "random_state": 42
        },
        "retraining": {
            "min_improvement": 0.05,        # 5% improvement required on recent data
            "max_old_data_regression": 0.10  # max 10% regression on old reference data
        }
    }


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "retail_price": np.random.uniform(100.0, 1000.0, size=n),
        "product_rating": np.random.uniform(1.0, 5.0, size=n),
        "Region": np.random.choice(["North", "South", "East", "West"], size=n),
        "product_main_category": np.random.choice(["Clothing", "Electronics", "Home"], size=n),
        "BrandGroup": np.random.choice(["Brand_A", "Brand_B", "Other"], size=n),
        "OrderMonth": np.random.choice([1, 2, 3, 4, 5, 6], size=n),
        "discount_pct": np.random.uniform(0.1, 0.7, size=n)
    })
    return df


# -----------------------------------------------------------------------------
# Acceptance Rule Pure Logic Tests (Section 9 Requirement)
# -----------------------------------------------------------------------------

def test_acceptance_clearly_better_candidate_is_accepted(test_config):
    # Candidate improves recent MAE by 20% (0.20 -> 0.16)
    # Candidate old MAE only changes by +1% (0.10 -> 0.101)
    accepted, reason, imp_pct, reg_pct = evaluate_acceptance(
        current_recent_mae=0.20,
        candidate_recent_mae=0.16,
        current_old_mae=0.10,
        candidate_old_mae=0.101,
        config=test_config
    )
    assert accepted is True
    assert pytest.approx(imp_pct) == 20.0
    assert pytest.approx(reg_pct) == 1.0
    assert "Candidate accepted" in reason


def test_acceptance_insufficient_recent_improvement_is_rejected(test_config):
    # Candidate only improves recent MAE by 2% (0.20 -> 0.196) < 5% threshold
    # Old data regression is fine (+0%)
    accepted, reason, imp_pct, reg_pct = evaluate_acceptance(
        current_recent_mae=0.20,
        candidate_recent_mae=0.196,
        current_old_mae=0.10,
        candidate_old_mae=0.10,
        config=test_config
    )
    assert accepted is False
    assert pytest.approx(imp_pct) == 2.0
    assert "did not meet the required threshold" in reason


def test_acceptance_catastrophic_forgetting_is_rejected(test_config):
    # Candidate improves recent MAE substantially by 25% (0.20 -> 0.15)
    # BUT candidate regresses old reference data by 15% (0.10 -> 0.115) > 10% limit
    accepted, reason, imp_pct, reg_pct = evaluate_acceptance(
        current_recent_mae=0.20,
        candidate_recent_mae=0.15,
        current_old_mae=0.10,
        candidate_old_mae=0.115,
        config=test_config
    )
    assert accepted is False
    assert pytest.approx(imp_pct) == 25.0
    assert pytest.approx(reg_pct) == 15.0
    assert "catastrophic forgetting" in reason.lower()


def test_acceptance_worse_candidate_is_rejected(test_config):
    # Candidate is worse on recent data (0.20 -> 0.25, -25% improvement)
    accepted, reason, imp_pct, reg_pct = evaluate_acceptance(
        current_recent_mae=0.20,
        candidate_recent_mae=0.25,
        current_old_mae=0.10,
        candidate_old_mae=0.10,
        config=test_config
    )
    assert accepted is False
    assert imp_pct < 0.0


# -----------------------------------------------------------------------------
# End-to-End Retraining Workflow Tests
# -----------------------------------------------------------------------------

def test_retrain_model_pipeline_acceptance(synthetic_data, test_config):
    ref_train = synthetic_data.iloc[:100].copy()
    ref_val = synthetic_data.iloc[100:150].copy()
    
    # Initial baseline model trained on ref_train
    baseline_model = train_candidate_model(ref_train, config=test_config)
    
    # Recent batches: 2 batches
    b1 = synthetic_data.iloc[150:175].copy()
    b2 = synthetic_data.iloc[175:200].copy()
    
    # Mock evaluate_acceptance to simulate a clearly better candidate
    with patch("src.retrainer.evaluate_acceptance") as mock_eval:
        mock_eval.return_value = (True, "Candidate improved recent MAE by 15%", 15.0, 2.0)
        
        result = retrain_model(
            current_model=baseline_model,
            ref_train_df=ref_train,
            ref_val_df=ref_val,
            recent_batches=[b1, b2],
            config=test_config,
            val_batches_count=1,
            reason="Simulated degradation"
        )
        
    assert isinstance(result, RetrainingResult)
    assert result.accepted is True
    assert result.decision == "ACCEPTED"
    # Active model should now be updated to candidate model
    assert result.active_model == result.candidate_model


def test_retrain_model_pipeline_rejection(synthetic_data, test_config):
    ref_train = synthetic_data.iloc[:100].copy()
    ref_val = synthetic_data.iloc[100:150].copy()
    baseline_model = train_candidate_model(ref_train, config=test_config)
    
    b1 = synthetic_data.iloc[150:175].copy()
    b2 = synthetic_data.iloc[175:200].copy()
    
    # Mock evaluate_acceptance to simulate rejection
    with patch("src.retrainer.evaluate_acceptance") as mock_eval:
        mock_eval.return_value = (False, "Candidate failed old data regression check", 12.0, 18.0)
        
        result = retrain_model(
            current_model=baseline_model,
            ref_train_df=ref_train,
            ref_val_df=ref_val,
            recent_batches=[b1, b2],
            config=test_config,
            val_batches_count=1,
            reason="Simulated degradation"
        )
        
    assert isinstance(result, RetrainingResult)
    assert result.accepted is False
    assert result.decision == "REJECTED"
    # Active model must remain the original baseline model!
    assert result.active_model == baseline_model


def test_retrain_model_empty_batches_raises():
    with pytest.raises(ValueError, match="Cannot retrain without recent labelled batches"):
        retrain_model(
            current_model=MagicMock(),
            ref_train_df=pd.DataFrame(),
            ref_val_df=pd.DataFrame(),
            recent_batches=[]
        )
