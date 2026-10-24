"""
Unit tests for decision_agent.py
Tests the hybrid agent's rule engine, LLM integration, safety guardrails,
and template fallback per PROJECT_SPEC.md Section 5.7 and Section 9.
"""

import pytest
import json
from unittest.mock import patch

from src.decision_agent import (
    evaluate_rules,
    DecisionAgent,
    DecisionReport,
    extract_json_from_llm,
    build_template_fallback,
    ACTION_CONTINUE,
    ACTION_INVESTIGATE,
    ACTION_INTENSIFY_MONITORING,
    ACTION_REQUEST_LABELS,
    ACTION_PROPOSE_RETRAIN,
)


# -----------------------------------------------------------------------------
# Test 1: Each Table Row in Section 5.7 Gives Expected Default & Allowed Actions
# -----------------------------------------------------------------------------

def test_rules_table_row1_failed_quality():
    # Batch failed quality checks
    eval_res = evaluate_rules(quality_passed=False, drift_detected=True, performance_status="normal")
    assert eval_res.default_action == ACTION_INVESTIGATE
    assert eval_res.allowed_actions == [ACTION_INVESTIGATE]


def test_rules_table_row2_drift_unverified_labels():
    # Drift detected, labels unavailable (unverified)
    eval_res = evaluate_rules(quality_passed=True, drift_detected=True, performance_status="unverified")
    assert eval_res.default_action == ACTION_REQUEST_LABELS
    assert eval_res.allowed_actions == [ACTION_REQUEST_LABELS, ACTION_INTENSIFY_MONITORING]


def test_rules_table_row3_drift_performance_normal():
    # Drift detected, labels available, performance normal
    eval_res = evaluate_rules(quality_passed=True, drift_detected=True, performance_status="normal")
    assert eval_res.default_action == ACTION_INVESTIGATE
    assert eval_res.allowed_actions == [ACTION_INVESTIGATE, ACTION_INTENSIFY_MONITORING]


def test_rules_table_row4_drift_performance_degraded():
    # Drift detected, performance degraded
    eval_res = evaluate_rules(quality_passed=True, drift_detected=True, performance_status="degraded")
    assert eval_res.default_action == ACTION_PROPOSE_RETRAIN
    assert eval_res.allowed_actions == [ACTION_PROPOSE_RETRAIN, ACTION_INVESTIGATE]


def test_rules_table_row5_no_drift_performance_degraded():
    # No feature drift, performance degraded (concept drift)
    eval_res = evaluate_rules(quality_passed=True, drift_detected=False, performance_status="degraded")
    assert eval_res.default_action == ACTION_INVESTIGATE
    assert eval_res.allowed_actions == [ACTION_INVESTIGATE, ACTION_PROPOSE_RETRAIN]


def test_rules_table_row6_no_drift_performance_normal():
    # No drift, performance normal
    eval_res = evaluate_rules(quality_passed=True, drift_detected=False, performance_status="normal")
    assert eval_res.default_action == ACTION_CONTINUE
    assert eval_res.allowed_actions == [ACTION_CONTINUE]


# -----------------------------------------------------------------------------
# Test 2: Invalid LLM Action is Replaced by Default (Guardrail Protection)
# -----------------------------------------------------------------------------

def test_invalid_llm_action_replaced_by_default():
    agent = DecisionAgent(use_llm=True)
    
    # Context: Drift detected and performance degraded -> allowed: [PROPOSE_RETRAIN, INVESTIGATE]
    quality = {"passed": True, "issues": []}
    drift = {"drift_detected": True, "ranking": ["retail_price"], "features": {"retail_price": {"psi": 0.35}}}
    perf = {"status": "degraded", "batch_mae": 0.25, "baseline_mae": 0.15}

    # Simulate LLM returning a disallowed or hallucinated action, e.g. "SHUTDOWN_SYSTEM" or "CONTINUE"
    mock_llm_json = json.dumps({
        "summary": "Everything looks okay to me, ignore warnings.",
        "observed_facts": ["Retail price jumped"],
        "possible_explanations": ["Inflation"],
        "recommended_action": "CONTINUE",  # Disallowed in degraded state!
        "follow_up_checks": ["None"]
    })

    with patch("src.decision_agent.call_llm", return_value=mock_llm_json):
        report = agent.decide(
            batch_id=16,
            quality_status=quality,
            drift_report=drift,
            performance_record=perf
        )

    # Guardrail must fire!
    assert report.guardrail_overridden is True
    # Action must be reset to the authoritative default action
    assert report.chosen_action == ACTION_PROPOSE_RETRAIN
    assert "[Guardrail Override]" in report.summary


# -----------------------------------------------------------------------------
# Test 3: Invalid JSON Falls Back to Template
# -----------------------------------------------------------------------------

def test_invalid_json_falls_back_to_template():
    agent = DecisionAgent(use_llm=True)
    
    quality = {"passed": True, "issues": []}
    drift = {"drift_detected": True, "ranking": ["retail_price"], "features": {"retail_price": {"psi": 0.30}}}
    perf = {"status": "normal", "batch_mae": 0.14, "baseline_mae": 0.15}

    # Simulate LLM returning conversational text instead of JSON
    gibberish = "I think you should investigate this drift because retail_price has changed."

    with patch("src.decision_agent.call_llm", return_value=gibberish):
        report = agent.decide(
            batch_id=10,
            quality_status=quality,
            drift_report=drift,
            performance_record=perf
        )

    assert report.used_llm is False
    assert report.chosen_action == ACTION_INVESTIGATE
    assert "Data drift detected in Batch 10" in report.summary
    assert len(report.observed_facts) > 0
    assert len(report.possible_explanations) > 0


# -----------------------------------------------------------------------------
# Test 4: Valid LLM Output with Allowed Action is Accepted
# -----------------------------------------------------------------------------

def test_valid_llm_choice_accepted():
    agent = DecisionAgent(use_llm=True)
    
    # Drift detected, performance normal -> allowed: [INVESTIGATE, INTENSIFY_MONITORING], default: INVESTIGATE
    quality = {"passed": True, "issues": []}
    drift = {"drift_detected": True, "ranking": ["retail_price"], "features": {"retail_price": {"psi": 0.30}}}
    perf = {"status": "normal", "batch_mae": 0.14, "baseline_mae": 0.15}

    # LLM chooses INTENSIFY_MONITORING (which is in allowed_actions)
    valid_json = json.dumps({
        "summary": "Statistical drift observed on retail_price. Recommend intensifying monitoring until labels confirm impact.",
        "observed_facts": ["retail_price PSI is 0.30"],
        "possible_explanations": ["Seasonal promotion"],
        "recommended_action": ACTION_INTENSIFY_MONITORING,
        "follow_up_checks": ["Track next 3 batches"]
    })

    with patch("src.decision_agent.call_llm", return_value=valid_json):
        report = agent.decide(
            batch_id=12,
            quality_status=quality,
            drift_report=drift,
            performance_record=perf
        )

    assert report.guardrail_overridden is False
    assert report.used_llm is True
    assert report.chosen_action == ACTION_INTENSIFY_MONITORING
    assert "Statistical drift observed" in report.summary


# -----------------------------------------------------------------------------
# Test 5: End-to-End Execution with No API Key (Zero Dependencies)
# -----------------------------------------------------------------------------

def test_end_to_end_no_api_key():
    # Ensure no API key is present
    with patch.dict("os.environ", {"GEMINI_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True):
        agent = DecisionAgent(use_llm=True)

        quality = {"passed": True, "issues": []}
        drift = {"drift_detected": True, "ranking": ["retail_price"], "features": {"retail_price": {"psi": 0.40}}}
        perf = {"status": "unverified", "batch_mae": None, "baseline_mae": 0.15}

        report = agent.decide(
            batch_id=20,
            quality_status=quality,
            drift_report=drift,
            performance_record=perf
        )

        assert isinstance(report, DecisionReport)
        assert report.used_llm is False
        assert report.chosen_action == ACTION_REQUEST_LABELS
        assert "unverified" in report.summary.lower()
        assert len(report.observed_facts) > 0


# -----------------------------------------------------------------------------
# Test 6: Robust JSON Extraction (Stripping Markdown Fences)
# -----------------------------------------------------------------------------

def test_extract_json_from_llm():
    raw_markdown = "```json\n{\"summary\": \"Test summary\", \"recommended_action\": \"CONTINUE\"}\n```"
    parsed = extract_json_from_llm(raw_markdown)
    assert parsed is not None
    assert parsed["summary"] == "Test summary"
    assert parsed["recommended_action"] == "CONTINUE"

    # Empty text
    assert extract_json_from_llm("") is None
    # Bad JSON
    assert extract_json_from_llm("{not valid json}") is None
