"""
Hybrid Decision Agent Module for DriftMonitor
Author: DriftMonitor Team
Purpose: Implements the hybrid agent decision layer per PROJECT_SPEC.md Section 5.7.
         Combines deterministic Python rules (authoritative on allowed actions),
         optional LLM reasoning/summarization (explains and selects within allowed set),
         and strict safety guardrails with a zero-dependency template fallback.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import os
import json
import re
import logging
import urllib.request
import urllib.error
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load optional .env if present
load_dotenv()

# Canonical Allowed Actions per Section 5.7
ACTION_CONTINUE = "CONTINUE"
ACTION_INVESTIGATE = "INVESTIGATE"
ACTION_INTENSIFY_MONITORING = "INTENSIFY_MONITORING"
ACTION_REQUEST_LABELS = "REQUEST_LABELS"
ACTION_PROPOSE_RETRAIN = "PROPOSE_RETRAIN"

ALL_ALLOWED_ACTIONS: Set[str] = {
    ACTION_CONTINUE,
    ACTION_INVESTIGATE,
    ACTION_INTENSIFY_MONITORING,
    ACTION_REQUEST_LABELS,
    ACTION_PROPOSE_RETRAIN,
}


@dataclass
class RuleEvaluation:
    """Outcome of deterministic Step 1 rule table evaluation."""
    situation: str
    default_action: str
    allowed_actions: List[str]


@dataclass
class DecisionReport:
    """Complete structured incident decision output."""
    batch_id: int
    quality_passed: bool
    drift_detected: bool
    performance_status: str
    allowed_actions: List[str]
    default_action: str
    chosen_action: str
    guardrail_overridden: bool
    used_llm: bool
    summary: str
    observed_facts: List[str]
    possible_explanations: List[str]
    follow_up_checks: List[str]
    raw_llm_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to standard dictionary."""
        return {
            "batch_id": self.batch_id,
            "quality_passed": self.quality_passed,
            "drift_detected": self.drift_detected,
            "performance_status": self.performance_status,
            "allowed_actions": self.allowed_actions,
            "default_action": self.default_action,
            "chosen_action": self.chosen_action,
            "guardrail_overridden": self.guardrail_overridden,
            "used_llm": self.used_llm,
            "summary": self.summary,
            "observed_facts": self.observed_facts,
            "possible_explanations": self.possible_explanations,
            "follow_up_checks": self.follow_up_checks,
            "raw_llm_response": self.raw_llm_response,
        }


# -----------------------------------------------------------------------------
# Step 1: Deterministic Rules Table (Section 5.7)
# -----------------------------------------------------------------------------

def evaluate_rules(
    quality_passed: bool,
    drift_detected: bool,
    performance_status: str
) -> RuleEvaluation:
    """
    Step 1: Evaluate authoritative Python rules table.
    Determines default action and sets bounded allowed actions.
    
    Rule Table (PROJECT_SPEC.md Section 5.7):
    | Situation | Default action | Allowed actions |
    |---|---|---|
    | No drift, performance normal | CONTINUE | CONTINUE |
    | Drift detected, labels unavailable | REQUEST_LABELS | REQUEST_LABELS, INTENSIFY_MONITORING |
    | Drift detected, labels available, performance normal | INVESTIGATE | INVESTIGATE, INTENSIFY_MONITORING |
    | Drift detected, performance degraded | PROPOSE_RETRAIN | PROPOSE_RETRAIN, INVESTIGATE |
    | No feature drift, performance degraded | INVESTIGATE | INVESTIGATE, PROPOSE_RETRAIN |
    | Batch failed quality checks | INVESTIGATE | INVESTIGATE |
    """
    perf = performance_status.lower()

    # Rule 1: Batch failed data quality checks
    if not quality_passed:
        return RuleEvaluation(
            situation="Batch failed quality checks",
            default_action=ACTION_INVESTIGATE,
            allowed_actions=[ACTION_INVESTIGATE]
        )

    # Rule 2: Drift detected, labels unavailable (unverified performance)
    if drift_detected and perf == "unverified":
        return RuleEvaluation(
            situation="Drift detected, labels unavailable",
            default_action=ACTION_REQUEST_LABELS,
            allowed_actions=[ACTION_REQUEST_LABELS, ACTION_INTENSIFY_MONITORING]
        )

    # Rule 3: Drift detected, performance degraded
    if drift_detected and perf == "degraded":
        return RuleEvaluation(
            situation="Drift detected, performance degraded",
            default_action=ACTION_PROPOSE_RETRAIN,
            allowed_actions=[ACTION_PROPOSE_RETRAIN, ACTION_INVESTIGATE]
        )

    # Rule 4: Drift detected, labels available, performance normal
    if drift_detected and perf == "normal":
        return RuleEvaluation(
            situation="Drift detected, labels available, performance normal",
            default_action=ACTION_INVESTIGATE,
            allowed_actions=[ACTION_INVESTIGATE, ACTION_INTENSIFY_MONITORING]
        )

    # Rule 5: No feature drift, performance degraded (e.g. concept drift / label shift)
    if not drift_detected and perf == "degraded":
        return RuleEvaluation(
            situation="No feature drift, performance degraded",
            default_action=ACTION_INVESTIGATE,
            allowed_actions=[ACTION_INVESTIGATE, ACTION_PROPOSE_RETRAIN]
        )

    # Rule 6: No drift, performance normal (or unverified when no drift)
    return RuleEvaluation(
        situation="No drift, performance normal",
        default_action=ACTION_CONTINUE,
        allowed_actions=[ACTION_CONTINUE]
    )


# -----------------------------------------------------------------------------
# Template Fallback (Runs with Zero API Keys)
# -----------------------------------------------------------------------------

def build_template_fallback(
    batch_id: int,
    rule_eval: RuleEvaluation,
    quality_status: Dict[str, Any],
    drift_report: Dict[str, Any],
    performance_record: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs an informative, deterministic incident report when an LLM is not used
    or when an LLM fails / returns invalid JSON.
    """
    ranking = drift_report.get("ranking", [])
    top_features = ranking[:3] if ranking else []
    feat_details = drift_report.get("features", {})
    issues = quality_status.get("issues", [])
    mae = performance_record.get("batch_mae")
    mae_str = f"MAE={mae:.4f}" if mae is not None else "labels unavailable"

    facts = []
    explanations = []
    follow_ups = []

    if not quality_status.get("passed", True):
        summary = (
            f"Batch {batch_id} failed quality validation with {len(issues)} issue(s). "
            f"Drift analysis was paused to prevent corrupted records from distorting statistical monitors."
        )
        facts = [f"Quality issues: {'; '.join(issues)}"]
        explanations = [
            "Upstream data pipeline schema deviation or formatting error.",
            "Malformed data ingestion from supplier source."
        ]
        follow_ups = [
            "Check upstream ingestion validation logs.",
            "Sanitize raw data batch before retraining or monitoring."
        ]

    elif rule_eval.situation == "Drift detected, labels unavailable":
        summary = (
            f"Statistical data drift was confirmed in Batch {batch_id} across {len(top_features)} feature(s), "
            f"led by {', '.join(top_features)}. Ground truth target labels have not yet arrived, "
            f"so model accuracy impact remains unverified."
        )
        drift_facts = []
        for feat in top_features:
            psi = feat_details.get(feat, {}).get("psi", 0.0)
            drift_facts.append(f"{feat} (PSI={psi:.4f})")
        facts = [
            f"Confirmed feature drift: {', '.join(drift_facts)}",
            "Performance status: unverified (delayed ground-truth labels)"
        ]
        explanations = [
            "Shifting customer or market distributions before outcome labels are recorded.",
            "Seasonal or geographic demand shifts."
        ]
        follow_ups = [
            "Expedite label acquisition for recent batches.",
            "Monitor persistence of drift signals in subsequent batches."
        ]

    elif rule_eval.situation == "Drift detected, performance degraded":
        summary = (
            f"Critical incident at Batch {batch_id}: Significant input feature drift coincided with sustained "
            f"model performance degradation ({mae_str}). Retraining is recommended subject to validation acceptance."
        )
        drift_facts = []
        for feat in top_features:
            psi = feat_details.get(feat, {}).get("psi", 0.0)
            drift_facts.append(f"{feat} (PSI={psi:.4f})")
        facts = [
            f"Confirmed feature drift: {', '.join(drift_facts)}",
            f"Model error degraded: {mae_str} (exceeded 1.2x baseline for consecutive batches)"
        ]
        explanations = [
            "Model features shifted into unexplored feature space causing loss of predictive capability.",
            "Market conditions have structurally altered price-discount dynamics."
        ]
        follow_ups = [
            "Train candidate model on reference data plus recent labelled batches.",
            "Evaluate candidate model against strict acceptance thresholds before deployment."
        ]

    elif rule_eval.situation == "Drift detected, labels available, performance normal":
        summary = (
            f"Data drift detected in Batch {batch_id} on {', '.join(top_features)}, "
            f"but model performance remains healthy and stable ({mae_str})."
        )
        facts = [
            f"Top drifted features: {', '.join(top_features)}",
            f"Model accuracy maintained within baseline bounds ({mae_str})"
        ]
        explanations = [
            "Drift is concentrated in features with low model importance.",
            "Feature interactions may offset prediction impact."
        ]
        follow_ups = [
            "Inspect feature importance for the drifting features.",
            "Maintain intensified surveillance to ensure errors do not lag."
        ]

    elif rule_eval.situation == "No feature drift, performance degraded":
        summary = (
            f"Model performance degradation observed at Batch {batch_id} ({mae_str}) "
            f"despite no statistical drift in measured input features. Possible concept drift or target shift."
        )
        facts = [
            "Input feature distributions are statistically stable",
            f"Model error degraded: {mae_str}"
        ]
        explanations = [
            "The mapping from features to outcome has changed (concept drift).",
            "Unobserved external factors (e.g. competitor promotions) impacting target."
        ]
        follow_ups = [
            "Investigate target distribution and pricing policies.",
            "Examine potential concept drift scenarios."
        ]

    else:
        # No drift, normal performance
        summary = (
            f"Batch {batch_id} is operating within normal parameters. "
            f"No statistical drift detected and model error is within baseline bounds ({mae_str})."
        )
        facts = [
            "All input features within statistical reference bounds",
            f"Model performance nominal ({mae_str})"
        ]
        explanations = [
            "Stable operating conditions."
        ]
        follow_ups = [
            "Continue standard monitoring schedule."
        ]

    return {
        "summary": summary,
        "observed_facts": facts,
        "possible_explanations": explanations,
        "recommended_action": rule_eval.default_action,
        "follow_up_checks": follow_ups
    }


# -----------------------------------------------------------------------------
# Step 2: Single LLM Call Function
# -----------------------------------------------------------------------------

def call_llm(prompt: str, timeout_sec: int = 15) -> str:
    """
    Single swappable LLM calling function per Section 5.7.
    Configured via environment variables:
      - LLM_PROVIDER: 'GEMINI' (default) or 'OPENAI'
      - GEMINI_API_KEY or GOOGLE_API_KEY
      - OPENAI_API_KEY
      
    If no key is configured or the call fails, raises an Exception
    so the agent cleanly and safely invokes the template fallback.
    """
    provider = os.getenv("LLM_PROVIDER", "GEMINI").upper()

    if provider == "GEMINI":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("No valid GEMINI_API_KEY configured in environment.")

        # REST endpoint for Gemini 1.5 / 2.0
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data["candidates"][0]["content"]["parts"][0]["text"]

    elif provider == "OPENAI":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key.startswith("your_"):
            raise ValueError("No valid OPENAI_API_KEY configured in environment.")

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a machine learning monitoring agent. Reply in valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data["choices"][0]["message"]["content"]

    raise ValueError(f"Unsupported LLM provider: {provider}")


def build_agent_prompt(
    batch_id: int,
    rule_eval: RuleEvaluation,
    quality_status: Dict[str, Any],
    drift_report: Dict[str, Any],
    performance_record: Dict[str, Any]
) -> str:
    """Constructs prompt providing only factual inputs and enforcing schema."""
    context = {
        "batch_id": batch_id,
        "quality_passed": quality_status.get("passed", True),
        "quality_issues": quality_status.get("issues", []),
        "drift_detected": drift_report.get("drift_detected", False),
        "top_features": [
            {
                "feature": f,
                "psi": drift_report.get("features", {}).get(f, {}).get("psi"),
                "ks_stat": drift_report.get("features", {}).get(f, {}).get("ks_stat"),
                "p_value": drift_report.get("features", {}).get(f, {}).get("ks_p", drift_report.get("features", {}).get(f, {}).get("chi2_p"))
            }
            for f in drift_report.get("ranking", [])[:3]
        ],
        "performance_status": performance_record.get("status", "unverified"),
        "batch_mae": performance_record.get("batch_mae"),
        "baseline_mae": performance_record.get("baseline_mae"),
        "allowed_actions": rule_eval.allowed_actions,
        "default_action": rule_eval.default_action
    }

    return (
        "You are DriftMonitor's decision layer. Analyze the following monitoring context "
        "and select exactly ONE action from the allowed_actions list.\n"
        "Rules:\n"
        "1. Never invent facts or numbers outside what is provided.\n"
        "2. Do not state performance dropped unless performance_status is 'degraded'.\n"
        "3. Recommended action MUST be one of the allowed_actions.\n\n"
        f"CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
        "Return ONLY a JSON object with this exact structure:\n"
        "{\n"
        '  "summary": "2-4 sentences in plain English",\n'
        '  "observed_facts": ["fact 1", "fact 2"],\n'
        '  "possible_explanations": ["hypothesis 1", "hypothesis 2"],\n'
        f'  "recommended_action": "ONE of {rule_eval.allowed_actions}",\n'
        '  "follow_up_checks": ["check 1", "check 2"]\n'
        "}"
    )


# -----------------------------------------------------------------------------
# Step 3: Guardrails & Hybrid Orchestrator
# -----------------------------------------------------------------------------

def extract_json_from_llm(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robustly parse JSON from raw LLM string, handling code fence wrappers."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()
    # Strip markdown ```json ... ``` blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


class DecisionAgent:
    """
    Hybrid Decision Agent orchestrating rules, LLM, and safety guardrails.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def decide(
        self,
        batch_id: int,
        quality_status: Dict[str, Any],
        drift_report: Dict[str, Any],
        performance_record: Dict[str, Any]
    ) -> DecisionReport:
        """
        Produce authoritative decision for a batch incident.
        
        Args:
            batch_id: Current batch ID.
            quality_status: Output from quality_checks.check_batch_quality.
            drift_report: Output from drift_detector.detect_drift.
            performance_record: Output from performance_monitor.evaluate_batch.
            
        Returns:
            DecisionReport containing chosen action, guardrail status, and explanations.
        """
        # Step 1: Deterministic rules (authoritative)
        quality_passed = bool(quality_status.get("passed", True))
        drift_detected = bool(drift_report.get("drift_detected", False))
        perf_status = str(performance_record.get("status", "unverified"))

        rule_eval = evaluate_rules(
            quality_passed=quality_passed,
            drift_detected=drift_detected,
            performance_status=perf_status
        )

        chosen_action = rule_eval.default_action
        guardrail_overridden = False
        used_llm_successfully = False
        raw_llm: Optional[str] = None
        parsed_llm: Optional[Dict[str, Any]] = None

        # Step 2: Attempt LLM call if enabled
        if self.use_llm:
            try:
                prompt = build_agent_prompt(
                    batch_id=batch_id,
                    rule_eval=rule_eval,
                    quality_status=quality_status,
                    drift_report=drift_report,
                    performance_record=performance_record
                )
                raw_llm = call_llm(prompt)
                parsed_llm = extract_json_from_llm(raw_llm)
            except Exception as exc:
                # LLM call or key unavailable -> smoothly proceed to guardrails/template
                logger.warning(f"LLM call or parsing failed for batch {batch_id}: {exc}")
                parsed_llm = None

        # Step 3: Guardrails
        if parsed_llm is not None:
            candidate_action = parsed_llm.get("recommended_action")

            # Guardrail check 1: Candidate action must be in allowed list
            if candidate_action in rule_eval.allowed_actions:
                chosen_action = candidate_action
                used_llm_successfully = True
                summary = str(parsed_llm.get("summary", ""))
                facts = list(parsed_llm.get("observed_facts", []))
                explanations = list(parsed_llm.get("possible_explanations", []))
                follow_ups = list(parsed_llm.get("follow_up_checks", []))
            else:
                # Disallowed action chosen by LLM: Guardrail activates!
                guardrail_overridden = True
                chosen_action = rule_eval.default_action
                template = build_template_fallback(
                    batch_id, rule_eval, quality_status, drift_report, performance_record
                )
                summary = (
                    f"[Guardrail Override] LLM proposed unauthorized action '{candidate_action}'. "
                    f"Reset to default allowed action '{chosen_action}'. " + template["summary"]
                )
                facts = template["observed_facts"]
                explanations = template["possible_explanations"]
                follow_ups = template["follow_up_checks"]
        else:
            # Fallback template used (no LLM, API error, or unparseable JSON)
            template = build_template_fallback(
                batch_id, rule_eval, quality_status, drift_report, performance_record
            )
            summary = template["summary"]
            facts = template["observed_facts"]
            explanations = template["possible_explanations"]
            follow_ups = template["follow_up_checks"]
            chosen_action = rule_eval.default_action

        return DecisionReport(
            batch_id=batch_id,
            quality_passed=quality_passed,
            drift_detected=drift_detected,
            performance_status=perf_status,
            allowed_actions=rule_eval.allowed_actions,
            default_action=rule_eval.default_action,
            chosen_action=chosen_action,
            guardrail_overridden=guardrail_overridden,
            used_llm=used_llm_successfully,
            summary=summary,
            observed_facts=facts,
            possible_explanations=explanations,
            follow_up_checks=follow_ups,
            raw_llm_response=raw_llm
        )
