"""Optional evidence-only LLM explanation with deterministic fallback and per-alert cache."""

from __future__ import annotations

import json
from threading import RLock
from typing import Protocol

import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.models.explanation import (
    AlertExplanationPayload,
    AnalystExplanation,
    ExplanationAnomalyEvidence,
    ExplanationMLEvidence,
    ExplanationNetworkEvidence,
    ExplanationRuleEvidence,
)
from app.models.fraud_alert import FraudAlert

SYSTEM_PROMPT = """You are an explanation assistant for a fraud analyst.
The supplied structured evidence is authoritative. Do not invent facts or add evidence.
Do not calculate a new risk score. Do not claim certainty that fraud occurred.
Explain why the alert was surfaced. Mention relevant rule, behavioural anomaly, ML, and
network evidence when present. Distinguish observed evidence from interpretation.
Recommend investigation steps, not automatic blocking. The model's prediction is a
model signal, not proof. Return only a JSON object with string `summary`, string-array
`why_flagged`, string-array `key_evidence`, and string `analyst_action`."""


class ExplanationProvider(Protocol):
    """Provider interface; the explanation service does not depend on SDK types."""

    def explain(self, payload: AlertExplanationPayload) -> dict[str, object]: ...


class AnthropicMessagesProvider:
    """Small httpx adapter for Anthropic Messages API; instantiated only when enabled."""

    def __init__(self, api_key: str, model: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    def explain(self, payload: AlertExplanationPayload) -> dict[str, object]:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 450,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain this alert using only the following JSON evidence:\n"
                        + payload.model_dump_json(),
                    }
                ],
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if block.get("type") == "text"
        )
        if not text:
            raise ValueError("LLM provider returned no explanation text.")
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON explanation.")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response must be a JSON object.")
        return parsed


class ExplanationService:
    """Generate/cache explanations without altering the source alert or its risk score."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: ExplanationProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or _configured_provider(self.settings)
        self._cache: dict[str, AnalystExplanation] = {}
        self._lock = RLock()

    def explain(self, alert: FraudAlert) -> AnalystExplanation:
        """Return cached explanation or create one from a bounded structured payload."""

        with self._lock:
            cached = self._cache.get(alert.alert_id)
            if cached is not None:
                return cached

        payload = build_explanation_payload(alert)
        explanation = self._try_provider(payload) if self.provider is not None else None
        if explanation is None:
            explanation = deterministic_explanation(payload)

        with self._lock:
            return self._cache.setdefault(alert.alert_id, explanation)

    def _try_provider(self, payload: AlertExplanationPayload) -> AnalystExplanation | None:
        if not self.settings.llm_enabled or not self.settings.llm_api_key:
            return None
        try:
            assert self.provider is not None
            content = self.provider.explain(payload)
            content["source"] = "llm"
            return AnalystExplanation.model_validate(content)
        except (httpx.HTTPError, ValidationError, ValueError, TypeError, json.JSONDecodeError):
            return None
        except Exception:
            # Provider SDK/transport-specific failures must retain usable analyst fallback.
            return None


def _configured_provider(settings: Settings) -> ExplanationProvider | None:
    if not settings.llm_enabled or not settings.llm_api_key:
        return None
    if settings.llm_provider.lower() == "anthropic" and settings.llm_model:
        return AnthropicMessagesProvider(
            settings.llm_api_key,
            settings.llm_model,
            settings.llm_timeout_seconds,
        )
    return None


def build_explanation_payload(alert: FraudAlert) -> AlertExplanationPayload:
    """Copy only the explicitly approved explanation fields; labels/patterns are absent."""

    rules = [
        ExplanationRuleEvidence(
            rule_id=signal.rule_id,
            severity=signal.severity,
            reason=signal.reason,
            evidence=signal.evidence,
        )
        for signal in alert.rule_evidence
    ]
    anomaly = (
        ExplanationAnomalyEvidence(
            anomaly_score=alert.anomaly_evidence.anomaly_score,
            is_anomalous=alert.anomaly_evidence.is_anomalous,
            model_name=alert.anomaly_evidence.model_name,
        )
        if alert.anomaly_evidence is not None
        else None
    )
    ml = (
        ExplanationMLEvidence(
            fraud_probability=alert.ml_evidence.fraud_probability,
            predicted_fraud=alert.ml_evidence.predicted_fraud,
            model_name=alert.ml_evidence.model_name,
        )
        if alert.ml_evidence is not None
        else None
    )
    network = [
        ExplanationNetworkEvidence(
            pattern_type=finding.pattern_type,
            involved_accounts=finding.involved_accounts[:8],
            transaction_ids=finding.transaction_ids[:8],
            reason=finding.reason,
            metrics={
                key: value
                for key, value in finding.metrics.items()
                if key != "timestamps"
            },
        )
        for finding in alert.network_evidence
    ]
    return AlertExplanationPayload(
        alert_id=alert.alert_id,
        transaction_id=alert.transaction_id,
        timestamp=alert.timestamp,
        sender_account=alert.sender_account,
        receiver_account=alert.receiver_account,
        amount=alert.amount,
        transaction_type=alert.transaction_type,
        risk_score=alert.risk_score,
        risk_level=alert.risk_level,
        rule_evidence=rules,
        anomaly_evidence=anomaly,
        ml_evidence=ml,
        network_evidence=network,
    )


def deterministic_explanation(payload: AlertExplanationPayload) -> AnalystExplanation:
    """Compose concise fallback strings solely from the allow-listed evidence payload."""

    why_flagged = [rule.reason for rule in payload.rule_evidence]
    if payload.anomaly_evidence is not None and payload.anomaly_evidence.is_anomalous:
        why_flagged.append(
            f"Isolation Forest marked the behavioural pattern anomalous "
            f"(score {payload.anomaly_evidence.anomaly_score:.3f})."
        )
    if payload.ml_evidence is not None:
        why_flagged.append(
            f"LightGBM produced a model probability of "
            f"{payload.ml_evidence.fraud_probability:.3f}."
        )
    why_flagged.extend(finding.reason for finding in payload.network_evidence)
    why_flagged = list(dict.fromkeys(why_flagged))

    key_evidence = [
        f"Transaction {payload.transaction_id}: {payload.transaction_type}, amount {payload.amount:,.2f}.",
    ]
    key_evidence.extend(
        f"Rule {rule.rule_id} ({rule.severity}): {rule.reason}."
        for rule in payload.rule_evidence
    )
    if payload.anomaly_evidence is not None:
        state = "anomalous" if payload.anomaly_evidence.is_anomalous else "not marked anomalous"
        key_evidence.append(
            f"Isolation Forest score {payload.anomaly_evidence.anomaly_score:.3f}; {state}."
        )
    if payload.ml_evidence is not None:
        key_evidence.append(
            f"LightGBM probability {payload.ml_evidence.fraud_probability:.3f}; "
            f"model output {'predicted fraud' if payload.ml_evidence.predicted_fraud else 'predicted safe'}."
        )
    for finding in payload.network_evidence:
        accounts = " → ".join(finding.involved_accounts)
        key_evidence.append(
            f"Network {finding.pattern_type}: accounts {accounts}; "
            f"transactions {', '.join(finding.transaction_ids)}."
        )

    if not why_flagged:
        why_flagged.append("No detector evidence was supplied for this alert.")
    summary = (
        f"{payload.risk_level.title()} risk alert for {payload.transaction_id}. "
        f"This score is decision support, not confirmation of fraud."
    )
    if payload.risk_level == "high":
        analyst_action = "Review the transaction and corroborating account/network activity; do not block automatically."
    else:
        analyst_action = "Review the cited evidence and surrounding account activity before taking action."
    return AnalystExplanation(
        source="deterministic",
        summary=summary,
        why_flagged=why_flagged,
        key_evidence=key_evidence,
        analyst_action=analyst_action,
    )