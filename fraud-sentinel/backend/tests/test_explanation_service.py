"""Tests for the optional, evidence-only analyst explanation service."""

import unittest

from app.config import Settings
from app.models.explanation import AlertExplanationPayload
from app.models.network_evidence import NetworkFinding
from app.services.explanation_service import (
    ExplanationService,
    build_explanation_payload,
    deterministic_explanation,
)
from tests.test_alert_api import fixture_dataset


class CapturingProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[AlertExplanationPayload] = []

    def explain(self, payload: AlertExplanationPayload) -> dict[str, object]:
        self.calls += 1
        self.payloads.append(payload)
        return {
            "summary": "The alert combines cited account-transfer evidence.",
            "why_flagged": ["A short directed cycle was observed."],
            "key_evidence": ["Transaction tx-1 is part of the cycle."],
            "analyst_action": "Review account activity before deciding on action.",
        }


class FailingProvider:
    def explain(self, payload: AlertExplanationPayload) -> dict[str, object]:
        self.last_payload = payload
        raise RuntimeError("provider unavailable")


class ExplanationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.alert = fixture_dataset().alerts[0]

    def test_deterministic_explanation_uses_structured_evidence(self) -> None:
        payload = build_explanation_payload(self.alert)
        explanation = deterministic_explanation(payload)

        self.assertEqual(explanation.source, "deterministic")
        self.assertIn(self.alert.transaction_id, explanation.summary)
        self.assertTrue(any("cycle" in item.lower() for item in explanation.why_flagged))
        self.assertIn("do not block automatically", explanation.analyst_action)

    def test_disabled_llm_returns_deterministic_explanation(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(
            settings=Settings(llm_enabled=False, llm_api_key="test-secret"),
            provider=provider,
        )

        explanation = service.explain(self.alert)

        self.assertEqual(explanation.source, "deterministic")
        self.assertEqual(provider.calls, 0)

    def test_missing_api_key_returns_deterministic_explanation(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(
            settings=Settings(llm_enabled=True, llm_provider="anthropic", llm_api_key=""),
            provider=provider,
        )

        explanation = service.explain(self.alert)

        self.assertEqual(explanation.source, "deterministic")
        self.assertEqual(provider.calls, 0)

    def test_provider_failure_falls_back_to_deterministic(self) -> None:
        service = ExplanationService(
            settings=Settings(llm_enabled=True, llm_api_key="configured"),
            provider=FailingProvider(),
        )
        explanation = service.explain(self.alert)
        self.assertEqual(explanation.source, "deterministic")
        self.assertIn(self.alert.transaction_id, explanation.summary)

    def test_llm_output_is_cached_by_alert_id(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(
            settings=Settings(llm_enabled=True, llm_api_key="configured"),
            provider=provider,
        )
        before_score = self.alert.risk_score

        first = service.explain(self.alert)
        second = service.explain(self.alert)

        self.assertEqual(first.source, "llm")
        self.assertEqual(first, second)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(self.alert.risk_score, before_score)

    def test_provider_payload_is_allow_listed_and_excludes_ground_truth(self) -> None:
        provider = CapturingProvider()
        service = ExplanationService(
            settings=Settings(llm_enabled=True, llm_api_key="configured"),
            provider=provider,
        )
        service.explain(self.alert)
        sent = provider.payloads[0].model_dump()

        self.assertFalse(hasattr(provider.payloads[0], "label"))
        self.assertFalse(hasattr(provider.payloads[0], "fraud_pattern"))
        self.assertNotIn("label", sent)
        self.assertNotIn("fraud_pattern", sent)
        self.assertNotIn("status", sent)
        self.assertNotIn("rule_component", sent)
        self.assertIn("rule_evidence", sent)
        self.assertIn("network_evidence", sent)

    def test_payload_is_stable_and_contains_only_alert_evidence(self) -> None:
        payload = build_explanation_payload(self.alert)
        self.assertEqual(payload.risk_score, self.alert.risk_score)
        self.assertEqual(payload.network_evidence[0].pattern_type, "circular_flow")
        self.assertEqual(payload.transaction_id, self.alert.transaction_id)

    def test_network_payload_is_bounded_and_omits_raw_timestamps(self) -> None:
        finding = NetworkFinding(
            pattern_type="high_connectivity_hub",
            involved_accounts=[f"account-{index}" for index in range(20)],
            transaction_ids=[f"transaction-{index}" for index in range(30)],
            severity="low",
            reason="Account had high transaction connectivity.",
            metrics={
                "total_degree": 30,
                "timestamps": [f"2025-01-{index + 1:02d}" for index in range(20)],
            },
        )
        alert = self.alert.model_copy(update={"network_evidence": [finding]})

        payload = build_explanation_payload(alert)

        self.assertEqual(len(payload.network_evidence[0].involved_accounts), 8)
        self.assertEqual(len(payload.network_evidence[0].transaction_ids), 8)
        self.assertNotIn("timestamps", payload.network_evidence[0].metrics)


if __name__ == "__main__":
    unittest.main()