"""Tests for the optional alert explanation API endpoint."""

import unittest

from fastapi.testclient import TestClient

from app.api.alerts import get_alert_store, get_explanation_service
from app.config import Settings
from app.main import app
from app.models.fraud_alert import FraudAlert
from app.services.alert_store import InMemoryAlertStore
from app.services.explanation_service import ExplanationService
from tests.test_alert_api import fixture_dataset


class ExplanationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryAlertStore(fixture_dataset())
        self.service = ExplanationService(settings=Settings(llm_enabled=False))
        app.dependency_overrides[get_alert_store] = lambda: self.store
        app.dependency_overrides[get_explanation_service] = lambda: self.service
        self.client = TestClient(app)
        self.client.__enter__()

    def _fixture_alert(self) -> FraudAlert:
        alert = self.store.get_alert("FS-tx-1")
        if alert is None:
            raise AssertionError("Explanation API fixture alert is missing.")
        return alert

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        app.dependency_overrides.pop(get_alert_store, None)
        app.dependency_overrides.pop(get_explanation_service, None)

    def test_explanation_endpoint_returns_deterministic_fallback_when_disabled(self) -> None:
        before_score = self._fixture_alert().risk_score

        response = self.client.post("/api/alerts/FS-tx-1/explanation")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["source"], "deterministic")
        self.assertTrue(body["summary"])
        self.assertTrue(body["why_flagged"])
        self.assertTrue(body["key_evidence"])
        self.assertTrue(body["analyst_action"])
        self.assertEqual(self._fixture_alert().risk_score, before_score)

    def test_unknown_alert_id_returns_404(self) -> None:
        response = self.client.post("/api/alerts/unknown/explanation")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Alert not found.")

    def test_explanation_does_not_change_alert_risk_score(self) -> None:
        before_alert = self._fixture_alert()
        before = before_alert.risk_score
        self.client.post("/api/alerts/FS-tx-1/explanation")
        after = self._fixture_alert().risk_score
        self.assertEqual(before, 75)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()