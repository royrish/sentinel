"""API tests for the JSON-backed unified fraud alert interface."""

import unittest
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.alerts import get_alert_store
from app.main import app
from app.models.fraud_alert import FraudAlert, FraudAlertDataset, RiskConfiguration, TransactionRiskEvidence
from app.models.network_evidence import NetworkFinding
from app.services.alert_store import InMemoryAlertStore

TIMESTAMP = datetime(2025, 1, 1, tzinfo=timezone.utc)


def fixture_dataset() -> FraudAlertDataset:
    related_cycle = NetworkFinding(
        pattern_type="circular_flow",
        involved_accounts=["A", "B", "C"],
        transaction_ids=["tx-1", "tx-2"],
        severity="medium",
        reason="Directed transfers form a bounded account cycle",
        metrics={"cycle_length": 3, "time_window_seconds": 80},
    )
    first = TransactionRiskEvidence(
        alert_id="FS-tx-1",
        transaction_id="tx-1",
        timestamp=TIMESTAMP,
        sender_account="A",
        receiver_account="B",
        amount=5000,
        transaction_type="transfer",
        risk_score=75,
        risk_level="high",
        rule_component=15,
        anomaly_component=10,
        ml_component=30,
        network_component=20,
        rule_evidence=[],
        anomaly_evidence=None,
        ml_evidence=None,
        network_evidence=[related_cycle],
        explanation_summary="High risk / investigation recommended. Linked account evidence.",
    )
    second = TransactionRiskEvidence(
        alert_id="FS-tx-2",
        transaction_id="tx-2",
        timestamp=TIMESTAMP,
        sender_account="B",
        receiver_account="C",
        amount=2500,
        transaction_type="transfer",
        risk_score=10,
        risk_level="low",
        rule_component=0,
        anomaly_component=0,
        ml_component=10,
        network_component=0,
        rule_evidence=[],
        anomaly_evidence=None,
        ml_evidence=None,
        network_evidence=[],
        explanation_summary="Low risk; continue routine monitoring.",
    )
    return FraudAlertDataset(
        generated_at=TIMESTAMP,
        total_transactions=2,
        risk_configuration=RiskConfiguration(
            medium_threshold=30,
            high_threshold=70,
            alert_threshold=30,
        ),
        selected_alert_count=1,
        alert_ids=["FS-tx-1"],
        alerts=[FraudAlert.model_validate(first.model_dump())],
        transaction_evidence=[first, second],
    )


class AlertApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryAlertStore(fixture_dataset())
        app.dependency_overrides[get_alert_store] = lambda: self.store
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        app.dependency_overrides.pop(get_alert_store, None)

    def test_list_alerts_supports_filter_and_limit(self) -> None:
        all_alerts = self.client.get("/api/alerts")
        high_alerts = self.client.get("/api/alerts?risk_level=high&limit=1")
        no_open_alerts = self.client.get("/api/alerts?status=confirmed_fraud")

        self.assertEqual(all_alerts.status_code, 200)
        self.assertEqual(all_alerts.json()["total"], 1)
        self.assertEqual(all_alerts.json()["items"][0]["alert_id"], "FS-tx-1")
        self.assertEqual(high_alerts.json()["total"], 1)
        self.assertEqual(len(high_alerts.json()["items"]), 1)
        self.assertEqual(no_open_alerts.json()["total"], 0)

    def test_get_alert_detail_includes_independent_evidence_and_status(self) -> None:
        response = self.client.get("/api/alerts/FS-tx-1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["risk_score"], 75)
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["status"], "open")
        self.assertEqual(body["network_evidence"][0]["pattern_type"], "circular_flow")
        self.assertIn("investigation recommended", body["explanation_summary"])

    def test_graph_endpoint_returns_typed_nodes_and_real_transaction_edges(self) -> None:
        response = self.client.get("/api/alerts/FS-tx-1/graph")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["alert_id"], "FS-tx-1")
        self.assertEqual({node["id"] for node in body["nodes"]}, {"A", "B", "C"})
        self.assertEqual({edge["transaction_id"] for edge in body["edges"]}, {"tx-1", "tx-2"})
        self.assertEqual(body["edges"][0]["source"], "A")
        self.assertEqual(body["edges"][0]["target"], "B")

    def test_non_alert_transaction_evidence_remains_retrievable(self) -> None:
        response = self.client.get("/api/transactions/tx-2/evidence")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["risk_level"], "low")
        self.assertEqual(response.json()["alert_id"], "FS-tx-2")

    def test_feedback_updates_runtime_status_and_summary(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={"alert_id": "FS-tx-1", "status": "confirmed_fraud"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"alert_id": "FS-tx-1", "status": "confirmed_fraud"})
        self.assertEqual(
            self.client.get("/api/alerts/FS-tx-1").json()["status"],
            "confirmed_fraud",
        )
        summary = self.client.get("/api/summary").json()
        self.assertEqual(summary["total_alerts"], 1)
        self.assertEqual(summary["high_risk"], 1)
        self.assertEqual(summary["open_alerts"], 0)
        self.assertEqual(summary["confirmed_fraud"], 1)

    def test_marked_safe_feedback_and_summary_status_counts(self) -> None:
        response = self.client.post(
            "/api/feedback",
            json={"alert_id": "FS-tx-1", "status": "marked_safe"},
        )
        self.assertEqual(response.status_code, 200)
        summary = self.client.get("/api/summary").json()
        self.assertEqual(summary["marked_safe"], 1)
        self.assertEqual(summary["confirmed_fraud"], 0)

    def test_invalid_ids_and_feedback_status_are_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/alerts/unknown").status_code, 404)
        self.assertEqual(self.client.get("/api/alerts/unknown/graph").status_code, 404)
        self.assertEqual(self.client.get("/api/transactions/unknown/evidence").status_code, 404)
        self.assertEqual(
            self.client.post(
                "/api/feedback",
                json={"alert_id": "FS-tx-1", "status": "open"},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.client.post(
                "/api/feedback",
                json={"alert_id": "unknown", "status": "marked_safe"},
            ).status_code,
            404,
        )

    def test_invalid_limit_is_rejected(self) -> None:
        self.assertEqual(self.client.get("/api/alerts?limit=0").status_code, 422)


if __name__ == "__main__":
    unittest.main()