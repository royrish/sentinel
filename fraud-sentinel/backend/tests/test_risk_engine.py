"""Tests for the independent-evidence integration and transparent risk formula."""

import unittest
from datetime import datetime, timezone

from app.models.anomaly_evidence import AnomalyEvidence, IsolationForestEvaluationReport
from app.models.lightgbm_evidence import (
    HoldoutMetrics,
    LightGBMConfiguration,
    LightGBMEvaluationReport,
    LightGBMEvidence,
)
from app.models.network_evidence import (
    NetworkEvaluationReport,
    NetworkFinding,
    NetworkFindingCounts,
    NetworkGraphStatistics,
)
from app.models.rule_evidence import (
    RuleEvaluationReport,
    RuleSignal,
    TransactionObservation,
    TransactionRuleEvidence,
)
from app.services.risk_engine import DetectorEvidenceBundle, RiskEngine, RiskThresholds

TIMESTAMP = datetime(2025, 1, 1, tzinfo=timezone.utc)


def rule_signal(rule_id: str, reason: str = "Observable rule reason") -> RuleSignal:
    return RuleSignal(
        rule_id=rule_id,
        severity="high",
        reason=reason,
        evidence={"amount": 42_500.0},
    )


def observation(transaction_id: str, sender: str, receiver: str) -> TransactionObservation:
    return TransactionObservation(
        transaction_id=transaction_id,
        timestamp=TIMESTAMP,
        sender_account=sender,
        receiver_account=receiver,
        amount=100,
        transaction_type="transfer",
        device_id="device-1",
        sender_account_created_at=TIMESTAMP,
        receiver_account_created_at=TIMESTAMP,
        first_time_payee=True,
    )


def evidence_bundle(transaction_ids: list[str]) -> DetectorEvidenceBundle:
    rule_rows = [
        TransactionRuleEvidence(
            transaction_id=transaction_id,
            rule_signals=[rule_signal("LARGE_COLLECT_FIRST_TIME_PAYEE")] if index == 0 else [],
        )
        for index, transaction_id in enumerate(transaction_ids)
    ]
    rules = RuleEvaluationReport(
        total_transactions_evaluated=len(transaction_ids),
        transactions_with_rule_triggers=1,
        rule_trigger_counts={},
        triggered_transaction_ids_by_rule={},
        triggered_transaction_ids=[transaction_ids[0]],
        transaction_evidence=rule_rows,
    )
    anomalies = IsolationForestEvaluationReport(
        contamination=0.01,
        n_estimators=200,
        random_state=42,
        total_transactions_scored=len(transaction_ids),
        transactions_marked_anomalous=0,
        anomaly_evidence=[
            AnomalyEvidence(
                transaction_id=transaction_id,
                anomaly_score=0.1,
                is_anomalous=False,
                feature_values={},
            )
            for transaction_id in transaction_ids
        ],
    )
    ml = LightGBMEvaluationReport(
        feature_names=[],
        dataset_size=len(transaction_ids),
        train_size=max(0, len(transaction_ids) - 1),
        test_size=min(1, len(transaction_ids)),
        class_counts={"safe": 1, "fraud": 1},
        train_class_counts={"safe": 1, "fraud": 1},
        test_class_counts={"safe": 0, "fraud": 1},
        configuration=LightGBMConfiguration(
            test_size=0.2,
            random_state=42,
            prediction_threshold=0.5,
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=7,
            max_depth=4,
            min_child_samples=5,
            scale_pos_weight=1,
        ),
        metrics=HoldoutMetrics(
            precision=0,
            recall=0,
            f1=0,
            roc_auc=None,
            average_precision=None,
            confusion_matrix={"tn": 0, "fp": 0, "fn": 1, "tp": 0},
        ),
        predictions=[
            LightGBMEvidence(
                transaction_id=transaction_id,
                fraud_probability=0.5 if index == 0 else 0,
                predicted_fraud=index == 0,
                dataset_partition="test" if index == 0 else "train",
            )
            for index, transaction_id in enumerate(transaction_ids)
        ],
        predicted_fraud_count=1,
    )
    network = NetworkEvaluationReport(
        statistics=NetworkGraphStatistics(
            node_count=0,
            edge_count=0,
            transaction_count=len(transaction_ids),
            weakly_connected_component_count=0,
            finding_counts=NetworkFindingCounts(
                rapid_mule_chain=0,
                circular_flow=0,
                high_connectivity_hub=0,
            ),
        ),
        findings=[],
    )
    return DetectorEvidenceBundle(rules=rules, anomalies=anomalies, ml=ml, network=network)


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(
            RiskThresholds(medium_threshold=30, high_threshold=70, alert_threshold=30)
        )

    def test_formula_adds_explicit_detector_components(self) -> None:
        values = self.engine.score_evidence(
            rule_evidence=[rule_signal("LARGE_COLLECT_FIRST_TIME_PAYEE")],
            anomaly_evidence=AnomalyEvidence(
                transaction_id="tx",
                anomaly_score=0.5,
                is_anomalous=True,
                feature_values={},
            ),
            ml_evidence=LightGBMEvidence(
                transaction_id="tx",
                fraud_probability=0.5,
                predicted_fraud=True,
                dataset_partition="test",
            ),
            network_evidence=[],
        )

        self.assertEqual(values[:6], (40.0, "medium", 15.0, 10.0, 15.0, 0.0))
        self.assertIn("Observable rule reason", values[6])
        self.assertIn("0.50", values[6])

    def test_score_is_clamped_and_component_caps_are_respected(self) -> None:
        findings = [
            NetworkFinding(
                pattern_type=pattern,
                involved_accounts=["A"],
                transaction_ids=["tx"],
                severity="medium",
                reason="network reason",
                metrics={},
            )
            for pattern in ("rapid_mule_chain", "circular_flow", "high_connectivity_hub")
        ]
        result = self.engine.score_evidence(
            rule_evidence=[
                rule_signal("LARGE_COLLECT_FIRST_TIME_PAYEE"),
                rule_signal("RAPID_TRANSFER_BEHAVIOUR"),
                rule_signal("CIRCULAR_FLOW_PRECURSOR"),
            ],
            anomaly_evidence=AnomalyEvidence(
                transaction_id="tx",
                anomaly_score=1,
                is_anomalous=True,
                feature_values={},
            ),
            ml_evidence=LightGBMEvidence(
                transaction_id="tx",
                fraud_probability=1,
                predicted_fraud=True,
                dataset_partition="test",
            ),
            network_evidence=findings,
        )

        self.assertEqual(result[0], 100)
        self.assertEqual(result[2:6], (30, 20, 30, 20))
        self.assertGreaterEqual(result[0], 0)
        self.assertLessEqual(result[0], 100)

    def test_risk_level_threshold_boundaries(self) -> None:
        ml = lambda probability: LightGBMEvidence(
            transaction_id="tx",
            fraud_probability=probability,
            predicted_fraud=probability >= 0.5,
            dataset_partition="test",
        )
        low = self.engine.score_evidence(
            rule_evidence=[], anomaly_evidence=None, ml_evidence=ml(0.99), network_evidence=[]
        )
        medium = self.engine.score_evidence(
            rule_evidence=[], anomaly_evidence=None, ml_evidence=ml(1), network_evidence=[]
        )
        high = self.engine.score_evidence(
            rule_evidence=[
                rule_signal("LARGE_COLLECT_FIRST_TIME_PAYEE"),
                rule_signal("RAPID_TRANSFER_BEHAVIOUR"),
                rule_signal("CIRCULAR_FLOW_PRECURSOR"),
            ],
            anomaly_evidence=AnomalyEvidence(
                transaction_id="tx", anomaly_score=0, is_anomalous=False, feature_values={}
            ),
            ml_evidence=ml(1),
            network_evidence=[
                NetworkFinding(
                    pattern_type="rapid_mule_chain",
                    involved_accounts=["A"],
                    transaction_ids=["tx"],
                    severity="medium",
                    reason="path",
                    metrics={},
                )
            ],
        )

        self.assertEqual(low[1], "low")
        self.assertEqual(medium[1], "medium")
        self.assertEqual(high[1], "high")
        self.assertGreaterEqual(high[0], 70)

    def test_combine_selects_only_threshold_alerts_and_preserves_all_evidence(self) -> None:
        observations = [observation("tx-1", "A", "B"), observation("tx-2", "B", "C")]
        dataset = self.engine.combine(observations, evidence_bundle(["tx-1", "tx-2"]))

        self.assertEqual(dataset.total_transactions, 2)
        self.assertEqual(len(dataset.transaction_evidence), 2)
        self.assertEqual(dataset.selected_alert_count, 1)
        self.assertEqual(dataset.alert_ids, ["FS-tx-1"])
        self.assertEqual(dataset.transaction_evidence[0].risk_score, 30)
        self.assertEqual(dataset.transaction_evidence[1].risk_score, 0)
        self.assertEqual(dataset.transaction_evidence[1].rule_evidence, [])

    def test_risk_records_have_no_ground_truth_fields(self) -> None:
        observations = [observation("tx-1", "A", "B"), observation("tx-2", "B", "C")]
        dataset = self.engine.combine(observations, evidence_bundle(["tx-1", "tx-2"]))

        for record in dataset.transaction_evidence:
            self.assertFalse(hasattr(record, "label"))
            self.assertFalse(hasattr(record, "fraud_pattern"))
            self.assertNotIn("label", record.model_dump())
            self.assertNotIn("fraud_pattern", record.model_dump())

    def test_explanation_is_deterministic_and_uses_only_supplied_evidence(self) -> None:
        args = {
            "rule_evidence": [rule_signal("LARGE_COLLECT_FIRST_TIME_PAYEE", "Large collect signal")],
            "anomaly_evidence": None,
            "ml_evidence": None,
            "network_evidence": [],
        }
        first = self.engine.score_evidence(**args)
        second = self.engine.score_evidence(**args)

        self.assertEqual(first, second)
        self.assertIn("Large collect signal", first[6])
        self.assertNotIn("fraud_pattern", first[6])
        self.assertNotIn("label", first[6])

    def test_risk_threshold_order_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            RiskThresholds(medium_threshold=70, high_threshold=70, alert_threshold=30)


if __name__ == "__main__":
    unittest.main()