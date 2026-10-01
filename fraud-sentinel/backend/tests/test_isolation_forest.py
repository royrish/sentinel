"""Tests for the unsupervised, observation-only Isolation Forest signal."""

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.data.synthetic_transactions import generate_dataset
from app.models.anomaly_evidence import AnomalyEvidence, IsolationForestEvaluationReport
from app.models.rule_evidence import TransactionObservation
from app.services.isolation_forest import (
    FEATURE_NAMES,
    IsolationForestDetector,
    evaluate_csv,
    load_observations_csv,
)


def observations_from_synthetic(count: int = 35) -> list[TransactionObservation]:
    dataset, _ = generate_dataset(seed=2025, days=14, baseline_count=count)
    return [
        TransactionObservation.model_validate(
            transaction.model_dump(exclude={"label", "fraud_pattern"})
        )
        for transaction in dataset.transactions
    ]


class IsolationForestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observations = observations_from_synthetic()

    def test_feature_extraction_encodes_categories_and_behaviour(self) -> None:
        features = IsolationForestDetector.extract_features(self.observations)

        self.assertEqual(features.matrix.shape, (len(self.observations), len(FEATURE_NAMES)))
        self.assertEqual(features.transaction_ids, tuple(sorted(
            (row.transaction_id for row in self.observations),
            key=lambda transaction_id: next(
                row.timestamp for row in self.observations if row.transaction_id == transaction_id
            ),
        )))
        self.assertIn("log_amount", features.feature_values[0])
        self.assertIn("sender_outgoing_count_1h", features.feature_values[0])
        self.assertIn("receiver_incoming_count_1h", features.feature_values[0])
        self.assertEqual(
            sum(
                float(features.feature_values[0][f"transaction_type_{kind}"])
                for kind in ("transfer", "collect_request", "merchant_payment", "bill_payment")
            ),
            1,
        )

    def test_fit_returns_typed_per_transaction_evidence(self) -> None:
        report = IsolationForestDetector(
            contamination=0.1, n_estimators=40, random_state=13
        ).fit_score(self.observations)

        self.assertIsInstance(report, IsolationForestEvaluationReport)
        self.assertEqual(report.total_transactions_scored, len(self.observations))
        self.assertEqual(len(report.anomaly_evidence), len(self.observations))
        self.assertIsInstance(report.anomaly_evidence[0], AnomalyEvidence)
        self.assertTrue(all(0 <= row.anomaly_score <= 1 for row in report.anomaly_evidence))
        self.assertTrue(all(row.model_name == "IsolationForest" for row in report.anomaly_evidence))
        self.assertEqual(
            report.transactions_marked_anomalous,
            sum(row.is_anomalous for row in report.anomaly_evidence),
        )

    def test_fixed_seed_produces_repeatable_evidence(self) -> None:
        detector_options = {"contamination": 0.1, "n_estimators": 40, "random_state": 23}
        first = IsolationForestDetector(**detector_options).fit_score(self.observations)
        second = IsolationForestDetector(**detector_options).fit_score(self.observations)

        self.assertEqual(first, second)

    def test_environment_settings_control_contamination_and_model_options(self) -> None:
        settings = Settings(
            isolation_forest_contamination=0.2,
            isolation_forest_n_estimators=35,
            isolation_forest_random_state=9,
        )
        report = IsolationForestDetector(settings=settings).fit_score(self.observations)

        self.assertEqual(report.contamination, 0.2)
        self.assertEqual(report.n_estimators, 35)
        self.assertEqual(report.random_state, 9)
        with self.assertRaises(ValidationError):
            Settings(isolation_forest_contamination=0.0)

    def test_ground_truth_changes_do_not_change_features_or_model_input(self) -> None:
        dataset, _ = generate_dataset(seed=32, days=14, baseline_count=30)
        rows = [transaction.model_dump(mode="json") for transaction in dataset.transactions]
        fieldnames = list(rows[0])

        with tempfile.TemporaryDirectory() as temporary_directory:
            original_path = Path(temporary_directory) / "original.csv"
            changed_path = Path(temporary_directory) / "changed_truth.csv"
            for path, change_truth in ((original_path, False), (changed_path, True)):
                with path.open("w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for index, original in enumerate(rows):
                        row = dict(original)
                        if change_truth:
                            row["label"] = "safe" if original["label"] == "fraud" else "fraud"
                            row["fraud_pattern"] = (
                                "circular_money_flow"
                                if index % 2
                                else "rapid_mule_chain"
                            )
                        writer.writerow(row)

            original_observations = load_observations_csv(original_path)
            changed_observations = load_observations_csv(changed_path)
            original_features = IsolationForestDetector.extract_features(original_observations)
            changed_features = IsolationForestDetector.extract_features(changed_observations)
            self.assertTrue((original_features.matrix == changed_features.matrix).all())
            original_report = IsolationForestDetector(
                contamination=0.1, n_estimators=30, random_state=4
            ).fit_score(original_observations)
            changed_report = IsolationForestDetector(
                contamination=0.1, n_estimators=30, random_state=4
            ).fit_score(changed_observations)
            self.assertEqual(original_report, changed_report)

    def test_batch_inference_writes_separate_json_and_preserves_input(self) -> None:
        dataset, _ = generate_dataset(seed=45, days=14, baseline_count=25)
        rows = [transaction.model_dump(mode="json") for transaction in dataset.transactions]
        fieldnames = list(rows[0])

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "transactions.csv"
            output_path = Path(temporary_directory) / "scores.json"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            input_before = input_path.read_bytes()

            report = evaluate_csv(input_path, output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(input_path.read_bytes(), input_before)
            self.assertEqual(report.total_transactions_scored, len(rows))
            self.assertEqual(
                IsolationForestEvaluationReport.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                ),
                report,
            )

    def test_empty_batch_and_invalid_observations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty transaction batch"):
            IsolationForestDetector().fit_score([])
        self.assertEqual(
            IsolationForestDetector.extract_features([]).matrix.shape,
            (0, len(FEATURE_NAMES)),
        )
        invalid = self.observations[0].model_copy(
            update={"timestamp": datetime(2025, 1, 1)}
        )
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            IsolationForestDetector.extract_features([invalid])

    def test_invalid_contamination_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contamination"):
            IsolationForestDetector(contamination=0.7)


if __name__ == "__main__":
    unittest.main()