"""Tests for the supervised LightGBM model signal."""

import csv
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from app.config import Settings
from app.data.synthetic_transactions import generate_dataset
from app.models.lightgbm_evidence import LightGBMEvaluationReport, LightGBMEvidence
from app.services.isolation_forest import FEATURE_NAMES
from app.services.lightgbm_classifier import (
    LightGBMFraudClassifier,
    evaluate_csv,
    load_labeled_transactions_csv,
)
from app.models.transaction import Transaction


def synthetic_transactions(baseline_count: int = 35) -> list[Transaction]:
    dataset, _ = generate_dataset(seed=991, days=14, baseline_count=baseline_count)
    return list(dataset.transactions)


def make_test_settings() -> Settings:
    return Settings(
        lightgbm_test_size=0.2,
        lightgbm_random_state=37,
        lightgbm_threshold=0.45,
        lightgbm_n_estimators=24,
        lightgbm_learning_rate=0.05,
        lightgbm_num_leaves=7,
        lightgbm_max_depth=4,
        lightgbm_min_child_samples=2,
    )


class LightGBMClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transactions = synthetic_transactions()
        self.settings = make_test_settings()

    def test_feature_matrix_excludes_ids_labels_and_patterns(self) -> None:
        features = LightGBMFraudClassifier.extract_features(self.transactions)

        self.assertEqual(features.matrix.shape, (len(self.transactions), len(FEATURE_NAMES)))
        self.assertEqual(features.feature_names, tuple(FEATURE_NAMES))
        for excluded in ("transaction_id", "label", "fraud_pattern"):
            self.assertNotIn(excluded, features.feature_names)
        self.assertEqual(len(features.target), len(self.transactions))

    def test_label_and_pattern_mutations_do_not_change_features(self) -> None:
        changed = [
            transaction.model_copy(
                update={
                    "label": "safe" if transaction.label == "fraud" else "fraud",
                    "fraud_pattern": (
                        "circular_money_flow"
                        if transaction.fraud_pattern != "circular_money_flow"
                        else "rapid_mule_chain"
                    ),
                }
            )
            for transaction in self.transactions
        ]

        original_features = LightGBMFraudClassifier.extract_features(self.transactions)
        changed_features = LightGBMFraudClassifier.extract_features(changed)

        self.assertTrue((original_features.matrix == changed_features.matrix).all())
        self.assertEqual(original_features.feature_names, changed_features.feature_names)
        self.assertFalse((original_features.target == changed_features.target).all())

    def test_stratified_split_is_reproducible_and_tracks_class_counts(self) -> None:
        first = LightGBMFraudClassifier(settings=self.settings).fit_evaluate(self.transactions)
        second = LightGBMFraudClassifier(settings=self.settings).fit_evaluate(self.transactions)

        self.assertEqual(first, second)
        self.assertEqual(first.dataset_size, len(self.transactions))
        self.assertEqual(first.train_size + first.test_size, first.dataset_size)
        self.assertGreater(first.train_class_counts["fraud"], 0)
        self.assertGreater(first.test_class_counts["fraud"], 0)
        self.assertEqual(first.class_counts["fraud"], 9)

    def test_training_uses_scale_positive_weight_and_configurable_threshold(self) -> None:
        report = LightGBMFraudClassifier(settings=self.settings).fit_evaluate(self.transactions)

        expected_weight = report.train_class_counts["safe"] / report.train_class_counts["fraud"]
        self.assertEqual(report.configuration.scale_pos_weight, expected_weight)
        self.assertEqual(report.configuration.prediction_threshold, 0.45)
        self.assertEqual(report.configuration.random_state, 37)
        self.assertEqual(report.configuration.n_estimators, 24)
        for evidence in report.predictions:
            self.assertEqual(evidence.predicted_fraud, evidence.fraud_probability >= 0.45)

    def test_evidence_schema_probability_bounds_and_holdout_metrics(self) -> None:
        report = LightGBMFraudClassifier(settings=self.settings).fit_evaluate(self.transactions)

        self.assertIsInstance(report, LightGBMEvaluationReport)
        self.assertTrue(all(isinstance(row, LightGBMEvidence) for row in report.predictions))
        self.assertEqual(len(report.predictions), len(self.transactions))
        self.assertTrue(all(0 <= row.fraud_probability <= 1 for row in report.predictions))
        self.assertTrue(all(row.model_name == "LightGBM" for row in report.predictions))
        self.assertEqual(
            sum(row.dataset_partition == "test" for row in report.predictions),
            report.test_size,
        )
        self.assertEqual(
            sum(report.metrics.confusion_matrix.values()),
            report.test_size,
        )
        self.assertEqual(report.metrics.evaluation_scope, "synthetic_dataset_holdout")

    def test_batch_inference_artifact_does_not_modify_source_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "transactions.csv"
            output_path = Path(temporary_directory) / "lightgbm.json"
            rows = [transaction.model_dump(mode="json") for transaction in self.transactions]
            for row in rows:
                row["fraud_pattern"] = "scenario-metadata-must-be-ignored"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            original_bytes = input_path.read_bytes()

            report = evaluate_csv(input_path, output_path, settings=self.settings)

            self.assertTrue(output_path.exists())
            self.assertEqual(input_path.read_bytes(), original_bytes)
            self.assertEqual(report.dataset_size, len(rows))
            self.assertFalse(hasattr(load_labeled_transactions_csv(input_path)[0], "fraud_pattern"))
            self.assertEqual(
                LightGBMEvaluationReport.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                ),
                report,
            )

    def test_empty_or_insufficient_class_data_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty dataset"):
            LightGBMFraudClassifier.extract_features([])
        safe_rows = [row for row in self.transactions if row.label == "safe"]
        with self.assertRaisesRegex(ValueError, "At least two safe and two fraud"):
            LightGBMFraudClassifier(settings=self.settings).fit_evaluate(safe_rows)

    def test_configuration_validation_and_settings_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "test_size"):
            LightGBMFraudClassifier(settings=self.settings, test_size=1.0)
        with self.assertRaisesRegex(ValueError, "prediction_threshold"):
            LightGBMFraudClassifier(settings=self.settings, prediction_threshold=1.1)
        with self.assertRaises(ValidationError):
            Settings(lightgbm_test_size=0.0)


if __name__ == "__main__":
    unittest.main()