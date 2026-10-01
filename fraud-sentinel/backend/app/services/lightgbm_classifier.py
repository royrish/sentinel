"""Supervised LightGBM fraud-probability signal with strict target/feature separation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from lightgbm import LGBMClassifier
from pydantic import ValidationError
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.config import Settings, get_settings
from app.models.lightgbm_evidence import (
    HoldoutMetrics,
    LightGBMConfiguration,
    LightGBMEvaluationReport,
    LightGBMEvidence,
    LabeledTransactionObservation,
)
from app.models.rule_evidence import TransactionObservation
from app.models.transaction import Transaction
from app.services.isolation_forest import FEATURE_NAMES, IsolationForestDetector

MODEL_NAME = "LightGBM"
TARGET_LABELS = {"safe": 0, "fraud": 1}


@dataclass(frozen=True)
class SupervisedFeatures:
    """Feature matrix and ground-truth target maintained as separate arrays."""

    transaction_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    matrix: np.ndarray
    target: np.ndarray


class LightGBMFraudClassifier:
    """Train a binary model signal; the model output is not a final fraud decision."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        test_size: float | None = None,
        random_state: int | None = None,
        prediction_threshold: float | None = None,
    ) -> None:
        effective_settings = settings or get_settings()
        self.test_size = (
            test_size if test_size is not None else effective_settings.lightgbm_test_size
        )
        self.random_state = (
            random_state
            if random_state is not None
            else effective_settings.lightgbm_random_state
        )
        self.prediction_threshold = (
            prediction_threshold
            if prediction_threshold is not None
            else effective_settings.lightgbm_threshold
        )
        self.n_estimators = effective_settings.lightgbm_n_estimators
        self.learning_rate = effective_settings.lightgbm_learning_rate
        self.num_leaves = effective_settings.lightgbm_num_leaves
        self.max_depth = effective_settings.lightgbm_max_depth
        self.min_child_samples = effective_settings.lightgbm_min_child_samples
        self._validate_configuration()
        self.model: LGBMClassifier | None = None

    def _validate_configuration(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if not 0 <= self.prediction_threshold <= 1:
            raise ValueError("prediction_threshold must be between 0 and 1.")

    @staticmethod
    def extract_features(
        transactions: Sequence[Transaction | LabeledTransactionObservation],
    ) -> SupervisedFeatures:
        """Extract behavioral columns and a separate target; IDs/labels/patterns are not features."""

        ids = [transaction.transaction_id for transaction in transactions]
        if not transactions:
            raise ValueError("Cannot extract LightGBM features from an empty dataset.")
        if len(ids) != len(set(ids)):
            raise ValueError("LightGBM feature extraction requires unique transaction IDs.")
        invalid_labels = sorted({row.label for row in transactions if row.label not in TARGET_LABELS})
        if invalid_labels:
            raise ValueError(f"Unsupported target labels: {invalid_labels}")

        observations = [
            TransactionObservation(
                transaction_id=row.transaction_id,
                timestamp=row.timestamp,
                sender_account=row.sender_account,
                receiver_account=row.receiver_account,
                amount=row.amount,
                transaction_type=row.transaction_type,
                device_id=row.device_id,
                sender_account_created_at=row.sender_account_created_at,
                receiver_account_created_at=row.receiver_account_created_at,
                first_time_payee=row.first_time_payee,
            )
            for row in transactions
        ]
        behavioral_features = IsolationForestDetector.extract_features(observations)
        targets_by_id = {row.transaction_id: TARGET_LABELS[row.label] for row in transactions}
        target = np.asarray(
            [targets_by_id[transaction_id] for transaction_id in behavioral_features.transaction_ids],
            dtype=np.int8,
        )
        return SupervisedFeatures(
            transaction_ids=behavioral_features.transaction_ids,
            feature_names=tuple(FEATURE_NAMES),
            matrix=behavioral_features.matrix,
            target=target,
        )

    def fit_evaluate(
        self, transactions: Sequence[Transaction | LabeledTransactionObservation]
    ) -> LightGBMEvaluationReport:
        """Train on a stratified training partition and calculate metrics on held-out rows."""

        features = self.extract_features(transactions)
        class_counts = Counter(int(value) for value in features.target)
        if class_counts[0] < 2 or class_counts[1] < 2:
            raise ValueError("At least two safe and two fraud labels are required for a stratified split.")

        indices = np.arange(len(features.transaction_ids))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=features.target,
        )
        y_train = features.target[train_indices]
        y_test = features.target[test_indices]
        train_positive_count = int(np.sum(y_train == 1))
        train_negative_count = int(np.sum(y_train == 0))
        if train_positive_count == 0 or train_negative_count == 0:
            raise ValueError("Stratified training split must contain both target classes.")
        scale_pos_weight = train_negative_count / train_positive_count

        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        self.model.fit(features.matrix[train_indices], y_train)

        test_probabilities = np.asarray(
            self.model.predict_proba(features.matrix[test_indices])
        )[:, 1]
        test_predictions = (test_probabilities >= self.prediction_threshold).astype(np.int8)
        confusion_values = confusion_matrix(y_test, test_predictions, labels=[0, 1]).ravel()
        try:
            roc_auc: float | None = float(roc_auc_score(y_test, test_probabilities))
        except ValueError:
            roc_auc = None
        try:
            average_precision: float | None = float(
                average_precision_score(y_test, test_probabilities)
            )
        except ValueError:
            average_precision = None
        metrics = HoldoutMetrics(
            precision=float(precision_score(y_test, test_predictions, zero_division=0)),
            recall=float(recall_score(y_test, test_predictions, zero_division=0)),
            f1=float(f1_score(y_test, test_predictions, zero_division=0)),
            roc_auc=roc_auc,
            average_precision=average_precision,
            confusion_matrix={
                "tn": int(confusion_values[0]),
                "fp": int(confusion_values[1]),
                "fn": int(confusion_values[2]),
                "tp": int(confusion_values[3]),
            },
        )

        all_probabilities = np.asarray(self.model.predict_proba(features.matrix))[:, 1]
        partition_by_index: dict[int, Literal["train", "test"]] = {
            **{int(index): "train" for index in train_indices},
            **{int(index): "test" for index in test_indices},
        }
        evidence = [
            LightGBMEvidence(
                transaction_id=transaction_id,
                fraud_probability=float(probability),
                predicted_fraud=bool(probability >= self.prediction_threshold),
                model_name=MODEL_NAME,
                dataset_partition=partition_by_index[index],
            )
            for index, (transaction_id, probability) in enumerate(
                zip(features.transaction_ids, all_probabilities, strict=True)
            )
        ]
        train_counts = _target_counts(y_train)
        test_counts = _target_counts(y_test)
        return LightGBMEvaluationReport(
            feature_names=list(features.feature_names),
            dataset_size=len(features.transaction_ids),
            train_size=len(train_indices),
            test_size=len(test_indices),
            class_counts={"safe": class_counts[0], "fraud": class_counts[1]},
            train_class_counts=train_counts,
            test_class_counts=test_counts,
            configuration=LightGBMConfiguration(
                test_size=self.test_size,
                random_state=self.random_state,
                prediction_threshold=self.prediction_threshold,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                scale_pos_weight=scale_pos_weight,
            ),
            metrics=metrics,
            predictions=evidence,
            predicted_fraud_count=sum(item.predicted_fraud for item in evidence),
        )


def load_labeled_transactions_csv(path: Path) -> list[LabeledTransactionObservation]:
    """Load observable fields plus target, silently excluding fraud_pattern metadata."""

    required_fields = set(LabeledTransactionObservation.model_fields)
    transactions: list[LabeledTransactionObservation] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required_fields.difference(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required transaction columns: " + ", ".join(missing))
        for line_number, row in enumerate(reader, start=2):
            try:
                transactions.append(LabeledTransactionObservation.model_validate(row))
            except ValidationError as error:
                raise ValueError(f"Invalid transaction data on CSV line {line_number}: {error}") from error
    if not transactions:
        raise ValueError(f"CSV contains no transactions: {path}")
    return transactions


def evaluate_csv(
    input_path: Path,
    output_path: Path,
    *,
    settings: Settings | None = None,
) -> LightGBMEvaluationReport:
    """Fit/evaluate one stratified split and write a separate JSON result artifact."""

    transactions = load_labeled_transactions_csv(input_path)
    report = LightGBMFraudClassifier(settings=settings).fit_evaluate(transactions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _target_counts(target: np.ndarray) -> dict[Literal["safe", "fraud"], int]:
    counts = Counter(int(value) for value in target)
    return {"safe": counts[0], "fraud": counts[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic_transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "lightgbm_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate_csv(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "dataset_size": report.dataset_size,
                "train_size": report.train_size,
                "test_size": report.test_size,
                "class_counts": report.class_counts,
                "train_class_counts": report.train_class_counts,
                "test_class_counts": report.test_class_counts,
                "configuration": report.configuration.model_dump(),
                "holdout_metrics": report.metrics.model_dump(),
                "predictions": len(report.predictions),
                "predicted_fraud_count": report.predicted_fraud_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()