"""Observation-only Isolation Forest behavioural anomaly signal."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest

from app.config import Settings, get_settings
from app.models.anomaly_evidence import (
    AnomalyEvidence,
    AnomalyFeatureValue,
    IsolationForestEvaluationReport,
)
from app.models.rule_evidence import TransactionObservation

TRANSACTION_TYPES = ("transfer", "collect_request", "merchant_payment", "bill_payment")
FEATURE_NAMES = (
    "log_amount",
    "hour_sin",
    "hour_cos",
    *(f"transaction_type_{transaction_type}" for transaction_type in TRANSACTION_TYPES),
    "first_time_payee",
    "sender_account_age_days",
    "receiver_account_age_days",
    "sender_outgoing_count_1h",
    "receiver_incoming_count_1h",
)
MODEL_NAME = "IsolationForest"
VELOCITY_WINDOW = timedelta(hours=1)


@dataclass(frozen=True)
class BehavioralFeatures:
    """Aligned numeric matrix and interpretable feature values for observations."""

    transaction_ids: tuple[str, ...]
    matrix: NDArray[np.float64]
    feature_values: tuple[dict[str, AnomalyFeatureValue], ...]


class IsolationForestDetector:
    """Fit and apply an unsupervised Isolation Forest to observable features only."""

    def __init__(
        self,
        *,
        contamination: float | None = None,
        n_estimators: int | None = None,
        random_state: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        settings = settings or get_settings()
        self.contamination = (
            contamination if contamination is not None else settings.isolation_forest_contamination
        )
        self.n_estimators = (
            n_estimators if n_estimators is not None else settings.isolation_forest_n_estimators
        )
        self.random_state = random_state if random_state is not None else settings.isolation_forest_random_state
        self._validate_configuration()
        self.model: IsolationForest | None = None
        self._decision_center = 0.0
        self._decision_scale = 1.0

    def _validate_configuration(self) -> None:
        if not 0 < float(self.contamination) <= 0.5:
            raise ValueError("contamination must be a number in (0, 0.5].")
        if self.n_estimators < 10:
            raise ValueError("n_estimators must be at least 10.")

    @staticmethod
    def extract_features(
        observations: Sequence[TransactionObservation],
    ) -> BehavioralFeatures:
        """Extract numeric behavioural features, excluding labels by input contract."""

        ids = [row.transaction_id for row in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Feature extraction requires unique transaction IDs.")
        for row in observations:
            if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
                raise ValueError(f"{row.transaction_id} timestamp must include a timezone.")
            if row.sender_account_created_at.tzinfo is None or row.receiver_account_created_at.tzinfo is None:
                raise ValueError(f"{row.transaction_id} account creation timestamps must include a timezone.")
            if row.transaction_type not in TRANSACTION_TYPES:
                raise ValueError(f"{row.transaction_id} has an unsupported transaction type.")

        ordered = sorted(observations, key=lambda row: (row.timestamp, row.transaction_id))
        outgoing_windows: dict[str, deque] = defaultdict(deque)
        incoming_windows: dict[str, deque] = defaultdict(deque)
        rows_by_id: dict[str, dict[str, AnomalyFeatureValue]] = {}

        for row in ordered:
            cutoff = row.timestamp - VELOCITY_WINDOW
            sender_events = outgoing_windows[row.sender_account]
            receiver_events = incoming_windows[row.receiver_account]
            while sender_events and sender_events[0] < cutoff:
                sender_events.popleft()
            while receiver_events and receiver_events[0] < cutoff:
                receiver_events.popleft()

            sender_outgoing_count = len(sender_events) + 1
            receiver_incoming_count = len(receiver_events) + 1
            sender_events.append(row.timestamp)
            receiver_events.append(row.timestamp)

            hour = row.timestamp.hour + row.timestamp.minute / 60 + row.timestamp.second / 3600
            hour_angle = 2 * np.pi * hour / 24
            values: dict[str, AnomalyFeatureValue] = {
                "log_amount": float(np.log1p(row.amount)),
                "hour_sin": float(np.sin(hour_angle)),
                "hour_cos": float(np.cos(hour_angle)),
                "first_time_payee": row.first_time_payee,
                "sender_account_age_days": max(
                    (row.timestamp - row.sender_account_created_at).total_seconds() / 86_400,
                    0.0,
                ),
                "receiver_account_age_days": max(
                    (row.timestamp - row.receiver_account_created_at).total_seconds() / 86_400,
                    0.0,
                ),
                "sender_outgoing_count_1h": float(sender_outgoing_count),
                "receiver_incoming_count_1h": float(receiver_incoming_count),
            }
            values.update(
                {
                    f"transaction_type_{transaction_type}": float(
                        row.transaction_type == transaction_type
                    )
                    for transaction_type in TRANSACTION_TYPES
                }
            )
            rows_by_id[row.transaction_id] = values

        feature_values = tuple(rows_by_id[row.transaction_id] for row in ordered)
        matrix = np.asarray(
            [[float(values[name]) for name in FEATURE_NAMES] for values in feature_values],
            dtype=np.float64,
        )
        return BehavioralFeatures(
            transaction_ids=tuple(row.transaction_id for row in ordered),
            matrix=matrix.reshape((len(ordered), len(FEATURE_NAMES))),
            feature_values=feature_values,
        )

    def fit_score(
        self, observations: Sequence[TransactionObservation]
    ) -> IsolationForestEvaluationReport:
        """Fit on the supplied observation population and emit evidence per row."""

        if not observations:
            raise ValueError("Cannot fit Isolation Forest on an empty transaction batch.")
        features = self.extract_features(observations)
        if len(features.transaction_ids) < 2:
            raise ValueError("At least two transactions are required to fit Isolation Forest.")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        predictions = self.model.fit_predict(features.matrix)
        decision_values = self.model.decision_function(features.matrix)
        self._decision_center = float(np.median(decision_values))
        median_absolute_deviation = float(
            np.median(np.abs(decision_values - self._decision_center))
        )
        scale = median_absolute_deviation * 1.4826
        if scale <= np.finfo(np.float64).eps:
            scale = float(np.std(decision_values))
        self._decision_scale = max(scale, np.finfo(np.float64).eps)

        evidence = [
            AnomalyEvidence(
                transaction_id=transaction_id,
                anomaly_score=self._normalize_score(float(decision_value)),
                is_anomalous=int(prediction) == -1,
                model_name=MODEL_NAME,
                feature_values=feature_values,
            )
            for transaction_id, decision_value, prediction, feature_values in zip(
                features.transaction_ids,
                decision_values,
                predictions,
                features.feature_values,
                strict=True,
            )
        ]
        return IsolationForestEvaluationReport(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            total_transactions_scored=len(evidence),
            transactions_marked_anomalous=sum(item.is_anomalous for item in evidence),
            anomaly_evidence=evidence,
        )

    def _normalize_score(self, decision_value: float) -> float:
        standardized_distance = (decision_value - self._decision_center) / self._decision_scale
        return float(1 / (1 + np.exp(np.clip(standardized_distance, -60, 60))))


def load_observations_csv(path: Path) -> list[TransactionObservation]:
    """Load observable columns only; label/fraud_pattern are neither parsed nor used."""

    required_fields = set(TransactionObservation.model_fields)
    observations: list[TransactionObservation] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required_fields.difference(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required observable columns: " + ", ".join(missing))
        for line_number, row in enumerate(reader, start=2):
            try:
                observations.append(TransactionObservation.model_validate(row))
            except ValidationError as error:
                raise ValueError(f"Invalid observable data on CSV line {line_number}: {error}") from error
    if not observations:
        raise ValueError(f"CSV contains no transactions: {path}")
    return observations


def evaluate_csv(input_path: Path, output_path: Path) -> IsolationForestEvaluationReport:
    """Score a CSV and write a separate JSON artifact without modifying the input."""

    observations = load_observations_csv(input_path)
    report = IsolationForestDetector().fit_score(observations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


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
        default=Path(__file__).resolve().parents[2] / "data" / "isolation_forest_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate_csv(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "model_name": report.model_name,
                "contamination": report.contamination,
                "n_estimators": report.n_estimators,
                "random_state": report.random_state,
                "total_transactions_scored": report.total_transactions_scored,
                "transactions_marked_anomalous": report.transactions_marked_anomalous,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()