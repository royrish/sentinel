"""Combine independent detector evidence into demo risk scores and alert records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings, get_settings
from app.models.anomaly_evidence import IsolationForestEvaluationReport
from app.models.fraud_alert import (
    FraudAlert,
    FraudAlertDataset,
    RiskConfiguration,
    RiskLevel,
    TransactionRiskEvidence,
)
from app.models.lightgbm_evidence import LightGBMEvaluationReport
from app.models.network_evidence import NetworkEvaluationReport, NetworkFinding
from app.models.rule_evidence import RuleEvaluationReport, RuleSignal, TransactionObservation
from app.services.isolation_forest import IsolationForestDetector
from app.services.lightgbm_classifier import (
    LightGBMFraudClassifier,
    load_labeled_transactions_csv,
)
from app.services.network_analysis import NetworkThresholds, TransactionNetworkAnalyzer
from app.services.rule_engine import RuleEngine, RuleThresholds

RULE_POINTS = {
    "LARGE_COLLECT_FIRST_TIME_PAYEE": 15.0,
    "RAPID_TRANSFER_BEHAVIOUR": 10.0,
    "CIRCULAR_FLOW_PRECURSOR": 10.0,
}
NETWORK_POINTS = {
    "rapid_mule_chain": 10.0,
    "circular_flow": 10.0,
    "high_connectivity_hub": 5.0,
}


class RiskThresholds(BaseModel):
    """Configurable score bands and alert selection boundary."""

    model_config = ConfigDict(frozen=True)

    medium_threshold: float = Field(ge=0, le=100)
    high_threshold: float = Field(ge=0, le=100)
    alert_threshold: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_order(self) -> RiskThresholds:
        if self.medium_threshold >= self.high_threshold:
            raise ValueError("medium_threshold must be lower than high_threshold")
        return self

    @classmethod
    def from_settings(cls, settings: Settings) -> RiskThresholds:
        return cls(
            medium_threshold=settings.risk_medium_threshold,
            high_threshold=settings.risk_high_threshold,
            alert_threshold=settings.risk_alert_threshold,
        )


class DetectorEvidenceBundle(BaseModel):
    """The four independent structured detector reports consumed by the risk layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rules: RuleEvaluationReport
    anomalies: IsolationForestEvaluationReport
    ml: LightGBMEvaluationReport
    network: NetworkEvaluationReport


class RiskEngine:
    """Score detector evidence; never sees labels, scenario IDs, or transaction model rows."""

    def __init__(
        self,
        thresholds: RiskThresholds | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.thresholds = thresholds or RiskThresholds.from_settings(settings or get_settings())

    def score_evidence(
        self,
        *,
        rule_evidence: Sequence[RuleSignal],
        anomaly_evidence,
        ml_evidence,
        network_evidence: Sequence[NetworkFinding],
    ) -> tuple[float, RiskLevel, float, float, float, float, str]:
        """Apply the explicit 30/20/30/20 formula and return score plus evidence summary."""

        rule_component = min(
            30.0,
            sum(RULE_POINTS.get(signal.rule_id, 0.0) for signal in rule_evidence),
        )
        anomaly_component = (
            20.0 * anomaly_evidence.anomaly_score
            if anomaly_evidence is not None and anomaly_evidence.is_anomalous
            else 0.0
        )
        ml_component = (
            30.0 * ml_evidence.fraud_probability if ml_evidence is not None else 0.0
        )
        network_component = min(
            20.0,
            sum(NETWORK_POINTS.get(finding.pattern_type, 0.0) for finding in network_evidence),
        )
        risk_score = round(
            min(100.0, max(0.0, rule_component + anomaly_component + ml_component + network_component)),
            2,
        )
        if risk_score >= self.thresholds.high_threshold:
            risk_level: RiskLevel = "high"
        elif risk_score >= self.thresholds.medium_threshold:
            risk_level = "medium"
        else:
            risk_level = "low"
        summary = _build_explanation_summary(
            risk_level=risk_level,
            rule_evidence=rule_evidence,
            anomaly_evidence=anomaly_evidence,
            ml_evidence=ml_evidence,
            network_evidence=network_evidence,
        )
        return (
            risk_score,
            risk_level,
            rule_component,
            anomaly_component,
            ml_component,
            network_component,
            summary,
        )

    def combine(
        self,
        observations: Sequence[TransactionObservation],
        evidence: DetectorEvidenceBundle,
    ) -> FraudAlertDataset:
        """Join detector outputs by transaction ID and preserve full per-row evidence."""

        observation_ids = [row.transaction_id for row in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("Risk Engine requires unique transaction IDs.")
        expected_ids = set(observation_ids)
        rule_by_id = {row.transaction_id: row.rule_signals for row in evidence.rules.transaction_evidence}
        anomaly_by_id = {row.transaction_id: row for row in evidence.anomalies.anomaly_evidence}
        ml_by_id = {row.transaction_id: row for row in evidence.ml.predictions}
        network_by_id: dict[str, list[NetworkFinding]] = defaultdict(list)
        for finding in evidence.network.findings:
            for transaction_id in finding.transaction_ids:
                network_by_id[transaction_id].append(finding)

        detector_ids = {
            "rules": set(rule_by_id),
            "anomalies": set(anomaly_by_id),
            "ml": set(ml_by_id),
        }
        for detector_name, transaction_ids in detector_ids.items():
            if transaction_ids != expected_ids:
                raise ValueError(
                    f"{detector_name} evidence transaction IDs do not match source observations."
                )

        records: list[TransactionRiskEvidence] = []
        for row in sorted(observations, key=lambda item: (item.timestamp, item.transaction_id)):
            rules = rule_by_id[row.transaction_id]
            anomaly = anomaly_by_id[row.transaction_id]
            ml = ml_by_id[row.transaction_id]
            network = network_by_id[row.transaction_id]
            (
                score,
                level,
                rule_component,
                anomaly_component,
                ml_component,
                network_component,
                summary,
            ) = self.score_evidence(
                rule_evidence=rules,
                anomaly_evidence=anomaly,
                ml_evidence=ml,
                network_evidence=network,
            )
            records.append(
                TransactionRiskEvidence(
                    alert_id=f"FS-{row.transaction_id}",
                    transaction_id=row.transaction_id,
                    timestamp=row.timestamp,
                    sender_account=row.sender_account,
                    receiver_account=row.receiver_account,
                    amount=row.amount,
                    transaction_type=row.transaction_type,
                    risk_score=score,
                    risk_level=level,
                    rule_component=rule_component,
                    anomaly_component=anomaly_component,
                    ml_component=ml_component,
                    network_component=network_component,
                    rule_evidence=rules,
                    anomaly_evidence=anomaly,
                    ml_evidence=ml,
                    network_evidence=network,
                    explanation_summary=summary,
                )
            )

        alert_ids = [
            row.alert_id
            for row in records
            if row.risk_score >= self.thresholds.alert_threshold
        ]
        selected_ids = set(alert_ids)
        alerts = [
            FraudAlert.model_validate(row.model_dump())
            for row in records
            if row.alert_id in selected_ids
        ]
        return FraudAlertDataset(
            generated_at=datetime.now(timezone.utc),
            total_transactions=len(records),
            risk_configuration=RiskConfiguration(
                medium_threshold=self.thresholds.medium_threshold,
                high_threshold=self.thresholds.high_threshold,
                alert_threshold=self.thresholds.alert_threshold,
            ),
            selected_alert_count=len(alert_ids),
            alert_ids=alert_ids,
            alerts=alerts,
            transaction_evidence=records,
        )


def _build_explanation_summary(
    *,
    risk_level: RiskLevel,
    rule_evidence: Sequence[RuleSignal],
    anomaly_evidence,
    ml_evidence,
    network_evidence: Sequence[NetworkFinding],
) -> str:
    prefix = {
        "low": "Low risk; continue routine monitoring.",
        "medium": "Medium risk; review the supporting evidence.",
        "high": "High risk / investigation recommended.",
    }[risk_level]
    reasons = [signal.reason for signal in rule_evidence]
    if anomaly_evidence is not None and anomaly_evidence.is_anomalous:
        reasons.append("Isolation Forest flagged unusual behavioural activity")
    if ml_evidence is not None and ml_evidence.fraud_probability > 0:
        reasons.append(
            f"LightGBM model probability was {ml_evidence.fraud_probability:.2f}"
        )
    reasons.extend(finding.reason for finding in network_evidence)
    if not reasons:
        reasons.append("no detector signal contributed evidence")
    return f"{prefix} " + "; ".join(dict.fromkeys(reasons)) + "."


def run_detectors(
    input_path: Path,
    *,
    settings: Settings | None = None,
) -> tuple[list[TransactionObservation], DetectorEvidenceBundle]:
    """Run each existing detector independently over the configured transaction CSV."""

    effective_settings = settings or get_settings()
    labeled_observations = load_labeled_transactions_csv(input_path)
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
        for row in labeled_observations
    ]
    rules = RuleEngine(RuleThresholds.from_settings(effective_settings)).evaluate_observations(
        observations
    )
    anomalies = IsolationForestDetector(settings=effective_settings).fit_score(observations)
    ml = LightGBMFraudClassifier(settings=effective_settings).fit_evaluate(labeled_observations)
    network = TransactionNetworkAnalyzer(
        NetworkThresholds.from_settings(effective_settings)
    ).analyze(observations)
    return observations, DetectorEvidenceBundle(
        rules=rules,
        anomalies=anomalies,
        ml=ml,
        network=network,
    )


def generate_fraud_alert_dataset(
    input_path: Path,
    output_path: Path,
    *,
    settings: Settings | None = None,
) -> FraudAlertDataset:
    """Run all detectors independently, combine their evidence, and save a JSON artifact."""

    risk_engine = RiskEngine(settings=settings)
    observations, detector_evidence = run_detectors(input_path, settings=settings)
    dataset = risk_engine.combine(observations, detector_evidence)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return dataset


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
        default=Path(__file__).resolve().parents[2] / "data" / "fraud_alerts.json",
    )
    args = parser.parse_args()
    dataset = generate_fraud_alert_dataset(args.input, args.output)
    by_level = {
        level: sum(
            row.risk_level == level
            for row in dataset.transaction_evidence
            if row.alert_id in set(dataset.alert_ids)
        )
        for level in ("high", "medium", "low")
    }
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "total_transactions": dataset.total_transactions,
                "selected_alert_count": dataset.selected_alert_count,
                "alerts_by_risk_level": by_level,
                "risk_configuration": dataset.risk_configuration.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()