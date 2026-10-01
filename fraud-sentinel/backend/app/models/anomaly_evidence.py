"""Structured output contracts for observation-based anomaly signals."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

AnomalyFeatureValue: TypeAlias = str | int | float | bool


class AnomalyEvidence(BaseModel):
    """One Isolation Forest signal; this is not a fraud classification."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    anomaly_score: float = Field(ge=0, le=1)
    is_anomalous: bool
    model_name: Literal["IsolationForest"] = "IsolationForest"
    feature_values: dict[str, AnomalyFeatureValue]


class IsolationForestEvaluationReport(BaseModel):
    """Batch anomaly output plus the unsupervised model configuration used."""

    model_name: Literal["IsolationForest"] = "IsolationForest"
    score_interpretation: Literal["Higher scores indicate more anomalous behaviour."] = (
        "Higher scores indicate more anomalous behaviour."
    )
    contamination: float
    n_estimators: int
    random_state: int
    total_transactions_scored: int
    transactions_marked_anomalous: int
    anomaly_evidence: list[AnomalyEvidence]