"""Request-safe payload and response contract for optional analyst explanations."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExplanationRuleEvidence(BaseModel):
    rule_id: str
    severity: Literal["low", "medium", "high"]
    reason: str
    evidence: dict[str, str | int | float | bool | list[str] | None]


class ExplanationAnomalyEvidence(BaseModel):
    anomaly_score: float
    is_anomalous: bool
    model_name: Literal["IsolationForest"]


class ExplanationMLEvidence(BaseModel):
    fraud_probability: float
    predicted_fraud: bool
    model_name: Literal["LightGBM"]


class ExplanationNetworkEvidence(BaseModel):
    pattern_type: Literal["rapid_mule_chain", "circular_flow", "high_connectivity_hub"]
    involved_accounts: list[str]
    transaction_ids: list[str]
    reason: str
    metrics: dict[str, int | float | str | list[str]]


class AlertExplanationPayload(BaseModel):
    """Explicit allow-list passed to an optional provider; excludes labels and patterns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alert_id: str
    transaction_id: str
    timestamp: datetime
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: str
    risk_score: float
    risk_level: Literal["low", "medium", "high"]
    rule_evidence: list[ExplanationRuleEvidence]
    anomaly_evidence: ExplanationAnomalyEvidence | None
    ml_evidence: ExplanationMLEvidence | None
    network_evidence: list[ExplanationNetworkEvidence]


class AnalystExplanation(BaseModel):
    """Concise analyst explanation and its provenance."""

    source: Literal["llm", "deterministic"]
    summary: str = Field(min_length=1)
    why_flagged: list[str]
    key_evidence: list[str]
    analyst_action: str = Field(min_length=1)