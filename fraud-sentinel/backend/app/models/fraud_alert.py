"""Unified, typed alert and response contracts for the integration layer."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.anomaly_evidence import AnomalyEvidence
from app.models.lightgbm_evidence import LightGBMEvidence
from app.models.network_evidence import NetworkFinding
from app.models.rule_evidence import RuleSignal
from app.models.transaction import TransactionType

RiskLevel = Literal["low", "medium", "high"]
AlertStatus = Literal["open", "confirmed_fraud", "marked_safe"]
FeedbackStatus = Literal["confirmed_fraud", "marked_safe"]


class RiskConfiguration(BaseModel):
    medium_threshold: float
    high_threshold: float
    alert_threshold: float
    rule_component_max: float = 30
    anomaly_component_max: float = 20
    ml_component_max: float = 30
    network_component_max: float = 20


class TransactionRiskEvidence(BaseModel):
    """Complete per-transaction evidence, including transactions below alert cutoff."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    transaction_id: str
    timestamp: datetime
    sender_account: str
    receiver_account: str
    amount: float
    transaction_type: TransactionType
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    rule_component: float = Field(ge=0, le=30)
    anomaly_component: float = Field(ge=0, le=20)
    ml_component: float = Field(ge=0, le=30)
    network_component: float = Field(ge=0, le=20)
    rule_evidence: list[RuleSignal] = Field(default_factory=list)
    anomaly_evidence: AnomalyEvidence | None = None
    ml_evidence: LightGBMEvidence | None = None
    network_evidence: list[NetworkFinding] = Field(default_factory=list)
    explanation_summary: str


class FraudAlert(TransactionRiskEvidence):
    """Selected, human-reviewable alert. Status never defaults to a fraud verdict."""

    status: AlertStatus = "open"


class FraudAlertDataset(BaseModel):
    """Complete batch result: all transaction evidence plus selected alert references."""

    generated_at: datetime
    total_transactions: int
    risk_configuration: RiskConfiguration
    selected_alert_count: int
    alert_ids: list[str]
    alerts: list[FraudAlert]
    transaction_evidence: list[TransactionRiskEvidence]


class AlertsResponse(BaseModel):
    total: int
    items: list[FraudAlert]


class FeedbackRequest(BaseModel):
    alert_id: str = Field(min_length=1)
    status: FeedbackStatus


class FeedbackResponse(BaseModel):
    alert_id: str
    status: FeedbackStatus


class AlertSummaryResponse(BaseModel):
    total_alerts: int
    high_risk: int
    medium_risk: int
    low_risk: int
    open_alerts: int
    confirmed_fraud: int
    marked_safe: int


class GraphNode(BaseModel):
    id: str
    type: Literal["account"] = "account"


class GraphEdge(BaseModel):
    source: str
    target: str
    transaction_id: str
    amount: float
    timestamp: datetime


class AlertGraphResponse(BaseModel):
    alert_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]