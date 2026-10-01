"""Typed observation inputs and transparent rule-engine evidence models."""

from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from app.models.transaction import TransactionType

Severity = Literal["low", "medium", "high"]
EvidenceValue: TypeAlias = str | int | float | bool | list[str] | None


class TransactionObservation(BaseModel):
    """Only observable transaction attributes available to deterministic rules."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str = Field(min_length=1)
    timestamp: datetime
    sender_account: str = Field(min_length=1)
    receiver_account: str = Field(min_length=1)
    amount: float = Field(gt=0)
    transaction_type: TransactionType
    device_id: str = Field(min_length=1)
    sender_account_created_at: datetime
    receiver_account_created_at: datetime
    first_time_payee: bool


class RuleSignal(BaseModel):
    """One deterministic rule finding, with no combined score or classification."""

    rule_id: str = Field(min_length=1)
    triggered: Literal[True] = True
    severity: Severity
    reason: str = Field(min_length=1)
    evidence: dict[str, EvidenceValue]


class TransactionRuleEvidence(BaseModel):
    """Rule findings associated with one evaluated transaction."""

    transaction_id: str
    rule_signals: list[RuleSignal] = Field(default_factory=list)


class RuleEvaluationReport(BaseModel):
    """Machine-readable batch rule report; deliberately excludes ground truth."""

    total_transactions_evaluated: int
    transactions_with_rule_triggers: int
    rule_trigger_counts: dict[str, int]
    triggered_transaction_ids_by_rule: dict[str, list[str]]
    triggered_transaction_ids: list[str]
    transaction_evidence: list[TransactionRuleEvidence]