"""Schema for generated UPI transaction records."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


TransactionType = Literal["transfer", "collect_request", "merchant_payment", "bill_payment"]
FraudPattern = Literal[
    "large_collect_first_time_payee",
    "rapid_mule_chain",
    "circular_money_flow",
]


class Transaction(BaseModel):
    """Ground-truth transaction data; this model contains no detection output."""

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
    label: Literal["fraud", "safe"]
    fraud_pattern: FraudPattern | None = None
