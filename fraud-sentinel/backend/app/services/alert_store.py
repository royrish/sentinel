"""JSON-backed startup data with runtime-only in-memory alert feedback."""

from __future__ import annotations

from threading import RLock

from app.models.fraud_alert import (
    AlertGraphResponse,
    AlertSummaryResponse,
    AlertStatus,
    FeedbackStatus,
    FraudAlert,
    FraudAlertDataset,
    GraphEdge,
    GraphNode,
    RiskLevel,
    TransactionRiskEvidence,
)


class InMemoryAlertStore:
    """Small replaceable store; feedback updates reset when the API reloads its artifact."""

    def __init__(self, dataset: FraudAlertDataset) -> None:
        self._lock = RLock()
        self._risk_by_transaction_id = {
            item.transaction_id: item for item in dataset.transaction_evidence
        }
        self._alert_by_id = {alert.alert_id: alert for alert in dataset.alerts}

    def list_alerts(
        self,
        *,
        risk_level: RiskLevel | None = None,
        status: AlertStatus | None = None,
        limit: int = 50,
    ) -> list[FraudAlert]:
        with self._lock:
            items = [
                alert
                for alert in self._alert_by_id.values()
                if (risk_level is None or alert.risk_level == risk_level)
                and (status is None or alert.status == status)
            ]
            return sorted(items, key=lambda item: (-item.risk_score, item.timestamp, item.alert_id))[:limit]

    def get_alert(self, alert_id: str) -> FraudAlert | None:
        with self._lock:
            return self._alert_by_id.get(alert_id)

    def get_transaction_evidence(self, transaction_id: str) -> TransactionRiskEvidence | None:
        return self._risk_by_transaction_id.get(transaction_id)

    def set_feedback(self, alert_id: str, status: FeedbackStatus) -> FraudAlert | None:
        with self._lock:
            alert = self._alert_by_id.get(alert_id)
            if alert is None:
                return None
            updated = alert.model_copy(update={"status": status})
            self._alert_by_id[alert_id] = updated
            return updated

    def summary(self) -> AlertSummaryResponse:
        with self._lock:
            alerts = list(self._alert_by_id.values())
            return AlertSummaryResponse(
                total_alerts=len(alerts),
                high_risk=sum(alert.risk_level == "high" for alert in alerts),
                medium_risk=sum(alert.risk_level == "medium" for alert in alerts),
                low_risk=sum(alert.risk_level == "low" for alert in alerts),
                open_alerts=sum(alert.status == "open" for alert in alerts),
                confirmed_fraud=sum(alert.status == "confirmed_fraud" for alert in alerts),
                marked_safe=sum(alert.status == "marked_safe" for alert in alerts),
            )

    def graph(self, alert_id: str) -> AlertGraphResponse | None:
        alert = self.get_alert(alert_id)
        if alert is None:
            return None
        related_ids = {alert.transaction_id}
        for finding in alert.network_evidence:
            related_ids.update(finding.transaction_ids)
        records = [
            self._risk_by_transaction_id[transaction_id]
            for transaction_id in related_ids
            if transaction_id in self._risk_by_transaction_id
        ]
        records.sort(key=lambda item: (item.timestamp, item.transaction_id))
        nodes = sorted(
            {
                account
                for item in records
                for account in (item.sender_account, item.receiver_account)
            }
        )
        return AlertGraphResponse(
            alert_id=alert_id,
            nodes=[GraphNode(id=account) for account in nodes],
            edges=[
                GraphEdge(
                    source=item.sender_account,
                    target=item.receiver_account,
                    transaction_id=item.transaction_id,
                    amount=item.amount,
                    timestamp=item.timestamp,
                )
                for item in records
            ],
        )