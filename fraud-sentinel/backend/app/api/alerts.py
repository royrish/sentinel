"""Typed API endpoints for unified alerts and associated transaction evidence."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.models.fraud_alert import (
    AlertGraphResponse,
    AlertStatus,
    AlertsResponse,
    AlertSummaryResponse,
    FeedbackRequest,
    FeedbackResponse,
    FraudAlert,
    RiskLevel,
    TransactionRiskEvidence,
)
from app.services.alert_store import InMemoryAlertStore

router = APIRouter()


def get_alert_store(request: Request) -> InMemoryAlertStore:
    store = getattr(request.app.state, "alert_store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Alert data is not loaded.",
        )
    return store


AlertStoreDependency = Annotated[InMemoryAlertStore, Depends(get_alert_store)]


@router.get("/alerts", response_model=AlertsResponse)
def list_alerts(
    store: AlertStoreDependency,
    risk_level: RiskLevel | None = None,
    alert_status: Annotated[AlertStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
) -> AlertsResponse:
    items = store.list_alerts(risk_level=risk_level, status=alert_status, limit=limit)
    return AlertsResponse(total=len(items), items=items)


@router.get("/alerts/{alert_id}/graph", response_model=AlertGraphResponse)
def get_alert_graph(alert_id: str, store: AlertStoreDependency) -> AlertGraphResponse:
    graph = store.graph(alert_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return graph


@router.get("/alerts/{alert_id}", response_model=FraudAlert)
def get_alert(alert_id: str, store: AlertStoreDependency) -> FraudAlert:
    alert = store.get_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return alert


@router.get(
    "/transactions/{transaction_id}/evidence",
    response_model=TransactionRiskEvidence,
)
def get_transaction_evidence(
    transaction_id: str, store: AlertStoreDependency
) -> TransactionRiskEvidence:
    evidence = store.get_transaction_evidence(transaction_id)
    if evidence is None:
        raise HTTPException(status_code=404, detail="Transaction evidence not found.")
    return evidence


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    feedback: FeedbackRequest,
    store: AlertStoreDependency,
) -> FeedbackResponse:
    alert = store.set_feedback(feedback.alert_id, feedback.status)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found.")
    return FeedbackResponse(alert_id=alert.alert_id, status=feedback.status)


@router.get("/summary", response_model=AlertSummaryResponse)
def get_alert_summary(store: AlertStoreDependency) -> AlertSummaryResponse:
    return store.summary()