"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.alerts import router as alerts_router
from app.api.health import router as health_router
from app.config import get_settings
from app.models.fraud_alert import FraudAlertDataset
from app.services.alert_store import InMemoryAlertStore
from app.services.explanation_service import ExplanationService
from app.services.risk_engine import generate_fraud_alert_dataset

settings = get_settings()
BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_TRANSACTION_CSV = BACKEND_DIRECTORY / "data" / "synthetic_transactions.csv"
DEFAULT_ALERT_ARTIFACT = BACKEND_DIRECTORY / "data" / "fraud_alerts.json"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Load the generated evidence artifact, creating it if this is a fresh checkout."""

    if not DEFAULT_ALERT_ARTIFACT.exists():
        generate_fraud_alert_dataset(DEFAULT_TRANSACTION_CSV, DEFAULT_ALERT_ARTIFACT)
    dataset = FraudAlertDataset.model_validate_json(
        DEFAULT_ALERT_ARTIFACT.read_text(encoding="utf-8")
    )
    application.state.alert_store = InMemoryAlertStore(dataset)
    application.state.explanation_service = ExplanationService(settings=settings)
    yield


app = FastAPI(title="Fraud Sentinel API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")