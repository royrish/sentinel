"""Health-check route."""

from fastapi import APIRouter

from app.models.health import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Report that the API process is responding."""

    return HealthResponse(status="ok", service="fraud-sentinel-backend")