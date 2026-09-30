"""Response models for service health."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Public health-check response."""

    status: Literal["ok"]
    service: Literal["fraud-sentinel-backend"]