"""Tests for the API health endpoint."""

import unittest

from fastapi.testclient import TestClient

from app.main import app


class HealthEndpointTests(unittest.TestCase):
    def test_health_returns_expected_response(self) -> None:
        with TestClient(app) as client:
            response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "fraud-sentinel-backend"},
        )


if __name__ == "__main__":
    unittest.main()