"""FastAPI アプリの起動骨格のテスト。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from rough2ink.app import app
from rough2ink.core.params import AnalysisParams

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_params_defaults_returns_analysis_params_json() -> None:
    response = client.get("/api/params/defaults")
    assert response.status_code == 200

    body = response.json()
    expected = AnalysisParams().model_dump()
    assert body == expected


def test_index_html_is_served_at_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "rough2ink" in response.text
