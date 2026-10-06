import pytest
from fastapi.testclient import TestClient

from galaxies.api import create_app
from galaxies.config import Settings
from galaxies.demo import demo_comparison


@pytest.fixture()
def client(trained):
    with TestClient(create_app(trained[0])) as connection:
        yield connection


def test_http_health_schema_and_comparison(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "/api/compare" in client.get("/openapi.json").json()["paths"]
    response = client.post("/api/compare", json=demo_comparison().model_dump(mode="json"))
    assert response.status_code == 200, response.text
    assert response.json()["decision"] == "demonstration_only"
    trace = response.json()["ranking"][0]["trace_id"]
    assert client.get(f"/api/traces/{trace}").status_code == 200


def test_invalid_features_and_mismatched_horizon_return_422(client):
    data = demo_comparison().model_dump(mode="json")
    data["creatives"][0]["features"]["views"] = 1000
    assert client.post("/api/compare", json=data).status_code == 422
    data = demo_comparison().model_dump(mode="json")
    data["context"]["horizon_days"] = 30
    assert client.post("/api/compare", json=data).status_code == 422


def test_cross_origin_mutation_rejected(client):
    response = client.post(
        "/api/compare",
        json=demo_comparison().model_dump(mode="json"),
        headers={"origin": "https://attacker.example"},
    )
    assert response.status_code == 403


def test_authentication_and_nonlocal_bind(trained):
    settings, _ = trained
    with pytest.raises(ValueError, match="loopback"):
        create_app(Settings(workspace=settings.workspace, host="0.0.0.0"))
    protected = settings.model_copy(update={"api_token": "test-secret"})
    with TestClient(create_app(protected)) as client:
        assert client.get("/api/models").status_code == 401
        assert (
            client.get("/api/models", headers={"Authorization": "Bearer test-secret"}).status_code
            == 200
        )


def test_invalid_upload_and_path_escape(client, trained):
    assert (
        client.post(
            "/api/media", files={"file": ("script.html", b"<script>bad</script>", "text/html")}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/datasets", files={"file": ("bad.jsonl", b"{}", "text/plain")}).status_code
        == 422
    )
    with pytest.raises(ValueError, match="fora"):
        trained[0].contained("../state.sqlite3")


def test_real_prospective_registration_rejects_synthetic_model(client):
    from datetime import timedelta

    from galaxies.io import utcnow

    request = demo_comparison()
    request.context.as_of = utcnow()
    request.context.planned_datetime = utcnow() + timedelta(days=2)
    response = client.post(
        "/api/prospective",
        json={
            "creative": request.creatives[0].model_dump(mode="json"),
            "context": request.context.model_dump(mode="json"),
        },
    )
    assert response.status_code == 422
    assert "sintético" in response.json()["detail"]
