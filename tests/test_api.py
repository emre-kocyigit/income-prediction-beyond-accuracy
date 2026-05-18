"""API smoke tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.models.loader import model_path


@pytest.fixture
def client():
    return TestClient(app)


def test_health_without_model(client):
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not model_path("xgboost", config=config).is_file():
        r = client.get("/health")
        assert r.status_code == 503
        return
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_sample(client):
    if not model_path("xgboost").is_file():
        pytest.skip("Run `make train` first")

    import pandas as pd
    import yaml

    root = Path(__file__).resolve().parents[1]
    with open(root / "config.yaml") as f:
        config = yaml.safe_load(f)
    row = pd.read_csv(config["data"]["processed_train"]).iloc[0]
    payload = row.drop(labels=["income"]).to_dict()
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
