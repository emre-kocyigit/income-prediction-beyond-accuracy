"""Smoke tests for persisted models."""

from pathlib import Path

import pytest

from src.data.schema import load_config, load_processed_split
from src.models.loader import load_model


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "xgboost"])
def test_model_predicts(name, config):
    path = Path(config["paths"]["models"]) / f"{name}.pkl"
    if not path.is_file():
        pytest.skip(f"No artifact at {path}; run `make train`")

    model = load_model(name, config=config)
    _, _, X_test, _ = load_processed_split(config)
    preds = model.predict(X_test.head(20))
    proba = model.predict_proba(X_test.head(20))
    assert len(preds) == 20
    assert proba.shape == (20, 2)
