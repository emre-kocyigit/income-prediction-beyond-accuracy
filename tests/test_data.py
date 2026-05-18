"""Contract tests — processed data must match training feature schema."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, assert_schema, load_config


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "config.yaml")


def test_processed_train_exists(config):
    path = Path(config["data"]["processed_train"])
    assert path.is_file(), f"Missing {path}; run notebook 03_feature_engineering.ipynb"


def test_feature_schema_train(config):
    train = pd.read_csv(config["data"]["processed_train"])
    X = train.drop(columns=[TARGET_COLUMN])
    assert_schema(X)
    assert len(FEATURE_COLUMNS) == 18


def test_no_target_leakage_in_features(config):
    train = pd.read_csv(config["data"]["processed_train"])
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN in train.columns
