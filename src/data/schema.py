"""Feature contract for processed Adult Income data."""

from pathlib import Path

import pandas as pd
import yaml

TARGET_COLUMN = "income"

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "hours-per-week",
    "native-country",
    "has_capital_gain",
    "has_capital_loss",
    "capital_gain_log",
    "capital_loss_log",
    "age_group",
    "is_married",
    "native_region",
    "hours_category",
]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_processed_split(config: dict | None = None, config_path: str = "config.yaml"):
    """Load train/test CSVs; returns (X_train, y_train, X_test, y_test)."""
    if config is None:
        config = load_config(config_path)

    train = pd.read_csv(config["data"]["processed_train"])
    test = pd.read_csv(config["data"]["processed_test"])

    assert_schema(train.drop(columns=[TARGET_COLUMN], errors="ignore"))
    assert_schema(test.drop(columns=[TARGET_COLUMN], errors="ignore"))

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def assert_schema(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(FEATURE_COLUMNS)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected columns: {sorted(extra)}")
