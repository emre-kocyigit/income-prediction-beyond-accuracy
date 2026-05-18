"""Load persisted sklearn pipelines for inference."""

from pathlib import Path

import joblib

from src.data.schema import load_config

DEFAULT_MODEL = "xgboost"


def model_path(name: str, config: dict | None = None, config_path: str = "config.yaml") -> Path:
    if config is None:
        config = load_config(config_path)
    base = config["paths"]["models"].rstrip("/")
    return Path(base) / f"{name}.pkl"


def load_model(name: str = DEFAULT_MODEL, config: dict | None = None):
    path = model_path(name, config=config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Run `make train` or notebooks/04_modeling.ipynb first."
        )
    return joblib.load(path)
