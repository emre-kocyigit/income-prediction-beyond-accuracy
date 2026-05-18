"""FastAPI inference service — loads trained pipeline from models/."""

from functools import lru_cache
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.schema import FEATURE_COLUMNS, load_config
from src.models.loader import DEFAULT_MODEL, load_model

app = FastAPI(
    title="Income Prediction API",
    description="Fairness-aware income classifier (audit / demo — not for production decisions).",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    """One row of processed features (same schema as train.csv without income)."""

    age: int = Field(..., ge=17, le=90)
    workclass: str
    education: str
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    hours_per_week: int = Field(..., alias="hours-per-week", ge=1, le=99)
    native_country: str = Field(..., alias="native-country")
    has_capital_gain: int = Field(..., ge=0, le=1)
    has_capital_loss: int = Field(..., ge=0, le=1)
    capital_gain_log: float
    capital_loss_log: float
    age_group: str
    is_married: int = Field(..., ge=0, le=1)
    native_region: str
    hours_category: str

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model: str


@lru_cache
def _get_estimator():
    config = load_config()
    name = config.get("deployment", {}).get("model", DEFAULT_MODEL)
    return name, load_model(name, config=config)


@app.get("/health")
def health():
    try:
        name, _ = _get_estimator()
        return {"status": "ok", "model": name}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    name, estimator = _get_estimator()
    row = body.model_dump(by_alias=True)
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred_raw = estimator.predict(X)[0]
    proba = float(estimator.predict_proba(X)[0, 1])
    try:
        label = int(pred_raw)
    except (TypeError, ValueError):
        label = 1 if ">50K" in str(pred_raw) else 0
    return PredictResponse(prediction=label, probability=proba, model=name)
