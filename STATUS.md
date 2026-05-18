# Project completion tracker

Each step: **current status → changes → test → result**.  
Principles: one config source (`config.yaml`), reproducible training script, contract tests on schema, no claims without artifacts.

---

## Step 0 — Scope & README

| Field | Detail |
|--------|--------|
| **Current status** | README promised PyTorch / neural net; not implemented. |
| **Changes** | Removed neural network from README and stack; added this tracker. |
| **Test** | Manual: README matches repo capabilities. |
| **Result** | Scope aligned with LR, RF, XGBoost, LightGBM only. |

---

## Step 1 — Train & save models (artifacts)

| Field | Detail |
|--------|--------|
| **Current status** | `models/` empty; notebooks 05/06/API need `.pkl` files. |
| **Changes** | `make train` → `scripts/train_and_save_models.py`; `config.yaml` paths fixed. |
| **Test** | `make train` then `pytest tests/test_models.py -v` |
| **Result** | `models/logistic_regression.pkl`, `random_forest.pkl`, `xgboost.pkl` written. LightGBM skipped if package not installed. |

---

## Step 2 — `src/` package (load data + model)

| Field | Detail |
|--------|--------|
| **Current status** | Done. |
| **Changes** | `src/data/schema.py`, `src/models/loader.py` — config-driven paths & feature contract. |
| **Test** | `pytest tests/test_data.py tests/test_models.py -v` |
| **Result** | 3 schema tests + 3 model smoke tests **passed**. |

---

## Step 3 — FastAPI inference

| Field | Detail |
|--------|--------|
| **Current status** | Done. |
| **Changes** | `app/main.py` — `/health`, `/predict`; deployment model from `config.yaml` (`deployment.model: xgboost`). |
| **Test** | `pytest tests/test_api.py -v` |
| **Result** | 2 tests **passed** (`httpx==0.27.2` pinned for Starlette TestClient). |

---

## Step 4 — Evaluation notebook gaps (05)

| Field | Detail |
|--------|--------|
| **Current status** | Intro sections 1–7 partly plan-only; loads non-existent `neural_network`. |
| **Changes** | Model list trimmed; optional sections documented or added incrementally. |
| **Test** | Run `05_evaluation.ipynb` after Step 1. |
| **Result** | *Pending* |

---

## Step 5 — Explainability (07)

| Field | Detail |
|--------|--------|
| **Current status** | Not started. |
| **Changes** | `07_explainability.ipynb` — SHAP summary + error-by-subgroup (XGBoost). |
| **Test** | Notebook runs after models exist. |
| **Result** | *Pending* |

---

## Step 6 — Findings & README key results

| Field | Detail |
|--------|--------|
| **Current status** | `reports/findings.md` empty; README key findings placeholder. |
| **Changes** | Synthesize metrics/fairness into `findings.md` + README bullet summary. |
| **Test** | Review against generated `reports/*.csv`. |
| **Result** | *Pending* |

---

## Quick commands

```bash
make install
make train      # writes models/*.pkl (gitignored)
make test       # schema + model + API smoke tests
make serve      # http://127.0.0.1:8000/docs
```
