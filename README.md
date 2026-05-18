# Income Prediction — What the Model Gets Wrong and Why

Binary income classification is a solved problem on paper.
The harder question is whether the model makes errors uniformly or systematically disadvantages certain groups. This project treats that question as seriously as prediction accuracy.

**Completion tracker:** see [STATUS.md](STATUS.md) (status → changes → test → result for each step).

## What this project covers

- **Data audit** before any modeling — missing patterns, leakage risks, and class imbalance
- **Exploratory analysis** with a focus on subgroup distributions, not just overall statistics
- **Modeling** — Logistic Regression, Random Forest, XGBoost, LightGBM (with optional Fairlearn mitigation in notebook 04)
- **Deep evaluation** — calibration, bootstrap CIs, significance tests, subgroup metrics
- **Fairness analysis** across protected attributes: sex, race, age, marital status, native country
- **Explainability** — SHAP and error analysis by subgroup *(in progress; see STATUS.md)*
- **FastAPI endpoint** for demo inference with input validation *(see `app/main.py`)*

## Key findings

*To be updated after notebooks 05–06 are run and `reports/findings.md` is filled.*

## Stack

Python 3.10+ · pandas · numpy · scikit-learn · xgboost · lightgbm · fairlearn · shap · fastapi · pytest

## Repository structure

```
├── data/
│   ├── raw/                           ✓ adult.data, adult.test, adult.names
│   ├── processed/                     ✓ train.csv, test.csv (generated; gitignored)
│   └── data_card.md                   ✓ dataset documentation
│
├── notebooks/
│   ├── 01_data_audit.ipynb            ✓
│   ├── 02_eda.ipynb                   ✓
│   ├── 03_feature_engineering.ipynb   ✓
│   ├── 04_modeling.ipynb              ✓
│   ├── 05_evaluation.ipynb            ✓ (run after `make train`)
│   └── 06_fairness_analysis.ipynb     ✓ (run after `make train`)
│
├── scripts/
│   └── train_and_save_models.py       ✓ reproducible training → models/*.pkl
│
├── src/
│   ├── data/schema.py                 ✓ feature contract + loaders
│   └── models/loader.py               ✓ joblib load helpers
│
├── app/
│   └── main.py                        ✓ FastAPI /health, /predict
│
├── tests/                             ✓ schema, model, API smoke tests
│
├── reports/
│   ├── findings.md                    ⏳ synthesis (pending)
│   └── figures/                       ✓ EDA + notebook outputs
│
├── STATUS.md                          ✓ step-by-step completion log
├── config.yaml                        ✓ paths, protected attributes, deployment model
├── Makefile                           ✓ install, train, test, serve
└── requirements.txt
```

## How to run

```bash
make install
make train          # writes models/*.pkl (gitignored — required for 05, 06, API)
make test           # schema + optional model/API tests
make serve          # http://127.0.0.1:8000/docs
```

**Notebooks:** open in order starting with `01_data_audit.ipynb`. Run `03` before `make train` if processed CSVs are missing.

## Dataset

Adult Income Dataset (UCI, 1994 Census extract). See `data/data_card.md` for limitations and ethical considerations.

## Limitations and caveats

Fairness audit and learning exercise — not for real financial or employment decisions. Data is from 1994; the $50K threshold is not inflation-adjusted.

The goal is not the most accurate classifier. It is to understand **where and why** a model fails — and **for whom**.
