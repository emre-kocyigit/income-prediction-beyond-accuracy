# Income Prediction — What the Model Gets Wrong and Why

Binary income classification is a solved problem on paper.
The harder question is whether the model makes errors uniformly or systematically disadvantages certain groups. This project treats that question as seriously as prediction accuracy.

## What this project covers

- **Data audit** before any modeling — missing patterns, leakage risks, and class imbalance
- **Exploratory analysis** with a focus on subgroup distributions,
  not just overall statistics
- **Four modeling approaches** with justified selection — Logistic Regression, Random Forest, XGBoost/LightGBM, and a PyTorch neural network
- **Deep evaluation** — calibration curves, confidence intervals,
  statistical significance testing between models
- **Fairness analysis** across all protected attributes: sex, race, age, marital status, and native country
- **Explainability** — global and local SHAP values, counterfactual examples, and error analysis by subgroup
- **FastAPI endpoint** for real-time inference with responsible deployment considerations

## Key findings

*To be updated as analysis progresses.*

## Stack

Python 3.10+
pandas · numpy · scikit-learn
xgboost · lightgbm
pytorch
shap · fairlearn
fastapi · uvicorn · pydantic
pytest

## Repository structure

├── data/
│   ├── raw/                  ← original dataset, never modified
│   ├── processed/            ← cleaned and engineered features
│   └── data_card.md         ← dataset documentation and known limitations
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_fairness_analysis.ipynb
│
├── src/
│   ├── data/                 ← loading and preprocessing modules
│   ├── models/               ← baseline and tree model modules
│   └── evaluation/           ← metrics and fairness modules
│
├── app/                      ← FastAPI inference endpoint
├── tests/                    ← unit tests
├── reports/
│   ├── findings.md           ← written conclusions
│   └── figures/              ← saved plots
│
├── config.yaml
├── Makefile
└── requirements.txt

## How to run

**Install dependencies**
```bash
make install
```

**Run tests**
```bash
make test
```

**Start the API**
```bash
make serve
```

**Run notebooks**

Open in order — each notebook builds on the previous one.
Start with `01_data_audit.ipynb`.

## Dataset

Adult Income Dataset — UCI ML Repository, extracted from the
1994 US Census by Barry Becker. See `data/data_card.md` for
full documentation, known limitations, and ethical considerations.

## Limitations and caveats

This project is a fairness audit and learning exercise, not a
production system. The dataset is from 1994 and reflects social
and economic conditions that have changed significantly. The
$50K income threshold is not inflation-adjusted. Models trained
here should not be used for real financial or employment decisions.

The goal is not to build the most accurate classifier. It is to
understand where and why a model fails — and for whom.