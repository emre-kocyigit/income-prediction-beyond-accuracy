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

```
├── data/
│   ├── raw/                           ✓ adult.data, adult.test, adult.names
│   ├── processed/                     ✓ train.csv, test.csv
│   └── data_card.md                   ✓ dataset documentation and known limitations
│
├── notebooks/
│   ├── 01_data_audit.ipynb            ✓ completed
│   ├── 02_eda.ipynb                   ✓ completed
│   ├── 03_feature_engineering.ipynb   ✓ completed
│   ├── 04_modeling.ipynb              ✓ completed
│   ├── 05_evaluation.ipynb            ✓ completed
│   └── 06_fairness_analysis.ipynb     ✓ completed
│
├── src/
│   ├── data/                          ⏳ preprocessing modules (placeholder)
│   ├── models/                        ⏳ baseline and tree models (placeholder)
│   └── evaluation/                    ⏳ metrics and fairness modules (placeholder)
│
├── app/
│   └── main.py                        ⏳ FastAPI inference endpoint (placeholder)
│
├── tests/
│   └── test_data.py                   ⏳ unit tests (placeholder)
│
├── reports/
│   ├── findings.md                    ✓ written conclusions
│   └── SECRET_SCAN_REPORT.md          ✓ security audit results
│
├── config.yaml                        ✓ project configuration
├── Makefile                           ✓ automation commands
├── requirements.txt                   ✓ dependency specifications
├── SECURITY_AUDIT_REPORT.md           ✓ comprehensive security analysis
└── LICENSE
```

**Legend:**  
✓ = Complete with content  
⏳ = Structure exists, implementation pending

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