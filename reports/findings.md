# Findings (draft)

Synthesis pending full runs of `05_evaluation.ipynb` and `06_fairness_analysis.ipynb`.

## Engineering baseline (verified)

| Check | Result |
|--------|--------|
| Feature schema contract (`src/data/schema.py`) | 18 features, no target leakage |
| `make train` | Saves LR, RF, XGBoost pipelines to `models/` |
| `make test` | 8 tests — schema + model smoke + API |

## Next

1. Run notebooks 05–06 and paste metric/fairness highlights here.
2. Add `07_explainability.ipynb` (SHAP summary).
3. Update README **Key findings** from this file.
