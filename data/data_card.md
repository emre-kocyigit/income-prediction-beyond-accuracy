# Data Card — Adult Income Dataset

## Dataset Overview
The Adult Income dataset is a classic binary classification benchmark derived from the 1994 US Census database. The task is to predict whether an individual's annual income exceeds $50,000. Beyond prediction accuracy, this dataset is widely used to study bias, fairness, and discrimination in machine learning models — which is the primary focus of this project.

## Source and Collection
- **Original source:** UCI Machine Learning Repository
- **Extracted by:** Barry Becker from the 1994 US Census database
- **Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Collection method:** Census survey responses — self-reported and
  administratively recorded

## Dataset Characteristics
| Property | Value |
|---|---|
| Instances | 48,842 (train: 32,561 / test: 16,281) |
| Features | 14 |
| Target | Binary (<=50K / >50K) |
| Missing values | Yes — encoded as `?` |
| Class balance | ~75% <=50K, ~25% >50K |
| Feature types | Mix of continuous and categorical |

## Features
| Feature | Type | Notes |
|---|---|---|
| age | continuous | Ranges 17–90 |
| workclass | categorical | 8 categories, includes `?` |
| fnlwgt | continuous | Census sampling weight — not used in modeling |
| education | categorical | 16 levels |
| education-num | continuous | Numerical encoding of education — redundant with education |
| marital-status | categorical | 7 categories |
| occupation | categorical | 14 categories, includes `?` — potential proxy for race and gender |
| relationship | categorical | 6 categories — correlated with marital-status |
| race | categorical | 5 categories — heavily skewed toward White (85%+) |
| sex | categorical | Binary — Male/Female only, does not reflect gender diversity |
| capital-gain | continuous | Heavily zero-inflated, highly skewed |
| capital-loss | continuous | Heavily zero-inflated, highly skewed |
| hours-per-week | continuous | Self-reported working hours |
| native-country | categorical | 41 categories — 90%+ United States |

## Target Variable
Binary classification: `<=50K` or `>50K` annual income.

**Important limitation:** The $50,000 threshold is from 1994 and has
not been adjusted for inflation. In today's terms this is approximately
$105,000 — a detail that significantly affects how the label should be
interpreted in any real-world context.

## Known Issues and Limitations
- **Temporal:** Data is 30 years old. Labor market structure, wage
  distribution, and demographic composition have changed significantly
  since 1994.
- **Missing values:** Encoded as `?` in `workclass`, `occupation`, and
  `native-country` — requires careful handling, not standard null treatment.
- **Class imbalance:** ~75/25 split — accuracy alone is a misleading metric.
- **Underrepresentation:** Several racial groups have very few instances,
  making subgroup analysis statistically limited for those groups.
- **Binary gender:** `sex` field is binary and does not capture gender
  diversity.
- **Redundant features:** `education` and `education-num` encode the same
  information. `relationship` and `marital-status` are highly correlated.
- **fnlwgt:** Census sampling weight — included in the raw data but not
  meaningful as a predictive feature.

## Ethical Considerations
The dataset contains the following protected attributes: `sex`, `race`,
`age`, `marital-status`, and `native-country`. In this project, these
attributes are used exclusively for fairness auditing — to measure whether
model predictions are systematically biased against any group. They are
not used as predictive features in modeling.

This kind of audit is precisely what tools like the
[Azure Responsible AI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard)
are designed to support in production environments.

## Intended Use
- Fairness and bias research
- Machine learning education
- Model auditing and explainability studies

## Out-of-Scope Use
This dataset must not be used to make real financial, employment, or
lending decisions. It is a research benchmark with significant known
limitations. Any model trained on this data reflects the social and
economic conditions of 1994 United States and should not be generalized
beyond that context.