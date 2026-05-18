
## Explainability Findings (Step 5)

### 5.2 Global SHAP
- Globally, the model relies most on: is_married (0.939), age (0.732),
  capital_gain_log (0.452), education (0.423), occupation (0.416).
- is_married dominates by a wide margin — it is the single strongest predictor,
  reflecting that married-civ-spouse status strongly correlates with dual-income
  households and career seniority in the 1994 census.
- capital_gain_log and capital_loss_log show high magnitude but sparse signal
  (most values = 0); when non-zero they exert outsized influence on prediction.
- hours-per-week (0.369), marital-status (0.241), and relationship (0.203)
  are consistent mid-tier drivers across the full sample.

### 5.3 Subgroup Errors
- False negatives concentrate in structurally disadvantaged groups:
  Amer-Indian-Eskimo (FNR=0.421), age ≤25 (FNR=0.564), age 65+ (FNR=0.387),
  Never-married (FNR=0.450), Divorced (FNR=0.385), Widowed (FNR=0.465).
- False positives concentrate in groups with high predicted selection rates:
  Married-civ-spouse (FPR=0.465), age 46-55 (FPR=0.327), age 36-45 (FPR=0.302).
- Largest FNR gap by sex: Female FNR=0.239 vs Male FNR=0.121 (Δ=0.118) —
  the model misses nearly twice as many true high earners among women.
- Largest selection rate gap by sex: Male=0.436 vs Female=0.132 (Δ=0.304) —
  men are predicted >50K at 3.3× the rate of women.
- Largest FNR gap by race: Amer-Indian-Eskimo FNR=0.421 vs White FNR=0.131
  (Δ=0.290) — though sample size is small (n=159), the gap is substantial.
- Native-country analysis is noisy for small groups (n<30); El-Salvador and
  Nicaragua show FNR=1.0 but with very few positive examples — treat with caution.

### 5.4 Local (Case Studies)
- FP case: Male, age 36-45, Asian-Pac-Islander, Married-civ-spouse, Exec-managerial,
  90 hours/week (extreme), HS-grad — model assigns P(>50K)=0.995 driven by
  is_married=1, extreme hours, and exec occupation. True label is ≤50K, likely
  because capital_gain_log=0 and education (HS-grad only) pull against >50K,
  but are outweighed by the marriage + hours + occupation combination.
- FN case: Male, age 22 (≤25), Never-married, Some-college, part-time (15h),
  Unknown occupation — model assigns P(>50K)=0.001. True label is >50K.
  The model cannot overcome the combination of youth, part-time hours, no marriage
  signal, and unknown occupation — all strong ≤50K indicators in the training data.

### 5.5 Counterfactuals
- FN — no single-feature flip was achievable: raising hours-per-week to 50
  moved prob 0.001 → 0.011; adding hours_category=full-time moved it to 0.013.
  The model's certainty is anchored in the intersection of age ≤25 + Never-married
  + is_married=0 — hours alone cannot overcome this.
- FP — highly robust to single and combined edits: reducing hours to 35,
  changing hours_category to part-time, and swapping occupation to Other-service
  individually and together left pred=1 (prob range 0.984–0.995). The FP is
  locked in by is_married=1 + age 36-45 — the model's two dominant features.
- Key insight: when is_married dominates (SHAP=0.939 globally), counterfactual
  flips require changing marital status — a feature that cannot ethically be
  used as a lever in any real decision context.

### 5.6 Fairness Cross-link
- SHAP importance differs meaningfully by sex: is_married and age dominate for
  both groups, but hours-per-week and occupation rank higher for Female (3rd, 4th)
  than for Male (where capital_gain_log ranks 3rd at 0.516). This suggests the
  model uses labour-market signals more heavily for women and investment income
  signals more heavily for men — reflecting 1994 structural patterns.
- The female FNR of 0.239 vs male FNR of 0.121 is partly explained by is_married
  having lower predictive lift for women (married women in 1994 were less likely
  to be primary earners), meaning the model's top feature systematically
  underserves female high earners.
- SHAP ≠ causal; all findings reflect 1994 census patterns and must not be
  used to inform real employment or financial decisions.

### Monitoring Recommendations
- Track FPR and FNR by sex and race on every new inference batch; alert if
  female FNR exceeds 0.25 or Amer-Indian-Eskimo FNR exceeds 0.45.
- Monitor distribution drift on top-3 SHAP features: is_married, age,
  capital_gain_log — shift in these is the earliest signal of model degradation.
- Re-evaluate fairness metrics if the model is ever retrained or deployed live.
- Consider whether is_married (a proxy for sex-correlated wealth patterns)
  should be included as a feature in any production version of this model.
