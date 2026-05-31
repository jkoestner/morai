---
name: model-review
description: Review a fitted mortality model (GLM, GAM, CatBoost, or Neural) focused on actual-vs-expected (A/E) analysis. Use when evaluating model fit, segment-level A/E, rate comparisons vs VBT, partial dependence, and overfitting checks. Model-family-agnostic; branches into family-specific diagnostics in the addendum section.
allowed-tools: Read, Bash, Write, Edit, Glob, Grep
---

# Mortality Model Review (A/E Focused)

Generalized review framework for any fitted mortality model exposing a
scikit-learn-style `.predict(X)` interface. Works for GLM, GAM, CatBoost, and
Neural models in this codebase.

**This skill reviews pre-trained models. It does not train them.**

For terminology, tolerance bands, expected A/E patterns, and standard review
dimensions, read `../_shared/actuarial-reference.md` before beginning.

## Inputs

- A fitted model (typically loaded via `joblib.load("files/models/<name>.joblib")`)
- Train and test datasets with at minimum: `death_claim_amount`, `amount_exposed`,
  `qx_raw`, `qx_vbt15`, and the standard review dimensions
- The `mapping` and preprocessing artifacts used to encode features for the model

## Workflow

### 1. Sanity Checks
- Confirm model type (`type(model).__name__`) and identify the family (GLM, GAM,
  CatBoost, Neural)
- Confirm predictions run end-to-end on a small slice
- Compute `expected_claims = model.predict(X) * amount_exposed`

### 2. Overall A/E
- Train A/E and Test A/E
- Flag overfitting if `|Train A/E - Test A/E| > 0.05` OR if Train A/E is in band
  but Test A/E is out of band

### 3. A/E by Dimension
Group by each dimension in the shared reference's "Standard Review Dimensions"
list. Report on both train and test:

```
| Dimension Value | Actual | Expected | Exposure | A/E |
```

Apply tolerance bands from the shared reference (sized by exposure level).
Flag any segment outside band.

### 4. Rate Comparison Charts
Use `morai.experience.charters.compare_rates` to plot weighted-average
(`amount_exposed`-weighted) rates for `qx_raw`, `qx_vbt15`, and `qx_model`
across age, duration, and other key dimensions.

### 5. Partial Dependence (model-agnostic)
Use `morai.experience.charters.pdp` — it calls `model.predict` so it works for
all families.

Priority features:
1. `duration` — line color by `insurance_plan`
2. `attained_age` — line color by `class_enh` or `binned_face`
3. `insurance_plan` — overall effect

Settings: `weight="amount_exposed"`, `secondary="death_count"`, `center="per_x"`.

### 6. Family-Specific Addendum

Run only the section matching the model's family.

#### GLM (`morai.models.core.GLM`)
- Coefficient table (estimate, SE, p-value) — flag |p| > 0.05 on retained terms
- Likelihood ratio vs a reduced model via `calc_likelihood_ratio`
- Sparse-category check: coefficients with very large SE

#### GAM (`GAMPy` / `GAMStats`)
- Smooth term plots — call out non-monotonic tails
- EDF per term
- Compare in-sample deviance to a comparable GLM if available

#### CatBoost
- Built-in importance: `PredictionValuesChange` and `LossFunctionChange`
- SHAP via TreeExplainer (fast, exact) — bar + summary
- Categorical handling: confirm `cat_features` were declared at fit time

#### Neural (`morai.models.neural.Neural`)
- Defer to the `nn-review` skill for SHAP (KernelExplainer), embedding cosine /
  PCA, and loss-curve inspection

## Output Template

```markdown
## Model Review — <model name> (<family>)

### Overall
- Train A/E: X.XX
- Test A/E: X.XX
- Overfitting risk: Low / Medium / High

### A/E by Dimension
<tables, with out-of-band rows flagged>

### Rate Comparison Highlights
<bullets on where qx_model diverges from qx_raw and qx_vbt15>

### Family-Specific Findings
<from the addendum section run>

### Key Findings
1. ...
2. ...

### Recommendations
1. ...
2. ...
```

## Common Code Patterns

### A/E Calculation
```python
ae = df.groupby(dim).agg(
    actual=("death_claim_amount", "sum"),
    expected=("expected_claims", "sum"),
    exposure=("amount_exposed", "sum"),
)
ae["A/E"] = ae["actual"] / ae["expected"]
```

### Model-Agnostic PDP
```python
from morai.experience import charters
charters.pdp(
    model=model,
    df=md_encoded,
    x_axis="duration",
    line_color="insurance_plan",
    weight="amount_exposed",
    secondary="death_count",
    mapping=mapping,
)
```

### Rate Comparison
```python
charters.compare_rates(
    df=md,
    rates=["qx_raw", "qx_vbt15", "qx_model"],
    x_axis="attained_age",
    weight="amount_exposed",
)
```
