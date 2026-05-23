---
name: nn-review
description: Review and analyze neural network mortality models. Use when evaluating model performance, generating A/E analysis, PDP plots, SHAP analysis, or embedding analysis for mortality prediction models.
allowed-tools: Read, Bash, Write, Edit, Glob, Grep
---

# Neural Network Mortality Model Review

This skill provides a comprehensive review framework for neural network mortality models, following actuarial best practices.

**Important:** This skill is for reviewing pre-trained models, not training new ones.

## Pre-trained Models

Models are stored in `files/models/` and should be loaded using joblib:

```python
import joblib

# Load the pre-trained neural network model
model = joblib.load("files/models/neural.joblib")

# Generate predictions (requires preprocessed X data)
predictions = model.predict(X)
```

## When to Use

- When reviewing an existing pre-trained model
- When evaluating model performance
- When comparing model predictions to actual experience
- When analyzing feature importance and relationships

## Analysis Framework

### 1. A/E Ratio Analysis (Actual vs Expected)

Generate A/E ratios by key dimensions for both train and test sets:

**Required dimensions:**
- `sex` (M/F)
- `smoker_status` (NS/S)
- `attained_age` (binned: 30-40, 41-50, 51-60, 61-70, 71-80, 81+)
- `duration` (binned: 1-5, 6-10, 11-15, 16-20, 21+)
- `observation_year`
- `insurance_plan` (Term, Perm, UL, ULSG, VL, VLSG)
- `face_amount_band` or `binned_face`
- `class_enh` (preferred class combination)

**Output format:**
```
| Dimension | Actual | Expected | Exposure | A/E |
|-----------|--------|----------|----------|-----|
```

**Key metrics:**
- Overall A/E should be close to 1.00
- Train vs Test A/E should be similar (check for overfitting)
- Flag any dimension with A/E < 0.80 or A/E > 1.20

### 2. Rate Comparison Charts

Compare predicted rates against:
- `qx_raw` (actual mortality rate)
- `qx_vbt15` (VBT 2015 benchmark)
- `qx_model` (model predictions)

Use weighted averages with `amount_exposed` as weights.

### 3. Partial Dependence Plots (PDP)

Generate PDP plots for key features:

**Priority features:**
1. `duration` - with line color by `insurance_plan`
2. `attained_age` - with line color by `class_enh` or `binned_face`
3. `insurance_plan` - overall effect

**PDP settings:**
- Use `weight="amount_exposed"` for weighted PDPs
- Include `secondary="death_count"` for context
- Apply `center="per_x"` for relative comparisons

### 4. SHAP Analysis

Perform SHAP analysis to understand feature importance:

**Required plots:**
1. **Bar plot** - overall feature importance
2. **Summary plot** - feature value impact distribution
3. **Waterfall plot** - single prediction explanation
4. **Dependence plot** - feature interaction effects

**Settings:**
- Use 100 background samples for KernelExplainer
- Use 100 explainer samples for computation
- Set seed=42 for reproducibility

### 5. Embedding Analysis

For embedded categorical features (e.g., `face_amount_band`, `insurance_plan`):

**Required analysis:**
1. **Cosine similarity heatmap** - shows which categories are similar
2. **PCA 2D plot** - visualize category relationships
3. **Embedding weights table** - raw dimension values

## Model Evaluation Criteria

### Good Model Indicators
- Overall A/E between 0.95 - 1.05
- Train/Test A/E difference < 0.05
- Consistent A/E across all dimensions (0.85 - 1.15)
- SHAP importance aligns with actuarial intuition
- Embeddings show sensible category groupings

### Warning Signs
- A/E < 0.80 or > 1.20 for any major segment
- Train A/E much better than Test A/E (overfitting)
- Early durations (1-5) with poor A/E (select mortality issue)
- Young ages (41-50) with poor A/E
- High SHAP importance for unexpected features

### Improvement Suggestions

If early duration A/E is poor:
- Add explicit duration indicators (dur_1, dur_2, etc.)
- Add select period flag
- Consider separate select/ultimate models

If age A/E shows patterns:
- Check spline configuration (n_knots, degree)
- Consider age x smoker interactions

If categorical features show issues:
- Consider embeddings instead of OHE
- Check category groupings (binned_face, class_enh)

## Output Template

```markdown
## Neural Network Model Review

### Overall Performance
- Train A/E: X.XX
- Test A/E: X.XX
- Overfitting Risk: Low/Medium/High

### A/E by Dimension
[Tables for each dimension]

### Key Findings
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Recommendations
1. [Recommendation 1]
2. [Recommendation 2]

### Charts Generated
- [ ] A/E ratio charts
- [ ] Rate comparison
- [ ] PDP plots
- [ ] SHAP analysis
- [ ] Embedding analysis
```

## Code Patterns

### A/E Calculation
```python
ae_table = df.groupby(dimension).agg({
    'death_claim_amount': 'sum',
    'expected_claims': 'sum',
    'amount_exposed': 'sum'
})
ae_table['A/E'] = ae_table['death_claim_amount'] / ae_table['expected_claims']
```

### PDP with morai
```python
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

### SHAP Setup
```python
Shap = neural.Shap(model=model, background_df=X_train, n_samples=100, seed=42)
shap_values = Shap.compute_values(explain_df=X_test, n_samples=100, seed=42)
shap.plots.bar(Shap.shap_values)
shap.summary_plot(Shap.shap_values, Shap.sample_explain_df)
```
