---
name: nn-review
description: Neural-network-specific diagnostics for a fitted mortality model. Use AFTER running the model-review skill to cover the items unique to neural nets — SHAP via KernelExplainer, categorical embedding analysis (cosine similarity, PCA), and training/validation loss inspection.
allowed-tools: Read, Bash, Write, Edit, Glob, Grep
---

# Neural Network Mortality Model — NN-Only Diagnostics

This skill covers the items unique to neural network mortality models. For the
generic A/E review (overall + segmented A/E, rate comparisons, PDP, tolerance
bands), run the `model-review` skill first.

For terminology and feature conventions, read
`../_shared/actuarial-reference.md`.

## When to Use

- After `model-review` on a `morai.models.neural.Neural` model
- When you need SHAP explanations, embedding analysis, or overfitting checks
  that draw on the NN training history

## Loading

```python
import joblib
model = joblib.load("files/models/neural.joblib")
```

## 1. SHAP (KernelExplainer)

Neural models don't have a fast TreeExplainer path — use KernelExplainer with
sampling.

```python
from morai.models import neural
import shap

Shap = neural.Shap(model=model, background_df=X_train, n_samples=100, seed=42)
shap_values = Shap.compute_values(explain_df=X_test, n_samples=100, seed=42)

shap.plots.bar(Shap.shap_values)
shap.summary_plot(Shap.shap_values, Shap.sample_explain_df)
# Plus: waterfall (single prediction), dependence (feature interaction)
```

Settings: 100 background samples, 100 explain samples, `seed=42` for
reproducibility.

## 2. Embedding Analysis

For categorical features encoded with embeddings (typically `insurance_plan`,
`face_amount_band`, `class_enh`):

1. **Cosine similarity heatmap** — which categories the model treats as similar
2. **PCA 2D plot** — visualize category relationships
3. **Embedding weights table** — raw dimension values

Check that the learned similarities match actuarial intuition (e.g., UL near
ULSG, VL near VLSG). Surprising groupings often indicate sparse-category issues
or that the model is using the embedding as a proxy for something else.

## 3. Overfitting via Training History

- Plot train vs validation loss across epochs
- Flag if validation loss bottoms out early then rises (classic overfit)
- Cross-reference with the Train/Test A/E gap from `model-review`

## 4. NN-Specific Improvement Suggestions

If early-duration A/E is poor:
- Add explicit duration indicators (dur_1, dur_2, …)
- Add a select-period flag
- Consider separate select/ultimate models

If age A/E shows patterns:
- Revisit spline configuration (n_knots, degree)
- Consider `attained_age × smoker_status` interaction

If a categorical embedding looks noisy:
- Check minimum exposure per category; merge sparse categories
- Try OHE for low-cardinality features instead of an embedding
- Reduce embedding dimensionality

## Output Addendum

Append to the `model-review` output:

```markdown
### NN-Specific Findings
- SHAP top features: …
- Embedding observations: …
- Loss curve / overfitting read: …

### NN-Specific Recommendations
1. …
2. …
```
