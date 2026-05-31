# Actuarial Reference for Mortality Model Work

Shared reference for the `model-review`, `nn-review`, and `mortality-table-build` skills.

## Key Terminology

### A/E Ratio (Actual to Expected)
- **A/E = 1.00**: Model perfectly predicts mortality
- **A/E < 1.00**: Model over-predicts (actual deaths less than expected)
- **A/E > 1.00**: Model under-predicts (actual deaths more than expected)

### Select vs Ultimate Mortality
- **Select Period**: First 15-25 policy years after underwriting
- **Select Mortality**: Lower mortality due to recent underwriting (healthy lives)
- **Ultimate Mortality**: Stable mortality after select period wears off
- **Selection Wear-off**: Gradual increase in mortality as select advantage fades

### VBT 2015
- **VBT**: Valuation Basic Table (SOA industry mortality table)
- **qx_vbt15**: Mortality rate from VBT 2015
- Used as benchmark for model comparisons

## Data Dimensions

### Insurance Plans
| Plan | Description | Mortality Characteristics |
|------|-------------|--------------------------|
| Term | Temporary coverage | Generally better mortality (more underwriting) |
| Perm | Permanent whole life | Stable, traditional product |
| UL | Universal Life | Flexible premium, moderate mortality |
| ULSG | UL with Secondary Guarantee | Often older ages, different risk profile |
| VL | Variable Life | Investment component, varied demographics |
| VLSG | VL with Secondary Guarantee | Similar to ULSG |

### Preferred Classes
- **Number of Preferred Classes**: 1-4 tiers of underwriting
- **Preferred Class**: Specific tier (1=best, 4=standard)
- **class_enh**: Combined field (e.g., "2_1" = 2 classes, tier 1)

### Face Amount Bands
Higher face amounts typically indicate more rigorous underwriting, higher
socioeconomic status, and better mortality experience.

## Expected Patterns

### By Duration
| Duration | Expected A/E Pattern | Reason |
|----------|---------------------|--------|
| 1-5 | Lower (0.60-0.80) | Strong select effect |
| 6-10 | Rising (0.80-0.95) | Select wearing off |
| 11-15 | Near 1.00 | Approaching ultimate |
| 16+ | Stable (~1.00) | Ultimate mortality |

### By Age
| Age Band | Considerations |
|----------|---------------|
| 30-50 | Lower deaths, higher variance |
| 51-70 | Core mortality experience |
| 71-80 | Increasing mortality rates |
| 81+ | High mortality, credibility concerns |

### By Smoker Status
| Status | Expected Pattern |
|--------|-----------------|
| NS (Non-Smoker) | Better mortality, majority of exposure |
| S (Smoker) | Higher mortality, smaller population |

## Model Quality Benchmarks

### A/E Tolerance by Segment Size
| Exposure Level | Acceptable A/E Range |
|----------------|---------------------|
| Very High (>$100B) | 0.95 - 1.05 |
| High ($10B-$100B) | 0.90 - 1.10 |
| Medium ($1B-$10B) | 0.85 - 1.15 |
| Low (<$1B) | 0.75 - 1.25 |

### Overfitting Indicators
- Train A/E significantly better than Test A/E (>0.05 difference)
- Very low training loss but high validation loss
- Model performs well on seen data but poorly on holdout

## Standard Review Dimensions

When grouping A/E for review, use these dimensions (binning shown where relevant):

- `sex` (M/F)
- `smoker_status` (NS/S)
- `attained_age` (binned: 30-40, 41-50, 51-60, 61-70, 71-80, 81+)
- `duration` (binned: 1-5, 6-10, 11-15, 16-20, 21+)
- `observation_year`
- `insurance_plan` (Term, Perm, UL, ULSG, VL, VLSG)
- `face_amount_band` or `binned_face`
- `class_enh` (preferred class combination)

## Feature Engineering Best Practices

### Splines for Age
- Use B-splines with 6-10 knots for `attained_age`
- Quantile-based knots capture data distribution
- Degree 3 (cubic) provides smooth curves

### Encoding by Model Family
| Feature | GLM | GAM | CatBoost | Neural |
|---------|-----|-----|----------|--------|
| sex | OHE | OHE | native cat | OHE/ordinal |
| smoker_status | OHE | OHE | native cat | OHE/ordinal |
| insurance_plan | OHE | OHE | native cat | Embedding |
| face_amount_band | ordinal | spline | native cat | Embedding |
| preferred_class | OHE | OHE | native cat | Embedding |
| attained_age | binned/poly | spline | numeric | spline or numeric |

### Interaction Terms to Consider
- `duration × preferred_class` (select effect varies by underwriting)
- `attained_age × smoker_status` (age-mortality slope differs)
- `duration × insurance_plan` (select patterns vary by product)

## Model-Family Notes

### CatBoost
- Handles categoricals natively; no OHE needed
- Use built-in feature importance (`PredictionValuesChange`, `LossFunctionChange`)
- SHAP via TreeExplainer (fast, exact)

### GLM (`morai.models.core.GLM`)
- Coefficients are log-odds (logit link) or log-rates (log link) — interpret carefully
- Use `calc_likelihood_ratio` to compare nested models
- Watch for unstable coefficients (large SE) on sparse categories

### GAM (`morai.models.r.GAMR`)
- Inspect smooth term plots — non-monotonic curves at the tails often indicate
  over-flexible splines
- Effective degrees of freedom (EDF) per term signals complexity

### Neural (`morai.models.neural.Neural`)
- Embeddings for high-cardinality categoricals
- SHAP via KernelExplainer (slow, sample 100 background / 100 explain)
- Inspect training/validation loss curves for overfitting
