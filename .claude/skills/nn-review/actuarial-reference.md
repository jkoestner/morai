# Actuarial Reference for Mortality Model Review

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
Higher face amounts typically indicate:
- More rigorous underwriting
- Higher socioeconomic status
- Better mortality experience

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

## Feature Engineering Best Practices

### Splines for Age
- Use B-splines with 6-10 knots for attained_age
- Quantile-based knots capture data distribution
- Degree 3 (cubic) provides smooth curves

### Embeddings vs OHE
| Feature | Recommended Encoding |
|---------|---------------------|
| sex | OHE or ordinal (2 categories) |
| smoker_status | OHE or ordinal (2-3 categories) |
| insurance_plan | Embedding (6 categories, relationships) |
| face_amount_band | Embedding (ordinal, many categories) |
| preferred_class | Embedding (ordinal) |
| class_enh | OHE or Embedding |

### Interaction Terms to Consider
- `duration × preferred_class` (select effect varies by underwriting)
- `attained_age × smoker_status` (age-mortality slope differs)
- `duration × insurance_plan` (select patterns vary by product)
