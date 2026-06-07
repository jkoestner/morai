---
name: mortality-table-build
description: Build a mortality rate table from a fitted model. Use when generating a 1-D or 2-D rate table, applying graduation, adding an ultimate period, building select/ultimate structure, or producing an output table for downstream use. Distinct from model-review — this skill produces a deliverable table, not diagnostics.
allowed-tools: Read, Bash, Write, Edit, Glob, Grep
---

# Mortality Table Build

Workflow for turning a fitted mortality model into a deliverable rate table
using `morai.experience.tables` and `morai.forecast.graduation`.

For terminology (select/ultimate, VBT, class_enh) and feature conventions, read
`../_shared/actuarial-reference.md` before beginning.

## Inputs

- A fitted model (`files/models/<name>.joblib`) with `.predict(X)`
- The `mapping`, `preprocess_feature_dict`, and `preprocess_params` used at fit
- A target grid (or let `generate_table` build one from the mapping)
- Optional: benchmark table for comparison (e.g., VBT 2015)

## Workflow

### 1. Define the Table Shape
Decide:
- **Dimensions** the table is indexed by (e.g., `attained_age`, `duration`,
  `sex`, `smoker_status`)
- **Multiplier features** vs base features — multiplier features become a
  separate `mult_table` (see `mult_method="glm"` vs `"mean"` in
  `tables.generate_table`)
- **Select period length** (e.g., 25 years) and ultimate handling

### 2. Generate Raw Rates
```python
from morai.experience import tables

rate_table, mult_table = tables.generate_table(
    model=model,
    mapping=mapping,
    preprocess_feature_dict=preprocess_feature_dict,
    preprocess_params=preprocess_params,
    grid=None,  # auto-build from mapping
    mult_features=["insurance_plan", "class_enh"],
    mult_method="glm",
)
```

### 3. Add Attained-Age / Issue-Age / Duration Columns
Ensure all three columns exist and are consistent:
```python
rate_table = tables.add_aa_ia_dur_cols(rate_table, max_age=121)
tables.check_aa_ia_dur_cols(rate_table)
```

### 4. Graduate Rates (optional but usually needed)
Use `morai.forecast.graduation` (WHL methods) to smooth rates along
`attained_age`. Graduate within each (sex, smoker, duration) slice; never across
those splits.

### 5. Add Ultimate Period
For select/ultimate structures:
```python
rate_table = tables.add_ultimate(rate_table, ...)
```
Confirm the join point (last select duration → first ultimate age) is smooth —
visualize the transition.

### 6. Build Select/Ultimate Table (if applicable)
```python
su = tables.get_su_table(rate_table, select_period=25)
```

### 7. Validate
- Compare to benchmark (e.g., VBT 2015) with `tables.compare_tables`
- Spot-check monotonicity: rates should generally rise with `attained_age`
- Spot-check select effect: low duration < high duration at same attained age
- Confirm no NaN, no zero/negative rates, no rates > 1

### 8. Output
```python
tables.output_table(rate_table, ...)
```
Write to `files/dataset/<table_name>.<ext>`. Document the source model,
generation date, and any graduation parameters in a sibling README or in the
mapping metadata.

## Output Template

```markdown
## Mortality Table Build — <table name>

### Source
- Model: <path / type>
- Generation date: <YYYY-MM-DD>
- Grid: <dimensions>

### Structure
- Select period: <N> years
- Multiplier features: <list>
- Multiplier method: <glm | mean>

### Graduation
- Method: <WHL / none>
- Parameters: <…>

### Validation
- Comparison to <benchmark>: <summary>
- Monotonicity checks: <pass/fail with notes>
- Select wear-off pattern: <observed pattern>

### Deliverable
- Path: <files/dataset/...>
- Rows: <N>
```

## Pitfalls

- **Encoding mismatch:** rate_table must be generated with the SAME
  `preprocess_feature_dict` and `preprocess_params` used to train the model
- **Multiplier inflation at high rates:** with `mult_method="glm"`, multipliers
  diverge as the base prediction approaches 1 — sanity-check the high-age tail
- **Ultimate join discontinuity:** if the model wasn't trained on long
  durations, the ultimate period it produces can jump — consider blending with a
  reference ultimate table
- **Graduating across select/ultimate boundary:** don't — graduate select and
  ultimate separately
