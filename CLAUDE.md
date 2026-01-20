# Morai

Morai is a Python package for actuarial mortality and experience data analysis. Named after the Greek Moirai (the three fates), it provides tools for mortality rate forecasting, statistical modeling, and interactive data exploration.

## Quick Reference

### Common Commands

```bash
# Run tests
pytest --cov=morai --cov-report=html

# Run dashboard
morai dashboard
# or
python -m morai.dashboard.app

# Install locally for development
uv pip install -e .

# Install with optional dependencies
uv pip install -e .[dev][neural][r]

# Lint and format
ruff check .
black .
```

### Project Structure

```
morai/
├── morai/
│   ├── dashboard/       # Plotly/Dash web application
│   │   ├── pages/       # Dashboard pages (input, explore, experience, models, tables, cdc)
│   │   └── components/  # Reusable UI components
│   ├── models/          # Predictive models
│   │   ├── core.py      # GLM and GAM models
│   │   ├── neural.py    # PyTorch neural networks
│   │   └── r.py         # R-based models (mgcv)
│   ├── experience/      # Experience analysis
│   │   ├── experience.py    # Normalization and metrics
│   │   ├── charters.py      # Visualization tools
│   │   ├── credibility.py   # Credibility methods
│   │   └── eda.py           # Exploratory data analysis
│   ├── forecast/        # Forecasting utilities
│   │   ├── metrics.py       # Statistical metrics (sMAPE, MAPE, etc.)
│   │   ├── graduation.py    # WHL graduation methods
│   │   └── preprocessors.py # Data preprocessing
│   ├── integrations/    # External data sources
│   │   ├── cdc.py       # CDC Wonder API
│   │   └── hmd.py       # Human Mortality Database
│   └── utils/           # Helpers, CLI, config, logging
├── tests/               # pytest test suite
└── notebooks/           # Jupyter notebooks for analysis workflows
```

## Code Conventions

### Style

- **Line length:** 88 characters (Black default)
- **Linter:** Ruff
- **Formatter:** Black
- **Docstrings:** NumPy/SciPy style
- **Type hints:** Required on function signatures

### Patterns

- Models inherit from scikit-learn's `BaseEstimator` and `RegressorMixin`
- Support both pandas DataFrames and Polars DataFrames where applicable
- Use custom logger from `morai.utils.custom_logger`
- Dashboard uses YAML-based configuration

### Testing

- Test files in `tests/` directory, named `test_*.py`
- Coverage excludes: dashboard, CLI, neural, r modules
- Use pytest fixtures for common test data

## Key Dependencies

- **Data:** pandas, polars, numpy
- **ML/Stats:** scikit-learn, statsmodels, catboost, pygam
- **Neural:** torch (optional)
- **Visualization:** plotly, seaborn, dash
- **Mortality:** pymort

## Important Notes

- The dashboard runs on port 8001 by default
- CDC integration uses XML-based API queries
- HMD integration requires authentication credentials
- Neural network models require the `[neural]` optional dependency
- R-based GAM models require the `[r]` optional dependency and R installation
