# Changelog

## [0.3.5](https://github.com/jkoestner/morai/tree/v0.3.5)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.3.4...v0.3.5)

**Enhancements**
- add claude skills (model-review, nn-review, mortality-table-build)
- cdc analytic enhancements (weekly, flu, cod refinements)
- create experience study calcs (variance, credibility, ci, summarize data)
- enhance the neural model including
  - new methods (score, rebalance_ae, set_deterministic, _loss)
  - provide test/train a/e during fit
  - add new parameters (device, min_delta, ae_interval, loss_target)

**Documentation**
- update tests

**Bugs**
- fixed a few neural model bugs
  - add weight decay only on non-bias non-embedding columns
  - fix `init bias` for binomial path
  - load the best state epoch instead of last state epoch

## [0.3.4](https://github.com/jkoestner/morai/tree/v0.3.4)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.3.3...v0.3.4)

**Enhancements**
- add exposure and experience study logic
- dashboard enhancements (bookmarks, analytics page, skeleton loads)
- update dependencies (i.e. pandas to 3.0.0) and actions

**Bug Fixes**
- enhance edge case for pdp
- performance improvements, especially in dashboard
- add r initialization from a saved state

**Documentation**
- new notebooks including (duckdb v polars, experience_study)
- update tests
- add mypy static type checker

## [0.3.3](https://github.com/jkoestner/morai/tree/v0.3.3)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.3.2...v0.3.3)

**Enhancements**
- add batch_size and clean up logging in `neural`
- enhance the pdp processing in `charters`

**Documentation**
- update tests

## [0.3.2](https://github.com/jkoestner/morai/tree/v0.3.2)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.3.1...v0.3.2)

**Enhancements**
- improve web dashboard with mobile device
- add residual analytic plots (qq plot, residual plot, calibration plot) in `charters`
- enhance pdp plot with ohe and spline functionality in `charters`
- add penalty options for GLM models (L1, L2, elasticnet) in `core`
- add spline creation in `preprocessing`
- improve neural network model in `neural`

**Bugs**
- small bug fixes

**Documentation**
- update notebooks
- add CLAUDE.md

## [0.3.1](https://github.com/jkoestner/morai/tree/v0.3.1)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.3.0...v0.3.1)

**Bugs**
- update version package compatability including polars functional changes
with version 1.33.0

**Documentation**
- enhance test coverage

## [0.3.0](https://github.com/jkoestner/morai/tree/v0.3.0)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.2.0...v0.3.0)

**Enhancements**
- add neural network model
- add relative risk charts
- add population excess death trends for CDC

**Documentation**
- update notebooks
- created tests (relative risk, normalize, polars, 
  metrics, bin features, groupby features, tables)

## [0.2.0](https://github.com/jkoestner/morai/tree/v0.2.0)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.1.0...v0.2.0)

**Enhancements**
- add GAM models to forecast (mcgv, pygam, statsmodels)
- add HMD integration (login needed)
- add in exports for tables

**Documentation**
- update notebooks to have images retained
- split out models into it's own folder
- update README

## [0.1.0](https://github.com/jkoestner/morai/tree/v0.1.0)

[Full Changelog](https://github.com/jkoestner/morai/compare/v0.0.2...v0.1.0)

**Enhancements**
- extended functionality to use polars in dashboards
- add credibility methods
- add cdc integration
- add whittaker-henderson-lowrie graduation

**Documentation**
- better dashboard graphics
- add in CHANGELOG
- add in type hints
