# Forecasting and portfolio research

Can macro-aware volatility forecasts improve out-of-sample accuracy for cyclical-sector ETFs
relative to historical-volatility benchmarks, and do these gains translate into better
risk-adjusted performance in volatility-managed portfolios?

## Guides

| Guide | Contents |
|---|---|
| [Project setup and pipeline](../README.md) | Data requirements, installation, automatic execution and output locations |
| [Forecasting](forecasting/README.md) | Model estimation, forecast tests, notebooks and output schemas |
| [Portfolio](portfolio/README.md) | Portfolio construction, performance tests, notebooks and output schemas |
| [Archive](archive/README.md) | Earlier experiments and historical workflow |

## Shared configuration

[research_config.py](research_config.py) defines the ETFs, sample window, selected models,
benchmark and comparison blocks. [portfolio/config.py](portfolio/config.py) holds portfolio-only
settings. Restart notebook kernels after changing imported configuration.

[utils/paths.py](utils/paths.py) resolves paths from the checkout; start notebook kernels anywhere
inside this repository. [utils/forecast_data.py](utils/forecast_data.py) defines forecast filenames
and validates coverage for every selected model, ETF and out-of-sample month, with dates and
realised values matching the current variance data. Shared portfolio functions live in
[utils/portfolio_utils.py](utils/portfolio_utils.py).

## Shared comparison blocks

The `BLOCKS` definition in [research_config.py](research_config.py) is the source for these pairs.
In each pair, a is the richer model; a positive difference means a has lower forecast loss or a
higher Sharpe ratio. Forecast tests run in e1–e6; portfolio p2 tests B0–B5 with Ledoit-Wolf
Sharpe ratio tests, so the forecast and portfolio answers can be matched.

| Block | Question | Pairs (a vs b) | Forecast notebook |
|---|---|---|---|
| B0 | Does fitting a historical model improve on simple volatility timing? | `garch` vs `hist_rw` | Covered by e6 |
| B1 | Does macro help within GARCH? | `garch_midas_baa10y` vs `garch` | [e1](forecasting/evaluation/e1_garch_family.ipynb) |
| B2 | Do macro + VIX help a linear model that uses only past volatility? | `en_full_macro` vs `en_no_macro` | [e2](forecasting/evaluation/e2_macro_value.ipynb) |
| B3 | Is VIX doing all the work in Elastic Net? | `en_full_macro` vs `en_no_vix` · `en_vix_only` vs `en_no_macro` · `en_full_macro` vs `en_vix_only` | [e3](forecasting/evaluation/e3_vix_effect_en.ipynb) |
| B4 | Do the VIX findings hold in XGBoost? | `xgb_full_macro` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_no_vix` · `xgb_vix_only` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_vix_only` | [e4](forecasting/evaluation/e4_vix_effect_xgb.ipynb) |
| B5 | With matched inputs, does XGBoost beat Elastic Net? | `xgb_*` vs `en_*` for no macro, macro without VIX, VIX only, full macro + VIX | [e5](forecasting/evaluation/e5_linear_vs_nonlinear.ipynb) |
| — | Which models beat the historical-volatility benchmark? | Every selected model vs `hist_rw` | [e6](forecasting/evaluation/e6_naive_vs_all_models.ipynb) |

## Workflow dependencies

Use the [root pipeline instructions](../README.md#run-the-analysis) for automatic execution.
For manual execution, follow the [forecasting run order](forecasting/README.md#how-to-run), then
the [portfolio run order](portfolio/README.md#how-to-run).

Build notebooks supply the forecasts used by both branches. Portfolio p3 additionally reads
`forecasting/results/e6_losses.csv` and `forecasting/results/e6_evaluation.csv`, so rerun e6
whenever forecasts change before running p3. p3 compares forecast-accuracy and portfolio-performance
rankings to answer whether forecast gains translate into better portfolios.
