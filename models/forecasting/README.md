# Volatility forecasting and evaluation

**Research question:** can macro-aware volatility forecasting improve out-of-sample forecast
accuracy for cyclical-sector ETFs relative to historical-volatility benchmarks, and do these gains
translate into better risk-adjusted performance in volatility-managed portfolios?

This folder answers the first half. It produces one-month-ahead forecasts of realised variance for
the five cyclical ETFs (XLB, XLE, XLF, XLI, XLY) from four model families, then tests whether the
macro-aware models forecast better than simpler ones. The portfolio side
(`models/portfolio/`) turns the same forecasts into volatility-managed portfolios.

We evaluate each model against the **historical-volatility benchmark** (`hist_rw`, next month's
variance forecast is this month's realised variance), which tests whether a fitted model adds
anything to the simplest forecast. We then compare the models **block by block**, where each block
isolates one modelling choice:

| Block | Question | Pairs (a vs b) | Notebook |
|---|---|---|---|
| B1 | Does macro help within GARCH? | `garch_midas_baa10y` vs `garch` | e1 |
| B2 | Do macro + VIX help a linear model that uses only past volatility? | `en_full_macro` vs `en_no_macro` | e2 |
| B3 | Is VIX doing all the work in Elastic Net? | `en_full_macro` vs `en_no_vix` · `en_vix_only` vs `en_no_macro` · `en_full_macro` vs `en_vix_only` | e3 |
| B4 | The same for XGBoost (robustness check for B3) | `xgb_full_macro` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_no_vix` · `xgb_vix_only` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_vix_only` | e4 |
| B5 | With matched inputs, does XGBoost beat Elastic Net? | `xgb_*` vs `en_*` for no macro, macro without VIX, VIX only, full macro + VIX | e5 |
| — | Which models beat the historical-volatility benchmark? | every selected model vs `hist_rw` | e6 |

In each pair, a is the richer model; a positive difference (`gain_t > 0`) means the addition helps
(lower forecast loss). The portfolio side (p2) runs Ledoit-Wolf Sharpe ratio tests on the same
pairs, plus B0 (`garch` vs `hist_rw`), so the two answers can be matched. The blocks are defined in
`BLOCKS` in `models/research_config.py`.

## Method

- **Target:** realised variance (RV) of month t+1, the sum of squared daily excess returns in the
  month. Every family forecasts the same target, and each forecast row dated t (the origin) is the
  forecast for `target_date` t+1.
- **Estimation:** expanding window with a 60-month burn-in (training window 2003-03 to 2008-02),
  refitted every month using only information available at the origin.
- **Models:**

  | Family | Variants | Build notebook | Notes |
  |---|---|---|---|
  | Historical | `hist_rw` | b0 | RV̂ₜ₊₁ = RVₜ; no estimated parameters (the Moreira & Muir portfolio) |
  | GARCH | `garch`, `garch_midas_baa10y` | b1 | Daily GARCH(1,1) by MLE; GARCH-MIDAS adds a long-run component driven by the credit spread (BAA10Y), with θ estimated per ETF. BAA10Y was chosen from four candidate drivers (UNRATE, CPIAUCSL, DGS3MO, BAA10Y) |
  | Elastic Net | `en_no_macro`, `en_no_vix`, `en_vix_only`, `en_full_macro` | b2 | Forecasts log RV; `alpha` and `l1_ratio` re-tuned monthly by 5-fold `TimeSeriesSplit` |
  | XGBoost | `xgb_no_macro`, `xgb_no_vix`, `xgb_vix_only`, `xgb_full_macro` | b3 | Forecasts log RV; refitted monthly, re-tuned every 24 months over an 8-point grid |

- **Predictor sets** (Elastic Net and XGBoost): every variant includes last month's log RV.
  `full_macro` adds all 11 macro variables (T10Y2Y, DGS3MO, BAA10Y, CPIAUCSL, T5YIE,
  T5YIE_MINUS_T10YIE, VIXCLS, INDPRO, UNRATE, UMCSENT, DCOILWTICO); `no_vix` drops VIXCLS;
  `vix_only` keeps VIXCLS only; `no_macro` uses no macro variables.
- **Level forecast:** log-RV models are converted with a log-normal correction,
  `fc_level = exp(fc_log + sigma2 / 2)`, where `sigma2` is the variance of the cross-validation
  residuals. GARCH and `hist_rw` forecast the level directly (`sigma2` is NaN).
- **Losses:** MSE on log RV, [log RV − log F]², is the primary loss; QLIKE,
  RV/F − log(RV/F) − 1 (Patton, 2011), on the level forecast; MSE on the level is reported only.
- **Tests:** nested pairs (a contains b's information) use **Clark-West** on MSE on log RV.
  **Diebold-Mariano** is used on QLIKE, and on MSE on log RV for non-nested pairs (B5, and GARCH /
  XGBoost vs `hist_rw` in e6). Every Elastic Net variant nests `hist_rw`, so e6 uses Clark-West for
  those. P-values are one-sided (H1: a beats b), with Newey-West standard errors and the Andrews
  (1991) bandwidth. The pooled result averages the loss differential across ETFs each month; the
  per-ETF results are exploratory, with no multiple-testing correction.
- **Caveats:** Clark-West is derived for OLS-estimated linear models, so it is an approximation for
  GARCH-MIDAS (MLE), tuned Elastic Net and XGBoost. DM is undersized for nested pairs, so its
  p-values there are supporting evidence only. B5 compares the two procedures as built, including
  their different re-tuning schedules (monthly vs every 24 months).
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months per ETF), the same for every model.
  Each notebook scores the common (ETF, month) rows where all compared models have a forecast.

## Folder layout

```text
forecasting/
├── build/          # b0–b3: model estimation
├── forecasts/      # Main forecasts and GARCH-MIDAS coefficients
├── evaluation/     # e1–e6: forecast comparisons
├── results/        # Build diagnostics and evaluation tables
└── robustness/     # r1–r3, crisis_config.py, forecasts/ and results/
```

## Notebooks

| Notebook | What it does |
|---|---|
| `build/b0_hist_rw` | Builds the random-walk benchmark `hist_rw` from monthly RV |
| `build/b1_garch_garchmidas` | Fits GARCH(1,1) and four GARCH-MIDAS candidates monthly, compares the candidate drivers, exports GARCH and BAA10Y GARCH-MIDAS forecasts and the θ path |
| `build/b2_elastic_net` | Builds the four Elastic Net variants; also builds the 60-month rolling-window forecasts for r3 |
| `build/b3_xgboost` | Builds the four XGBoost variants (plus an OLS lag baseline), with SHAP contributions and exploratory Clark-West tests |
| `evaluation/e1_garch_family` | B1: does the BAA10Y long-run component beat GARCH; how θ moves across origins |
| `evaluation/e2_macro_value` | B2: full macro + VIX vs past log RV alone, Elastic Net |
| `evaluation/e3_vix_effect_en` | B3: the contribution of VIX vs the other macro variables, Elastic Net |
| `evaluation/e4_vix_effect_xgb` | B4: the same VIX comparisons for XGBoost |
| `evaluation/e5_linear_vs_nonlinear` | B5: XGBoost vs Elastic Net with matched inputs |
| `evaluation/e6_naive_vs_all_models` | Every selected model vs `hist_rw`; its tables feed portfolio p3 |
| `robustness/r1_crisis_en` | Reruns the B3 tests without the GFC and/or COVID months |
| `robustness/r2_crisis_naive_vs_garch_en` | Reruns the e6 tests for GARCH and Elastic Net without the GFC and/or COVID months |
| `robustness/r3_rolling_window_en` | Reruns B3.2 (`en_vix_only` vs `en_no_macro`) with a 60-month rolling window instead of an expanding one |

The crisis windows (r1, r2) are the NBER recessions in `robustness/crisis_config.py`: GFC 2007-12
to 2009-06 (in sample from 2008-03) and COVID 2020-02 to 2020-04. A forecast is dropped when its
`target_date` falls in a window; nothing is refitted.

## How to run

For automatic execution, see the [root pipeline instructions](../../README.md#run-the-analysis).
For manual execution, review the [shared configuration](../README.md#shared-configuration), then:

1. Prepare `dataset/` (daily excess returns, monthly realised variance, final macro data).
2. Run **b0 → b1 → b2 → b3**. b1 reuses saved GARCH fits from `supporting/saved_fits/` when the
   data and settings are unchanged (`DSE4101_USE_FIT_CACHE=0` refits everything). b2 loads its
   saved forecasts unless `DSE4101_RECOMPUTE_FORECASTS=1` or the file is missing; a full Elastic
   Net rebuild is slow.
3. Run **e1 → e6** in any order after the build notebooks. Rerun e6 whenever the forecasts change,
   since portfolio p3 reads its tables.
4. Run **r1, r2, r3** in any order after the build notebooks (r1 and r3 compare against
   `e3_evaluation.csv`, r2 against `e6_evaluation.csv`).

## Outputs

`forecasts/`, written by b0 to b3:

| File | Written by | Contents |
|---|---|---|
| `hist_rw_forecasts.csv` | b0 | `hist_rw` |
| `garch_forecasts.csv` | b1 | `garch`, `garch_midas_baa10y` |
| `garchmidas_theta_by_origin.csv` | b1 | The BAA10Y θ per ETF and origin, read by e1 |
| `en_forecasts.csv` | b2 | The four `en_*` variants |
| `xgb_forecasts.csv` | b3 | The four `xgb_*` variants, plus `ols_har_no_macro` (not in the selected models) |

Every forecast file has the columns `model`, `etf`, `date` (origin, month end), `target_date`
(next month end), `actual_logRV`, `actual_RV`, `fc_log`, `sigma2` and `fc_level`.

`results/`, written by the build and evaluation notebooks:

| File | Written by | Contents |
|---|---|---|
| `e<n>_losses.csv` | e1 to e6 | Mean QLIKE, MSE on log RV and MSE on the level per model and ETF, plus `pooled` |
| `e<n>_evaluation.csv` | e1 to e6 | One row per pair, ETF, loss and test: `loss_ratio`, `gain_t`, `p_value`, `pct_change` |
| `garch_midas_candidate_losses.csv` | b1 | Losses of the four GARCH-MIDAS candidate drivers and GARCH, per ETF and pooled |
| `xgb_shap_mean_abs.csv`, `xgb_shap_share_pct.csv` | b3 | Mean absolute SHAP contribution of each predictor (`xgb_full_macro`), and its share of the total |
| `xgb_shap_contributions.csv` | b3 | SHAP contribution of each predictor per ETF and month |

In the evaluation files, `loss_ratio` is loss(a) / loss(b), `pct_change = (loss_ratio − 1) × 100`
(negative means a has the lower loss), and `gain_t > 0` favours a. The Clark-West statistic adjusts
for nesting, so it can be significant while the ratio is above 1. e6 labels its pairs E6.1 to
E6.10, one per selected model against `hist_rw`.

`robustness/forecasts/` and `robustness/results/`:

| File | Written by | Contents |
|---|---|---|
| `forecasts/en_rolling60_forecasts.csv` | b2 | `en_no_macro_roll60`, `en_vix_only_roll60` |
| `results/r1_crisis_evaluation.csv` | r1 | The B3 tests per `sample` (full, ex_GFC, ex_COVID, ex_both), with `n_months` |
| `results/r2_crisis_naive_vs_garch_en_evaluation.csv` | r2 | The e6 tests for GARCH and Elastic Net per `sample` |
| `results/r3_rolling_evaluation.csv` | r3 | The B3.2 tests per `window` (expanding, rolling) |
