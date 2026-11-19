# Portfolio construction and evaluation

**Research question:** can macro-aware volatility forecasting improve out-of-sample forecast
accuracy for cyclical-sector ETFs relative to historical-volatility benchmarks, and do these gains
translate into better risk-adjusted performance in volatility-managed portfolios?

This folder answers the second half. It turns each model's monthly variance forecasts into
volatility-managed portfolios of the five cyclical ETFs (XLB, XLE, XLF, XLI, XLY), following
Moreira & Muir (2017).

We evaluate each portfolio against two benchmarks: **buy-and-hold**, which tests whether
volatility timing improves the cyclical portfolio at all, and the **historical-volatility rule**
(`hist_rw`), which tests whether a fitted forecasting model adds value beyond using last month's
realised variance. We then compare the models **block by block**, where each block isolates one
modelling choice:

| Block | Question | Pairs (a vs b) |
|---|---|---|
| B0 | Does fitting a historical model improve on simple volatility timing? | `garch` vs `hist_rw` |
| B1 | Does macro help within GARCH? | `garch_midas_baa10y` vs `garch` |
| B2 | Do macro + VIX help a linear model that uses only past volatility? | `en_full_macro` vs `en_no_macro` |
| B3 | Is VIX doing all the work in Elastic Net? | `en_full_macro` vs `en_no_vix` · `en_vix_only` vs `en_no_macro` · `en_full_macro` vs `en_vix_only` |
| B4 | The same for XGBoost (robustness check for B3) | `xgb_full_macro` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_no_vix` · `xgb_vix_only` vs `xgb_no_macro` · `xgb_full_macro` vs `xgb_vix_only` |
| B5 | With matched inputs, does XGBoost beat Elastic Net? | `xgb_*` vs `en_*` for no macro, macro without VIX, VIX only, full macro + VIX |

In each pair, a is the richer model; a positive difference means the addition helps (higher Sharpe
ratio, or lower forecast loss). Portfolio pairs are tested with Ledoit-Wolf (2008) Sharpe ratio
tests; the forecast side (e1 to e5) tests the same pairs. The blocks are defined in `BLOCKS` in
`models/research_config.py`.

## Method

- **Weight:** wᵢ,ₜ = min(σ̄²ᵢ / σ̂²ᵢ,ₜ₊₁, 2), where σ̂² is the model's forecast (`fc_level`) of next
  month's realised variance.
- **Realised variance:** the sum of squared daily excess returns in the month, the same target as
  every forecast family.
- **Reference σ̄², two specifications:**
  - **fixed** (main): the ETF's mean monthly RV over the training window, 2003-03 to 2008-02, the
    same every month. Only the forecast moves the weights, so comparing portfolios across models
    tests forecast skill.
  - **expanding** (robustness): the mean from 2003-03 up to month t. It also re-levers the
    portfolio after crises; the **control** portfolio, min(σ̄²ᵢ,ₜ / σ̄²ᵢ,fixed, 2) with no forecast,
    isolates that effect (rp2).
- **Leverage cap:** k = 2 in the main results; rp1 rebuilds the fixed-σ̄² portfolios with k = 1.5,
  k = 3 and no cap.
- **Portfolio:** the 1/N average of the ETFs' managed excess returns, rescaled to buy-and-hold
  volatility over the out-of-sample period. For the per-ETF results, each ETF's managed series is
  rescaled to its own buy-and-hold volatility.
- **Benchmarks:** buy-and-hold (BH, w = 1) asks whether volatility timing works at all. `hist_rw`
  (σ̂²ₜ₊₁ = RVₜ, the Moreira & Muir portfolio) asks whether a forecast adds anything beyond the
  simplest volatility timing.
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months), the same for every model. Each
  forecast row dated t is the forecast for month t+1; p1 checks this on every run.

## Folder layout

```text
portfolio/          # p1–p3 notebooks (described below)
├── config.py       # Portfolio-only settings
├── results/
│   ├── portfolios/
│   └── eval/
└── robustness/     # rp1–rp2 and results/
```

## Notebooks

| Notebook | What it does |
|---|---|
| `p1_build_portfolios` | Loads forecasts and returns, builds every portfolio under both σ̄² specifications, saves them to `results/portfolios/` |
| `p2_portfolio_eval` | Compares each portfolio with BH and `hist_rw`, runs the block-by-block pair tests, for the cyclical bucket and per ETF; saves to `results/eval/` |
| `p3_forecast_vs_portfolio` | Ranks models by forecast accuracy (QLIKE) and by Sharpe ratio, and checks whether the two rankings agree (Spearman ρ) |
| `robustness/rp1_different_leverages` | Rebuilds the portfolios with leverage caps k = 1.5, 2, 3 and no cap |
| `robustness/rp2_timing_vs_relevering` | Under the expanding σ̄², separates the forecast's contribution from plain re-levering |

Portfolio-only settings live in [config.py](config.py): forecast column, leverage cap and σ̄²
specifications. See [shared configuration](../README.md#shared-configuration) for ETFs, sample,
models and comparison blocks.

## How to run

For automatic execution, see the [root pipeline instructions](../../README.md#run-the-analysis).
For manual execution, first prepare the data and [build the forecasts](../forecasting/README.md#how-to-run), then:

1. Run **p1 → p2**.
2. If the forecasts have changed, rerun `models/forecasting/evaluation/e6_naive_vs_all_models.ipynb`,
   then run **p3**.
3. Run **rp1** and **rp2** in any order after p1.

## Outputs

`results/portfolios/`, written by p1:

| File | Contents |
|---|---|
| `fixed/`, `expanding/` | Per specification: `returns.csv` (BH and each managed bucket), `positions.csv` and `weights.csv` (per ETF), `c.csv` (rescale factors), `sigma2_bar.csv`. `expanding/` also holds the control |
| `xret.csv` | Monthly excess returns of each ETF |
| `weight_diagnostics.csv` | Weight diagnostics per specification, model and ETF |
| `weight_diagnostics_pooled.csv` | The same, pooled across ETFs |
| `run_info.json` | Settings of the run (models, pairs, window), read by p2 and p3 |

`results/eval/`, written by p2 and p3:

| File | Written by | Contents |
|---|---|---|
| `vs_benchmarks_bucket_<spec>.csv` | p2 | Every bucket portfolio against BH (performance, Ledoit-Wolf, α) and against `hist_rw` |
| `vs_benchmarks_per_etf_<spec>.csv` | p2 | The same per ETF |
| `pairs.csv` | p2 | Every block-pair Ledoit-Wolf test (spec, block, pair, a, b, ETF) |
| `scoreboard_bucket.csv` | p2 | The portfolio answer per pair, under both specifications |
| `forecast_ranking.csv` | p3 | Models ranked by QLIKE per unit, with the QLIKE gain % vs `hist_rw` and e6's DM and Clark-West p-values on the log-RV MSE |
| `portfolio_ranking_<spec>.csv` | p3 | Models ranked by Sharpe ratio per unit, with ΔSR vs `hist_rw` and the Ledoit-Wolf p-value (main σ̄²) |
| `rank_summary_<spec>.csv` | p3 | Spearman ρ between accuracy and Sharpe ratio, with and without `hist_rw`, and `hist_rw`'s ranks (main σ̄²) |

`<spec>` is `fixed` or `expanding`; p3 writes the main specification (`fixed`) only. In p3 the unit
`Cyclical` is the pooled bucket.

`robustness/results/`, written by rp1 and rp2:

| File | Written by | Contents |
|---|---|---|
| `rp1_different_leverages.csv` | rp1 | Per cap and model: SR, ΔSR and Ledoit-Wolf p vs BH, α on BH, % capped, max weight, c |
| `rp2_timing_vs_relevering.csv` | rp2 | Control regressions under the expanding σ̄² (formerly `extension.csv` from p2) |
