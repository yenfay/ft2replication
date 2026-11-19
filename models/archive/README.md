# Archived research

This is the single guide to the archive. For current notebooks, configuration and run order,
use [the active research guide](../README.md).

| Folder | Contents |
|---|---|
| `scripts/elastic_net/` | Earlier Elastic Net experiments |
| `scripts/garch/` | Earlier GARCH-MIDAS work |
| `scripts/XGboost/` | Earlier XGBoost work |
| `scripts/ols_har.ipynb` | Earlier OLS/HAR notebook |
| `scripts/portfolio_construction/` | Earlier portfolio workflow and its saved outputs |
| `forecasts/` | Former root-level forecast exports |
| `dataset/price/` | Superseded price cleaner, original CRSP input and unadjusted daily panel |

Archived code and outputs retain their historical contents. Paths may refer to the old layout;
these files are excluded from the active workflow.

## Historical portfolio workflow

The notes below describe the archived implementation, not the current portfolio notebooks.
It constructed volatility-managed portfolios of five cyclical ETFs (XLB, XLE, XLF, XLI, XLY),
following Moreira & Muir (2017). File names are relative to `scripts/portfolio_construction/`.

### Historical files

| File | Purpose |
|---|---|
| `config.py` | Settings: models (`SELECTED`), cap `K_CAP = 2`, training window, `SPECS`, model pairs (`COMPARISONS`) |
| `portfolio_utils.py` | Shared functions |
| `01_build_portfolios.ipynb` | Builds and saves every portfolio. No evaluation |
| `02_eval_vs_buy_and_hold.ipynb` | Does volatility timing beat buy-and-hold? |
| `03_eval_model_comparisons.ipynb` | Does one forecast give a better portfolio than another? (research question) |
| `04_forecast_vs_portfolio.ipynb` | Do more accurate forecasts give better portfolios? |

The historical run order was 01 → 02 → 03 → 04, from that folder. Notebook 01 wrote
`results/portfolios/`; notebooks 02–04 wrote `results/eval/`.

### Historical method

- **Weight:** w = min(σ̄² / σ̂², 2), where σ̂² is the model's forecast of next month's variance.
- **Two references σ̄²:**
  - **fixed** (main): the 2003-03 to 2008-02 mean, the same every month;
  - **expanding** (robustness): the mean up to the current month.
- **Portfolio:** the 1/N average of the ETFs' managed excess returns, rescaled to buy-and-hold
  volatility. Buy-and-hold uses w = 1.
- **Sample:** out-of-sample from 2008-03 to 2025-12 (214 months), the same for every model.
- **Control (expanding only):** weight min(σ̄²_t / σ̄²_fixed, 2) with no forecast. It isolates the
  re-levering caused by the moving reference.

### Historical tests

| Notebook | Test |
|---|---|
| 02 | Ledoit-Wolf test of SR(model) = SR(BH); α on BH (Newey-West); drawdown, turnover, % at the cap. Extension: α on BH and the control |
| 03 | Ledoit-Wolf test of SR(a) = SR(b) for each pair in `COMPARISONS`, under both references |
| 04 | QLIKE and MSE forecast losses, ranks against Sharpe ranks, Spearman ρ |

