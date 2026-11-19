# Sector ETF Volatility Forecasting

This project studies whether macro-financial predictors improve volatility forecasts
for XLB, XLE, XLF, XLI and XLY, and whether those forecasts improve volatility-managed
portfolios. The out-of-sample period is March 2008 to December 2025.

## Repository structure

```text
.
├── dataset/
│   ├── price/                 # Raw ETF data, return cleaning and monthly variance
│   └── macro_variables/       # Raw macro series, alignment and transformations
├── models/
│   ├── research_config.py     # Selected models, sample and comparison blocks
│   ├── forecasting/           # Forecast estimation, evaluation and robustness
│   ├── portfolio/             # p1–p3, portfolio settings and results
│   ├── utils/                 # Shared paths and portfolio functions
│   └── archive/               # Earlier experiments
├── requirements.txt
├── run_all.ipynb      # Notebook entrypoint for the full workflow
└── run_pipeline.py
```

## Data setup

The pipeline starts from the supplied raw CSVs. It does not download CRSP data.
The following files must be present:

```text
dataset/price/raw/sector_etf_prices_with_xlre.csv
dataset/price/raw/risk_free_rate_returns.csv
dataset/macro_variables/raw/   
```

Price-cleaning scripts create daily excess returns and monthly realised variance.
Macro scripts align observations to month-end information and produce eleven predictors.

## Installation

From the repository root, install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## Run the analysis

Open [run_all.ipynb](run_all.ipynb) for progress and links to fresh executed notebooks.
It begins with the report's table/figure source index and runs the complete pipeline,
including embedded and standalone robustness analyses. Archived experiments are excluded from execution.

The notebook defaults to saved raw data, rebuilding forecasts, and using the existing GARCH
fit cache. Set `INCLUDE_DATA_CHECKS=True` to add the two online FRED/ALFRED diagnostic notebooks
after macro transformation. These checks require internet and `FRED_API_KEY`, even with
`START="raw"`. Change `START` to `"fetch"` for fresh macro downloads. The settings and execution
cells use the current kernel's Python environment.

The command-line entrypoint uses the same runner. Rebuild from the supplied raw data:

```bash
python3 run_pipeline.py --start raw
```

The execution order is:

**Data preparation → EDA → b0–b3 → e1–e6 → forecast robustness checks → p1–p3 → rp1–rp2**

b2 also generates the Elastic Net rolling-window forecasts. To re-estimate every model
without reusing saved GARCH fits, add `--no-fit-cache`.

To rebuild data and rerun downstream analyses using saved forecasts:

```bash
python3 run_pipeline.py --start raw --reuse-forecasts
```

Saved main forecasts and Elastic Net rolling forecasts must cover their selected models, ETFs
and every out-of-sample month.
The runner checks their dates and realised values against the rebuilt variance data before evaluation.
Reuse does not verify whether forecasts reflect changes to model code or macro predictors.

To download fresh macro observations first, use `--start fetch` instead. This requires
`FRED_API_KEY` in your environment or the repository-level `.env` file. Both modes
require the CRSP exports above.

Add `--include-data-checks` to run the optional online macro checks from the CLI.
Use `--dry-run` to list the selected steps without execution. For example:

```bash
python3 run_pipeline.py --start raw --include-data-checks --dry-run
```

## Outputs

- Main forecasts: `models/forecasting/forecasts/`
- Forecast evaluation tables: `models/forecasting/results/`
- Portfolio returns and weights: `models/portfolio/results/portfolios/`
- Portfolio performance tables: `models/portfolio/results/eval/`
- Research robustness results: `models/forecasting/robustness/results/` and `models/portfolio/robustness/results/`

Runs overwrite processed data and result CSVs. Source notebook displays can be older
snapshots; the runner prints where it saves fresh executed copies. Full model fitting can take
substantial time.

See [the model guide](models/README.md) for shared configuration and comparison blocks,
and the [forecasting](models/forecasting/README.md) and [portfolio](models/portfolio/README.md)
guides for methods, individual notebooks, manual execution and output schemas.
