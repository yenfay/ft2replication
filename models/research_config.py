"""Research design shared by the notebooks in models/: the ETFs, the sample, the models and the
comparison blocks.

Research question: can macro-aware volatility forecasting improve out-of-sample forecast accuracy
for cyclical-sector ETFs relative to historical-volatility benchmarks, and do these gains
translate into better risk-adjusted performance in volatility-managed portfolios?

Portfolio-only settings (leverage cap, sigma2_bar specifications) are in
portfolio/config.py.
"""

TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLY"]   # cyclical bucket

TRAIN_START = "2003-03"    # 2003-01 is dropped and 2003-02 has no lagged value
TRAIN_MONTHS = 60          # training window 2003-03 to 2008-02; out-of-sample is every month after it

# Model family (a key of FORECAST_FILES in utils/forecast_data.py) -> variants (values of the
# `model` column). Every selected variant must cover all five ETFs and every out-of-sample
# month from 2008-03 onwards.
SELECTED = {
    "Historical": ["hist_rw"],
    "Garch": ["garch", "garch_midas_baa10y"],
    "Elastic Net": ["en_no_macro", "en_no_vix", "en_vix_only", "en_full_macro"],
    "XGBoost": ["xgb_no_macro", "xgb_no_vix", "xgb_vix_only", "xgb_full_macro"],
}

# The historical-volatility benchmark: RV_hat(t+1) = RV(t) (forecasting/build/b0_hist_rw.ipynb).
BENCHMARK = "hist_rw"

# Comparison blocks, each answering one sub-question. The forecast evaluation (DM tests) and the
# portfolio evaluation (Ledoit-Wolf tests) run the same pairs, so their answers can be matched.
#
# Each pair is (pair_id, a, b, what it isolates), with a the richer model and b the simpler one.
# A positive difference always means "the addition helps": a has the lower forecast loss, or the
# higher Sharpe ratio.
BLOCKS = {
    "B0": {
        "title": "Volatility-timing baseline",
        "question": "Does the simplest volatility timing beat buy-and-hold, and does fitting a "
                    "historical-volatility model improve on it?",
        "models": ["hist_rw", "garch"],
        "pairs": [
            ("B0.1", "garch", "hist_rw", "fitted GARCH vs naive history"),
        ],
    },
    "B1": {
        "title": "GARCH family",
        "question": "Does macro information help within the GARCH family?",
        "models": ["garch", "garch_midas_baa10y"],
        "pairs": [
            ("B1.1", "garch_midas_baa10y", "garch", "credit spread (BAA10Y) as the long-run component"),
        ],
    },
    "B2": {
        "title": "Macro information in a linear model",
        "question": "Do macro variables plus VIX help a linear model that uses only past volatility?",
        "models": ["en_no_macro", "en_full_macro"],
        "pairs": [
            ("B2.1", "en_full_macro", "en_no_macro", "total effect of macro + VIX"),
        ],
    },
    "B3": {
        "title": "Effect of VIX, Elastic Net",
        "question": "Is VIX doing all the work, or do the other macro variables add information?",
        "models": ["en_no_macro", "en_no_vix", "en_vix_only", "en_full_macro"],
        "pairs": [
            ("B3.1", "en_full_macro", "en_no_vix", "VIX on top of the other macro variables"),
            ("B3.2", "en_vix_only", "en_no_macro", "VIX on its own"),
            ("B3.3", "en_full_macro", "en_vix_only", "other macro variables on top of VIX"),
        ],
    },
    "B4": {
        "title": "Effect of VIX, XGBoost (robustness check for B3)",
        "question": "Do the B3 findings hold with a non-linear model?",
        "models": ["xgb_no_macro", "xgb_no_vix", "xgb_vix_only", "xgb_full_macro"],
        "pairs": [
            ("B4.0", "xgb_full_macro", "xgb_no_macro", "total effect of macro + VIX"),
            ("B4.1", "xgb_full_macro", "xgb_no_vix", "VIX on top of the other macro variables"),
            ("B4.2", "xgb_vix_only", "xgb_no_macro", "VIX on its own"),
            ("B4.3", "xgb_full_macro", "xgb_vix_only", "other macro variables on top of VIX"),
        ],
    },
    "B5": {
        "title": "Elastic Net vs XGBoost specifications",
        "question": "With matched macro inputs, does the XGBoost specification beat Elastic Net?",
        "models": ["en_no_macro", "xgb_no_macro", "en_no_vix", "xgb_no_vix",
                   "en_vix_only", "xgb_vix_only", "en_full_macro", "xgb_full_macro"],
        "pairs": [
            ("B5.1", "xgb_no_macro", "en_no_macro", "no macro"),
            ("B5.2", "xgb_no_vix", "en_no_vix", "macro without VIX"),
            ("B5.3", "xgb_vix_only", "en_vix_only", "VIX only"),
            ("B5.4", "xgb_full_macro", "en_full_macro", "full macro + VIX"),
        ],
    },
}
