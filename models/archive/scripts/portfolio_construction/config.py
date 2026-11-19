"""Settings shared by every notebook in scripts/portfolio_construction.

Change a setting here, then rerun 01_build_portfolios.ipynb followed by notebooks 02-04.
"""
from portfolio_utils import CYCLICAL

# Model family (a key of FORECAST_FILES in portfolio_utils.py) -> variants (values of the `model`
# column). Every selected model must have a forecast for the first out-of-sample month (2008-03):
# ols_1_3_12m and the *_har Elastic Net variants start in 2009-02, so they are left out.
SELECTED = {
    "Garch": ["garch", "garch_midas_baa10y"],
    "HAR OLS": ["ols_1m"],
    "Elastic Net": ["en_full_macro", "en_no_macro", "en_no_vix"],
    "XGBoost": ["xgb_full_macro", "xgb_no_macro", "xgb_no_vix", "xgb_vix_only"],
}
VARIANTS = [v for variants in SELECTED.values() for v in variants]

TICKERS = CYCLICAL
FORECAST_COL = "fc_level"  # log-normal corrected level forecast of next month's RV
K_CAP = 2                 # leverage cap on the base weight. 
TRAIN_START = "2003-03"    # 2003-01 is dropped and 2003-02 has no lagged value
TRAIN_MONTHS = 60          # training window 2003-03 to 2008-02; OOS is every month after it

# sigma2_bar specifications. "fixed": the 2003-03 to 2008-02 mean, the same every month.
# "expanding": the mean from 2003-03 through month t, for holding month t+1.
SPECS = ["fixed", "expanding"]

# Model pairs (a, b) tested in 03_eval_model_comparisons: H0 SR(a) = SR(b), under both specs.
COMPARISONS = [
    # macro vs no macro, within family
    ("en_full_macro", "en_no_macro"),
    ("xgb_full_macro", "xgb_no_macro"),
    # VIX contribution
    ("en_full_macro", "en_no_vix"),
    ("xgb_full_macro", "xgb_no_vix"),
    # ML vs GARCH
    ("en_full_macro", "garch"),
    ("xgb_full_macro", "garch"),
]
