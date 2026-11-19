"""Build a daily panel of sector ETF raw and excess returns from CRSP data.

XLF is reconstructed from 19 September 2016 as a bundle of one XLF share
and 0.139146 XLRE shares, weighted by prior-close position values. XLRE is
then removed from the output. Risk-free rates are daily decimal returns.

Inputs: raw/sector_etf_prices_with_xlre.csv, raw/risk_free_rate_returns.csv
Output: clean/daily_returns_panel_ver2.csv
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
CLEAN_DIR = Path(__file__).resolve().parent / "clean"
PRICE_FILE = RAW_DIR / "sector_etf_prices_with_xlre.csv"
RF_FILE = RAW_DIR / "risk_free_rate_returns.csv"
OUT_FILE = CLEAN_DIR / "daily_returns_panel_ver2.csv"

SAMPLE_START_DATE = "2003-01-31"
XLRE_EX_DATE = "2016-09-19"
XLRE_PER_XLF_SHARE = 0.139146
PRICE_COL_CANDIDATES = ["DlyPrc", "DlyClose", "PRC"]

# Load daily returns and prices. Negative CRSP prices denote bid/ask midpoints.
price = pd.read_csv(PRICE_FILE)
price_col = next((c for c in PRICE_COL_CANDIDATES if c in price.columns), None)
if price_col is None:
    raise ValueError(f"A price column is required: {PRICE_COL_CANDIDATES}")

price = price[["Ticker", "DlyCalDt", "DlyRet", price_col]].copy()
price["DlyCalDt"] = pd.to_datetime(price["DlyCalDt"])
ret_wide = price.pivot_table(
    index="DlyCalDt", columns="Ticker", values="DlyRet", aggfunc="first"
).sort_index()
px_wide = price.pivot_table(
    index="DlyCalDt", columns="Ticker", values=price_col, aggfunc="first"
).sort_index().abs()
ret_wide = ret_wide.loc[SAMPLE_START_DATE:]

# Reconstruct the XLF/XLRE bundle using prices from the full trading calendar.
for ticker in ("XLF", "XLRE"):
    if ticker not in ret_wide.columns:
        raise ValueError(f"{ticker} is required to reconstruct the XLF series.")

ex_date = pd.Timestamp(XLRE_EX_DATE)
if ex_date not in ret_wide.index:
    next_dates = ret_wide.index[ret_wide.index >= ex_date]
    if next_dates.empty:
        raise ValueError(f"No trading dates on or after {XLRE_EX_DATE}.")
    ex_date = next_dates[0]
post_mask = ret_wide.index >= ex_date

px_prev = px_wide.shift(1).reindex(ret_wide.index)
value_xlf_prev = px_prev["XLF"]
value_xlre_prev = XLRE_PER_XLF_SHARE * px_prev["XLRE"]
w_xlf_prev = value_xlf_prev / (value_xlf_prev + value_xlre_prev)
w_xlre_prev = 1.0 - w_xlf_prev
ret_combined = w_xlf_prev * ret_wide["XLF"] + w_xlre_prev * ret_wide["XLRE"]
ret_wide.loc[post_mask, "XLF"] = ret_combined.loc[post_mask]
ret_wide = ret_wide.drop(columns="XLRE")

# Align daily decimal risk-free returns to the price panel's trading dates.
rf = pd.read_csv(RF_FILE)
date_col = "date" if "date" in rf.columns else rf.columns[0]
rate_col = "rf" if "rf" in rf.columns else rf.columns[1]
rf = rf[[date_col, rate_col]].rename(
    columns={date_col: "date", rate_col: "rf_daily"}
)
rf["date"] = pd.to_datetime(rf["date"])
price_dates = pd.DataFrame({"date": ret_wide.index})
rf_daily = price_dates.merge(rf, on="date", how="left").set_index("date")["rf_daily"]

# Export raw returns, excess returns and the daily risk-free rate.
raw_ret = ret_wide.add_prefix("ret_")
excess_ret = ret_wide.sub(rf_daily, axis=0).add_prefix("xret_")
panel = pd.concat([raw_ret, excess_ret], axis=1)
panel.insert(0, "rf_daily", rf_daily)
panel = panel.sort_index()
panel.index.name = "date"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
panel.to_csv(OUT_FILE)
