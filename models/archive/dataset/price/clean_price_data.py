"""
Clean CRSP daily sector ETF prices + risk-free rate into a daily panel of
raw and excess returns. Produces a data-quality report only — no realized
variance, no macro merge, no portfolio weights (handled downstream).

Inputs:
    raw/sector_etf_prices.csv       CRSP daily export (via WRDS)
    raw/risk_free_rate_returns.csv  daily risk-free rate series

Output:
    clean/daily_returns_panel.csv
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
CLEAN_DIR = Path(__file__).resolve().parent / "clean"

PRICE_FILE = f"{RAW_DIR}/sector_etf_prices.csv"
RF_FILE = f"{RAW_DIR}/risk_free_rate_returns.csv"
OUT_FILE = f"{CLEAN_DIR}/daily_returns_panel.csv"

EXPECTED_TICKERS = ["XLI", "XLB", "XLE", "XLY", "XLF", "XLRE", "XLU", "XLP", "XLV"]

# Output panel is truncated to start here 
SAMPLE_START_DATE = "2003-01-31"

CRSP_SENTINELS = [-66, -77, -88, -99]
IMPLAUSIBLE_ABS_RET_THRESHOLD = 0.5

# Assumption: risk_free_rate_returns.csv is already a daily decimal return
# (values like 0.0002), consistent with the price panel's return convention.
# No annualized -> daily conversion is applied.


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# PRICE DATA CLEANING
# ---------------------------------------------------------------------------
section("LOADING PRICE DATA")

price_raw = pd.read_csv(PRICE_FILE)
print(f"Loaded {len(price_raw):,} rows from {PRICE_FILE}")
print(f"Columns available: {list(price_raw.columns)}")

wanted_cols = ["PERMNO", "Ticker", "DlyCalDt", "DlyRet", "DlyVol"]
missing_cols = [c for c in wanted_cols if c not in price_raw.columns]
if missing_cols:
    print(f"NOTE: expected columns not present in file, skipping: {missing_cols}")
subset_cols = [c for c in wanted_cols if c in price_raw.columns]
price = price_raw[subset_cols].copy()
price["DlyCalDt"] = pd.to_datetime(price["DlyCalDt"])

tickers_present = sorted(price["Ticker"].unique())
missing_tickers = sorted(set(EXPECTED_TICKERS) - set(tickers_present))
extra_tickers = sorted(set(tickers_present) - set(EXPECTED_TICKERS))
print(f"\nTickers present in data ({len(tickers_present)}): {tickers_present}")
if missing_tickers:
    print(f"WARNING: expected tickers NOT found in the file at all: {missing_tickers}")
if extra_tickers:
    print(f"WARNING: unexpected tickers found: {extra_tickers}")

# --- PERMNO <-> Ticker stability check ---
section("PERMNO <-> TICKER STABILITY CHECK")

permnos_per_ticker = price.groupby("Ticker")["PERMNO"].nunique()
tickers_with_multiple_permnos = permnos_per_ticker[permnos_per_ticker > 1]
if len(tickers_with_multiple_permnos) > 0:
    print("FLAG: tickers mapping to more than one PERMNO:")
    for tkr in tickers_with_multiple_permnos.index:
        sub = price[price["Ticker"] == tkr]
        for permno, grp in sub.groupby("PERMNO"):
            print(f"  {tkr} -> PERMNO {permno}: {grp['DlyCalDt'].min().date()} to {grp['DlyCalDt'].max().date()} ({len(grp)} rows)")
else:
    print("OK: every ticker maps to exactly one PERMNO.")

tickers_per_permno = price.groupby("PERMNO")["Ticker"].nunique()
permnos_with_multiple_tickers = tickers_per_permno[tickers_per_permno > 1]
if len(permnos_with_multiple_tickers) > 0:
    print("FLAG: PERMNOs mapping to more than one ticker:")
    for permno in permnos_with_multiple_tickers.index:
        sub = price[price["PERMNO"] == permno]
        for tkr, grp in sub.groupby("Ticker"):
            print(f"  PERMNO {permno} -> {tkr}: {grp['DlyCalDt'].min().date()} to {grp['DlyCalDt'].max().date()} ({len(grp)} rows)")
else:
    print("OK: every PERMNO maps to exactly one ticker.")

# --- Coverage per ticker ---
section("COVERAGE PER TICKER (first/last available date)")

coverage = price.groupby("Ticker")["DlyCalDt"].agg(first_date="min", last_date="max", n_obs="count")
coverage = coverage.reindex(tickers_present)
print(coverage.to_string())
print("\nNo dates are dropped or truncated automatically -- use the above to decide")
print("how to handle the unbalanced panel (e.g. XLRE's later inception).")

# --- Missing values / sentinel codes / implausible returns ---
section("MISSING VALUES, CRSP SENTINEL CODES, AND IMPLAUSIBLE RETURNS")

n_missing_ret = price["DlyRet"].isna().sum()
print(f"Rows with missing DlyRet: {n_missing_ret}")
if n_missing_ret > 0:
    print(price.loc[price["DlyRet"].isna(), "Ticker"].value_counts().to_string())

print("\nCRSP sentinel code counts per ticker (DlyRet in {-66, -77, -88, -99}):")
sentinel_mask = price["DlyRet"].isin(CRSP_SENTINELS)
if sentinel_mask.any():
    print(price.loc[sentinel_mask].groupby("Ticker")["DlyRet"].value_counts().to_string())
else:
    print("None found.")

print(f"\nImplausible DlyRet values (== -1, or |DlyRet| > {IMPLAUSIBLE_ABS_RET_THRESHOLD}):")
implausible_mask = (price["DlyRet"] == -1) | (price["DlyRet"].abs() > IMPLAUSIBLE_ABS_RET_THRESHOLD)
implausible = price.loc[implausible_mask].sort_values("DlyCalDt")
if len(implausible) > 0:
    print(implausible.to_string(index=False))
else:
    print("None found.")

# --- Pivot to wide daily panel ---
section("PIVOT TO WIDE DAILY PANEL (outer join across tickers)")

ret_wide = price.pivot_table(index="DlyCalDt", columns="Ticker", values="DlyRet", aggfunc="first")
vol_wide = price.pivot_table(index="DlyCalDt", columns="Ticker", values="DlyVol", aggfunc="first")
ret_wide = ret_wide.sort_index()
vol_wide = vol_wide.sort_index()

print(f"Wide panel shape (outer join on trading dates): {ret_wide.shape[0]} dates x {ret_wide.shape[1]} tickers")
print("\nMissing-date counts per ticker after outer join (dates where OTHER tickers")
print("traded but this ticker has no return -- expected for ETFs with later")
print("inception, e.g. XLRE):")
print(ret_wide.isna().sum().to_string())

print("\nPairwise mismatched-date counts (dates where exactly one of the pair is")
print("missing a return -- large counts typically reflect different inception dates):")
pair_report = []
for i, t1 in enumerate(ret_wide.columns):
    for t2 in ret_wide.columns[i + 1:]:
        mismatch = ((ret_wide[t1].isna()) != (ret_wide[t2].isna())).sum()
        pair_report.append((t1, t2, mismatch))
pair_df = pd.DataFrame(pair_report, columns=["ticker_1", "ticker_2", "n_mismatched_dates"])
pair_df = pair_df.sort_values("n_mismatched_dates", ascending=False)
print(pair_df.to_string(index=False))

print("\nNo forward-fill or row-dropping has been applied -- the wide panel above")
print("keeps NaN for any ticker/date combination with no observed return.")

# --- Truncate to the requested sample start date ---
section(f"TRUNCATING TO SAMPLE_START_DATE = {SAMPLE_START_DATE}")

n_before = len(ret_wide)
ret_wide = ret_wide.loc[SAMPLE_START_DATE:]
vol_wide = vol_wide.loc[SAMPLE_START_DATE:]
n_after = len(ret_wide)
print(f"Dropped {n_before - n_after} dates before {SAMPLE_START_DATE} ({n_before} -> {n_after} rows).")

# ---------------------------------------------------------------------------
# RISK-FREE RATE MERGE
# ---------------------------------------------------------------------------
section("LOADING RISK-FREE RATE DATA")

rf_raw = pd.read_csv(RF_FILE)
print(f"Loaded {len(rf_raw):,} rows from {RF_FILE}")
print(f"Columns: {list(rf_raw.columns)}")

date_col = "date" if "date" in rf_raw.columns else rf_raw.columns[0]
rate_col = "rf" if "rf" in rf_raw.columns else rf_raw.columns[1]

rf = rf_raw[[date_col, rate_col]].rename(columns={date_col: "date", rate_col: "rf_raw"})
rf["date"] = pd.to_datetime(rf["date"])

print("\nSample rows:")
print(rf.head(10).to_string(index=False))
print("\nSummary statistics of rf_raw:")
print(rf["rf_raw"].describe().to_string())

rf["rf_daily"] = rf["rf_raw"]

# --- Merge onto price panel: left join on price panel's trading dates ---
section("MERGING RISK-FREE RATE ONTO PRICE PANEL")

price_dates = pd.DataFrame({"date": ret_wide.index})
rf_merged = price_dates.merge(rf[["date", "rf_daily"]], on="date", how="left")
rf_merged = rf_merged.set_index("date")

n_missing_rf = rf_merged["rf_daily"].isna().sum()
print(f"Trading dates in price panel: {len(rf_merged)}")
print(f"Trading dates with NO matching rf rate after merge: {n_missing_rf}")
if n_missing_rf > 0:
    missing_rf_dates = rf_merged.loc[rf_merged["rf_daily"].isna()].index
    print("Dates missing rf (first 20 shown):")
    print(list(missing_rf_dates[:20].strftime("%Y-%m-%d")))

# ---------------------------------------------------------------------------
# OUTPUT: raw + excess return panel
# ---------------------------------------------------------------------------
section("BUILDING OUTPUT PANEL")

rf_daily_series = rf_merged["rf_daily"]

raw_ret = ret_wide.add_prefix("ret_")
excess_ret = ret_wide.sub(rf_daily_series, axis=0).add_prefix("xret_")

panel = pd.concat([raw_ret, excess_ret], axis=1)
panel.insert(0, "rf_daily", rf_daily_series)
panel = panel.sort_index()
panel.index.name = "date"

import os
os.makedirs(CLEAN_DIR, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
panel.to_csv(OUT_FILE)

section("FINAL SUMMARY")
print(f"Output written to: {OUT_FILE}")
print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")
print(f"Number of tickers: {len(tickers_present)} -> {tickers_present}")
print(f"Number of rows: {len(panel)}")
print("\nRemaining missingness by column:")
print(panel.isna().sum().to_string())
