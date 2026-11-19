"""Narrow the leakage-safe FRED monthly panel to the final macro predictor set.

Starting point: clean/fred_monthly_panel.csv, produced by fetch_fred_data.py.
That script already drops series with no ALFRED/vintage history from 2000
onwards (NFCI, STLFSI4, ICSA, DTWEXBGS). This script performs a second,
economics-driven narrowing pass -- reducing collinearity within each FRED
channel. The representations and rationale are documented below.

All predictors, including oil, come from the monthly panel produced by
fetch_fred_data.py. Run that script before this one to refresh the API inputs.

Kept (11 variables, 6 channels):
    Rates/yield curve    T10Y2Y, DGS3MO            -- level vs slope, not collinear
    Credit spreads       BAA10Y                     -- single series
    Inflation            CPIAUCSL, T5YIE,
                         T5YIE - T10YIE (engineered)
    Vol/stress           VIXCLS                     -- single series
    Real activity        INDPRO, UNRATE, UMCSENT
    Commodity            DCOILWTICO

Dropped (4 variables):
    PCEPI  -- redundant with CPIAUCSL (corr > 0.9); slower to publish and
              partly derived from CPI itself, so structurally laggier
    T10YIE -- redundant with T5YIE at the level; retained only long enough
              to engineer the T5YIE - T10YIE curve-stress spread, then dropped
    M2SL   -- inflation channel already covered by the rates/yield curve
              channel; M2 growth has weak/noisy short-horizon predictive power
    CFNAI  -- mechanically double-counts INDPRO/UNRATE/UMCSENT (it's a
              composite built from ~85 series including them) and carries a
              3-4 week publication lag, the worst of any real-activity candidate
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path(__file__).resolve().parent
PANEL_PATH = DATA_DIR / "clean/fred_monthly_panel.csv"
OUTPUT_DIR = DATA_DIR / "final_clean"
OUTPUT_PATH = OUTPUT_DIR / "final_macro_data.csv"
ADF_OUTPUT_PATH = OUTPUT_DIR / "adf_test_results.csv"
ADF_SIGNIFICANCE = 0.05

DROP_NOW = ["PCEPI", "M2SL", "CFNAI"]
LOG_DIFF_VARS = ["CPIAUCSL", "INDPRO", "DCOILWTICO"]
START_DATE = pd.Timestamp("2003-01-31") # to follow the first month-end with T5YIE/T10YIE vintage data

FINAL_COLUMNS = [
    "T10Y2Y",
    "DGS3MO",
    "BAA10Y",
    "CPIAUCSL",
    "T5YIE",
    "T5YIE_MINUS_T10YIE",
    "VIXCLS",
    "INDPRO",
    "UNRATE",
    "UMCSENT",
    "DCOILWTICO",
]


def load_panel():
    panel = pd.read_csv(PANEL_PATH, index_col="date", parse_dates=True).sort_index()
    if panel.empty or panel.index.hasnans or panel.index.has_duplicates:
        raise ValueError("Monthly macro dates must be present and unique.")
    expected = pd.date_range(panel.index.min(), panel.index.max(), freq="ME")
    if not panel.index.equals(expected):
        raise ValueError("Macro panel must contain consecutive calendar month ends.")
    return panel


def drop_redundant_channels(panel):
    """Step 1: drop variables that are redundant within their channel.

    PCEPI, M2SL, and CFNAI are dropped outright (see module docstring).
    T10YIE is kept one more step, since it's needed to engineer the
    T5YIE - T10YIE inflation-curve spread before it's dropped.
    """
    return panel.drop(columns=DROP_NOW).copy()


def engineer_inflation_curve_spread(df):
    """Step 2: T5YIE - T10YIE, then drop T10YIE.

    Near-term minus long-term breakeven inflation expectations. An
    inverting spread (short-term jumping above long-term) signals an
    acute, near-term inflation shock (e.g. an energy spike) outrunning
    anchored long-run expectations -- a distinct signal from the level of
    T5YIE alone. T10YIE is dropped immediately after, since its only role
    here was constructing this spread.
    """
    df = df.copy()
    df["T5YIE_MINUS_T10YIE"] = df["T5YIE"] - df["T10YIE"]
    return df.drop(columns=["T10YIE"])


def apply_transformations(df):
    """Step 3: log-difference the asset-price-like / index-level series.

    Everything else (spreads, rates, VIX, UNRATE, UMCSENT) stays as a
    level -- see the module docstring for the rationale per variable. Log-differences are computed on the full history (before
    slicing to START_DATE) so the first usable month doesn't lose its
    diff to a missing prior observation.
    """
    df = df.copy()
    for col in LOG_DIFF_VARS:
        if (df[col].dropna() <= 0).any():
            raise ValueError(f"{col} must be positive before taking logarithms.")
        df[col] = np.log(df[col]).diff()
    return df


def slice_to_start_date(df):
    """Step 4: slice to 2003-01-31, the first month-end with T5YIE/T10YIE vintage data."""
    final = df.loc[df.index >= START_DATE].copy()

    if not np.isfinite(final[FINAL_COLUMNS].to_numpy()).all():
        raise ValueError(
            "Final predictors contain missing or non-finite values. "
            "Check the fetched monthly panel coverage."
        )

    print("No missing values in the final panel.")
    return final


def save_final_panel(final, output_dir=OUTPUT_DIR):
    final = final[FINAL_COLUMNS]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "final_macro_data.csv"
    final.to_csv(output_path)
    print(f"Saved {output_path} with columns: {list(final.columns)}")
    return final


def run_adf_tests(final, output_dir=OUTPUT_DIR):
    """Step 6: ADF stationarity check on every final variable.

    This is diagnostic, not a second pass at picking representations --
    We deliberately keep several variables (T10Y2Y, DGS3MO,
    T5YIE, T5YIE_MINUS_T10YIE) as levels even though they may read as
    non-stationary, because the level *is* the economically meaningful
    object (see apply_transformations' docstring). A failure to reject the null
    documents that regime-like persistence, it isn't a signal to
    re-difference those series.
    """
    rows = []
    for col in FINAL_COLUMNS:
        series = final[col].dropna()
        statistic, p_value, *_ = adfuller(series, autolag="AIC")
        rows.append(
            {
                "variable": col,
                "adf_statistic": statistic,
                "p_value": p_value,
                "n_obs": len(series),
                "stationary_at_5pct": p_value < ADF_SIGNIFICANCE,
            }
        )

    results = pd.DataFrame(rows).set_index("variable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adf_output_path = output_dir / "adf_test_results.csv"
    results.to_csv(adf_output_path)
    print(f"\nSaved {adf_output_path}")
    print("\n=== ADF stationarity test (H0: unit root / non-stationary) ===")
    print(results)
    return results


def main(output_dir=OUTPUT_DIR):
    panel = load_panel()
    df = drop_redundant_channels(panel)
    df = engineer_inflation_curve_spread(df)
    df = apply_transformations(df)
    final = slice_to_start_date(df)
    final = save_final_panel(final, output_dir)
    print(final.describe())
    run_adf_tests(final, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Directory for the macro panel and ADF results (default: final_clean).",
    )
    main(parser.parse_args().output_dir)
