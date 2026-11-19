"""Build a leakage-safe monthly panel of FRED macro predictors.

Anti-look-ahead contract: every series is indexed by `alignment_date`, the
date the value became *publicly known* (its release/vintage date), never by
`date`, the reference period the value describes. For forecast-formation
month t, `aggregate_monthly` resolves each series to whichever release was
the most recent one with alignment_date <= end of month t -- regardless of
which reference period that release describes (e.g. a September CPI print
released in October counts toward October, not September).
`validate_release_alignment` re-derives that same value independently from
raw release dates and asserts it matches, so a future edit that
accidentally indexes by reference period instead of release date will fail
loudly rather than silently leaking future information into month t.
"""

import argparse
import datetime as dt
import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
CFNAI_REALTIME_URLS = [
    "https://www.chicagofed.org/-/media/publications/cfnai/cfnai-realtime-1-xlsx.xlsx",
    "https://www.chicagofed.org/-/media/publications/cfnai/cfnai-realtime-2-xlsx.xlsx",
    "https://www.chicagofed.org/-/media/publications/cfnai/cfnai-realtime-3-xlsx.xlsx",
]

START = dt.date(2000, 1, 1)
END = dt.date(2026, 7, 31)
REALTIME_CHUNK_YEARS = 5
TRAILING_WINDOW_TRADING_DAYS = 5

SOURCE_FRED = "fred"
SOURCE_ALFRED = "alfred"
SOURCE_CHICAGO_FED_REALTIME = "chicago_fed_realtime"
SOURCE_DROP = "drop"

ALIGN_OBSERVATION_DATE = "observation_date"
ALIGN_RELEASE_DATE = "release_date"

AGG_TRAILING_MEAN = "trailing_mean"
AGG_LATEST_BY_MONTH_END = "latest_by_month_end"

OUTPUT_DIR = Path(__file__).resolve().parent
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")
CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
RAW_COLUMNS = ["date", "realtime_start", "value", "alignment_date"]

FINAL_PANEL_COLUMNS = [
    "T10Y2Y",
    "DGS3MO",
    "BAA10Y",
    "CPIAUCSL",
    "PCEPI",
    "T5YIE",
    "T10YIE",
    "M2SL",
    "VIXCLS",
    "INDPRO",
    "CFNAI",
    "UNRATE",
    "UMCSENT",
    "DCOILWTICO",
]

SERIES_CONFIG = {
    "T10Y2Y": {
        "category": "rates_yield_curve",
        "frequency": "daily",
        "description": "10Y minus 2Y Treasury spread",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "DGS3MO": {
        "category": "rates_yield_curve",
        "frequency": "daily",
        "description": "3-month Treasury rate",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "BAMLH0A0HYM2": {
        "category": "credit_spreads",
        "frequency": "daily",
        "description": "ICE BofA US High Yield OAS",
        "source": SOURCE_DROP,
        "drop_reason": "Dropped due to insufficient historical coverage.",
    },
    "BAA10Y": {
        "category": "credit_spreads",
        "frequency": "daily",
        "description": "Moody's Baa corporate yield minus 10Y Treasury",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "CPIAUCSL": {
        "category": "inflation",
        "frequency": "monthly",
        "description": "CPI",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "PCEPI": {
        "category": "inflation",
        "frequency": "monthly",
        "description": "PCE Price Index",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "T5YIE": {
        "category": "inflation",
        "frequency": "daily",
        "description": "5-year breakeven inflation",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "T10YIE": {
        "category": "inflation",
        "frequency": "daily",
        "description": "10-year breakeven inflation",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "M2SL": {
        "category": "inflation",
        "frequency": "monthly",
        "description": "M2 money stock",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "VIXCLS": {
        "category": "volatility_stress",
        "frequency": "daily",
        "description": "CBOE VIX",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
    "NFCI": {
        "category": "volatility_stress",
        "frequency": "weekly",
        "description": "Chicago Fed Financial Conditions Index",
        "source": SOURCE_DROP,
        "drop_reason": "Dropped: no ALFRED vintage data available from 2000 onwards.",
    },
    "STLFSI4": {
        "category": "volatility_stress",
        "frequency": "weekly",
        "description": "St. Louis Fed Financial Stress Index",
        "source": SOURCE_DROP,
        "drop_reason": "Dropped: no ALFRED vintage data available from 2000 onwards.",
    },
    "INDPRO": {
        "category": "real_activity_labour",
        "frequency": "monthly",
        "description": "Industrial Production",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "CFNAI": {
        "category": "real_activity_labour",
        "frequency": "monthly",
        "description": "Chicago Fed National Activity Index",
        "source": SOURCE_CHICAGO_FED_REALTIME,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "UNRATE": {
        "category": "real_activity_labour",
        "frequency": "monthly",
        "description": "Unemployment Rate",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "ICSA": {
        "category": "real_activity_labour",
        "frequency": "weekly",
        "description": "Initial jobless claims",
        "source": SOURCE_DROP,
        "drop_reason": "Dropped: no ALFRED vintage data available from 2000 onwards.",
    },
    "UMCSENT": {
        "category": "real_activity_labour",
        "frequency": "monthly",
        "description": "University of Michigan Consumer Sentiment",
        "source": SOURCE_ALFRED,
        "align_on": ALIGN_RELEASE_DATE,
        "aggregation": AGG_LATEST_BY_MONTH_END,
    },
    "DTWEXBGS": {
        "category": "dollar_commodity",
        "frequency": "daily",
        "description": "Trade Weighted U.S. Dollar Index",
        "source": SOURCE_DROP,
        "drop_reason": "Dropped: no ALFRED vintage data available from 2000 onwards.",
    },
    "NBUSBIS": {
        "category": "dollar_commodity",
        "frequency": "monthly",
        "description": "Nominal Broad U.S. Dollar Index",
        "source": SOURCE_DROP,
        'drop_reason': "Dropped: no ALFRED vintage data available from 2000 onwards.",
    },
    "DCOILWTICO": {
        "category": "dollar_commodity",
        "frequency": "daily",
        "description": "WTI crude oil spot price",
        "source": SOURCE_FRED,
        "align_on": ALIGN_OBSERVATION_DATE,
        "aggregation": AGG_TRAILING_MEAN,
    },
}


def get_fred_api_key():
    load_dotenv(OUTPUT_DIR.parents[1] / ".env")
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing FRED_API_KEY. Set it in your environment before running this script."
        )
    return api_key


def iter_realtime_windows(start, end, years_per_chunk):
    current_start = start
    while current_start <= end:
        current_end = min(
            dt.date(current_start.year + years_per_chunk - 1, 12, 31),
            end,
        )
        yield current_start, current_end
        current_start = current_end + dt.timedelta(days=1)


def request_fred_observations(series_id, api_key, params):
    response = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
    try:
        payload = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError(f"FRED API returned a non-JSON response for {series_id}.")

    if response.status_code >= 400 or "error_message" in payload:
        message = payload.get("error_message", response.text)
        raise RuntimeError(f"FRED API error for {series_id}: {message}")

    return pd.DataFrame(payload["observations"])


def fetch_fred_series(series_id, api_key):
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "observation_start": START.isoformat(),
        "observation_end": END.isoformat(),
        "sort_order": "asc",
    }
    return request_fred_observations(series_id, api_key, params)


def fetch_alfred_first_release_chunk(series_id, api_key, realtime_start, realtime_end):
    params = {
        "api_key": api_key,
        "file_type": "json",
        "series_id": series_id,
        "output_type": 4,
        "realtime_start": realtime_start.isoformat(),
        "realtime_end": realtime_end.isoformat(),
        "observation_start": START.isoformat(),
        "observation_end": END.isoformat(),
        "sort_order": "asc",
    }
    return request_fred_observations(series_id, api_key, params)


def fetch_alfred_first_release(series_id, api_key):
    chunks = []
    for realtime_start, realtime_end in iter_realtime_windows(
        START, END, REALTIME_CHUNK_YEARS
    ):
        chunk = fetch_alfred_first_release_chunk(
            series_id, api_key, realtime_start, realtime_end
        )
        chunks.append(chunk)
    raw = pd.concat(chunks, ignore_index=True)
    if raw.empty:
        raise RuntimeError(f"No ALFRED first-release rows returned for {series_id}.")
    return raw


def read_chicago_fed_excel(url):
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=60,
    )
    response.raise_for_status()
    return pd.ExcelFile(BytesIO(response.content))


def fetch_cfnai_chicago_realtime():
    rows = []
    for url in CFNAI_REALTIME_URLS:
        workbook = read_chicago_fed_excel(url)
        realtime = pd.read_excel(workbook, sheet_name="cfnai_realtime")
        release_dates = pd.read_excel(workbook, sheet_name="release_dates", header=None)

        date_column = realtime.columns[0]
        realtime[date_column] = pd.to_datetime(
            realtime[date_column], errors="coerce", format="mixed"
        )
        realtime_months = realtime[date_column].dt.to_period("M")

        release_dates = release_dates.rename(
            columns={0: "release_date", 1: "reference_month"}
        )
        release_dates["release_date"] = pd.to_datetime(
            release_dates["release_date"], errors="coerce", format="mixed"
        )
        release_dates["reference_month"] = pd.to_datetime(
            release_dates["reference_month"], errors="coerce", format="mixed"
        )
        release_dates = release_dates.dropna(
            subset=["release_date", "reference_month"]
        )

        for _, release in release_dates.iterrows():
            reference_month = release["reference_month"]
            release_date = release["release_date"]
            value_column = f"CF{reference_month.month}{reference_month.year}"
            if value_column not in realtime.columns:
                continue

            value = realtime.loc[
                realtime_months == reference_month.to_period("M"), value_column
            ]
            if value.empty:
                continue

            rows.append(
                {
                    "date": reference_month,
                    "realtime_start": release_date,
                    "value": value.iloc[0],
                    "alignment_date": release_date,
                }
            )

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No CFNAI real-time archive rows were parsed.")
    return raw


def clean_raw_observations(raw):
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    raw["realtime_start"] = pd.to_datetime(raw["realtime_start"])
    raw["value"] = pd.to_numeric(raw["value"].replace(".", pd.NA), errors="coerce")
    if "alignment_date" in raw.columns:
        raw["alignment_date"] = pd.to_datetime(raw["alignment_date"])
    return raw.drop_duplicates(["date", "realtime_start"]).sort_values(
        ["date", "realtime_start"]
    )


def add_alignment_date(raw, cfg):
    """Stamp each row with the date its value became publicly known.

    ALIGN_RELEASE_DATE: alignment_date = realtime_start, the true ALFRED/
    Chicago Fed vintage release date. `date` (the reference period) is kept
    only for display/tie-breaking and never drives bucketing -- this is
    what makes AGG_LATEST_BY_MONTH_END leakage-safe.

    ALIGN_OBSERVATION_DATE: alignment_date = date. Used only for plain-FRED
    daily market quotes (Treasury yields, VIX, WTI spot) that are published
    same-day with no revision history, so observation date == release date.
    This equivalence is an assumption, not something this script proves --
    unlike ALIGN_RELEASE_DATE series, these are NOT checked by
    validate_release_alignment (it returns immediately for this branch).
    """
    raw = raw.copy()
    if cfg["align_on"] == ALIGN_RELEASE_DATE:
        raw["alignment_date"] = raw["realtime_start"]
    elif cfg["align_on"] == ALIGN_OBSERVATION_DATE:
        raw["alignment_date"] = raw["date"]
    else:
        raise ValueError(f"Unknown alignment rule: {cfg['align_on']}")
    return raw


def align_series(raw, cfg):
    raw = clean_raw_observations(raw)
    raw = add_alignment_date(raw, cfg)
    # Filter by alignment_date (publication time), not `date` (reference
    # period) -- a release that lands outside [START, END] is excluded even
    # if the observation it reports on falls inside the window.
    raw = raw[
        (raw["alignment_date"] >= pd.Timestamp(START))
        & (raw["alignment_date"] <= pd.Timestamp(END))
    ].copy()
    return raw[RAW_COLUMNS].sort_values(["alignment_date", "date"])


def month_end_index():
    return pd.date_range(START, END, freq="ME")


def aggregate_monthly(raw, cfg):
    """Collapse aligned observations to one value per calendar month-end.

    All branches key off `alignment_date` (release date), never `date`
    (reference period) -- that is the entire look-ahead guard.

    AGG_TRAILING_MEAN: mean of the last TRAILING_WINDOW_TRADING_DAYS
    published values within the calendar month (daily market series with
    no release lag, e.g. VIX, Treasury yields). A short trailing window at
    month-end reflects "state at decision time" much more closely than a
    full-month average, while still damping a single anomalous print --
    unlike a full-month mean, it never mixes in data more than ~1 trading
    week stale. Only ever looks backward within the month, so it adds no
    look-ahead risk versus the plain full-month mean it replaced.

    AGG_LATEST_BY_MONTH_END: for month t, take the most recent value
    released within month t's calendar bucket, then forward-fill through
    any month with no new release. This implements "value for month t =
    the release most recently public as of end of month t, regardless of
    which reference period it describes" -- e.g. September's row can carry
    an August CPI print (if September's hasn't been released yet by
    month-end) but never a not-yet-released September or later print.
    """
    months = month_end_index()

    if cfg["aggregation"] == AGG_TRAILING_MEAN:
        # per-series window (e.g. {"window": 20} for USEPUINDXD), else the default 5
        window_size = cfg.get("window", TRAILING_WINDOW_TRADING_DAYS)

        ordered = raw.dropna(subset=["value"]).sort_values(["alignment_date", "date"]).copy()
        ordered["month_end"] = ordered["alignment_date"] + pd.offsets.MonthEnd(0)
        monthly = ordered.groupby("month_end")["value"].apply(
            lambda group: group.tail(window_size).mean()
        )
        return monthly.reindex(months)

    if cfg["aggregation"] == AGG_LATEST_BY_MONTH_END:
        ordered = raw.sort_values(["alignment_date", "date"])
        s = ordered.set_index("alignment_date")["value"]
        return s.resample("ME").last().ffill().reindex(months)

    raise ValueError(f"Unknown aggregation rule: {cfg['aggregation']}")


def fetch_raw_series(series_id, cfg, api_key):
    if cfg["source"] == SOURCE_FRED:
        return fetch_fred_series(series_id, api_key)
    if cfg["source"] == SOURCE_ALFRED:
        return fetch_alfred_first_release(series_id, api_key)
    if cfg["source"] == SOURCE_CHICAGO_FED_REALTIME:
        return fetch_cfnai_chicago_realtime()
    raise ValueError(f"Unknown retained source for {series_id}: {cfg['source']}")


def validate_no_manual_shift():
    with open(__file__, "r", encoding="utf-8") as script_file:
        source = script_file.read()
    manual_shift_call = "." + "shift("
    assert manual_shift_call not in source, "Manual shift call found in downloader."


def validate_release_alignment(series_id, raw, monthly, cfg):
    """Prove aggregate_monthly's AGG_LATEST_BY_MONTH_END output has no look-ahead.

    Independently re-derives, straight from raw release dates, "the most
    recent release with release_date <= month_end, regardless of reference
    period" -- your stated correctness rule -- and asserts it equals what
    the production resample/ffill path produced. Only runs for
    ALIGN_RELEASE_DATE series, since only those carry a real release date;
    see add_alignment_date's docstring for the ALIGN_OBSERVATION_DATE
    caveat.
    """
    if cfg["align_on"] != ALIGN_RELEASE_DATE:
        return

    assert (
        raw["alignment_date"] >= raw["realtime_start"]
    ).all(), f"{series_id} uses observations before release date."

    ordered = raw.sort_values(["alignment_date", "date"]).dropna(subset=["value"])
    for month_end, value in monthly.dropna().items():
        available = ordered[ordered["alignment_date"] <= month_end]
        assert not available.empty, f"{series_id} has a value before first release."
        expected = available.iloc[-1]["value"]
        assert value == expected, (
            f"{series_id} month-end {month_end.date()} is not the latest "
            "reference observation released by that date."
        )


def validate_final_panel(full_panel):
    validate_no_manual_shift()
    dropped = [sid for sid, cfg in SERIES_CONFIG.items() if cfg["source"] == SOURCE_DROP]
    assert list(full_panel.columns) == FINAL_PANEL_COLUMNS, (
        "Final panel columns do not match the required predictors."
    )
    assert not set(dropped).intersection(full_panel.columns), (
        "Dropped variables are present in the final panel."
    )
    assert full_panel.index.min() == month_end_index().min()
    assert full_panel.index.max() == month_end_index().max()


def write_dropped_series():
    dropped_rows = []
    for series_id, cfg in SERIES_CONFIG.items():
        if cfg["source"] == SOURCE_DROP:
            dropped_rows.append(
                {
                    "series_id": series_id,
                    "category": cfg["category"],
                    "frequency": cfg["frequency"],
                    "description": cfg["description"],
                    "reason": cfg["drop_reason"],
                }
            )
    dropped = pd.DataFrame(dropped_rows)
    dropped.to_csv(os.path.join(CLEAN_DIR, "dropped_series.csv"), index=False)
    return dropped


def remove_stale_dropped_raw_files(dropped):
    for series_id in dropped["series_id"]:
        path = os.path.join(RAW_DIR, f"{series_id}.csv")
        if os.path.exists(path):
            os.remove(path)


def print_summary(full_panel, dropped):
    print("\n=== Final macro predictor summary ===")
    print(f"Date range: {START.isoformat()} to {END.isoformat()}")
    print(f"Retained variables ({len(FINAL_PANEL_COLUMNS)}): {FINAL_PANEL_COLUMNS}")
    print(f"Dropped variables ({len(dropped)}): {list(dropped['series_id'])}")
    print("\nSeries coverage:")
    for series_id in FINAL_PANEL_COLUMNS:
        s = full_panel[series_id]
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        print(
            f"{series_id:12s} "
            f"first={first_valid.date() if first_valid is not None else 'NA'} "
            f"last={last_valid.date() if last_valid is not None else 'NA'} "
            f"missing={int(s.isna().sum())}"
        )


def main(from_raw=False):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CLEAN_DIR, exist_ok=True)

    api_key = None if from_raw else get_fred_api_key()
    dropped = write_dropped_series()
    if not from_raw:
        remove_stale_dropped_raw_files(dropped)
    all_monthly_data = {}

    for series_id, cfg in SERIES_CONFIG.items():
        if cfg["source"] == SOURCE_DROP:
            print(f"Dropping {series_id}: {cfg['drop_reason']}")
            continue

        print(
            f"{'Loading' if from_raw else 'Fetching'} {series_id} "
            f"({cfg['category']}, {cfg['frequency']}, source={cfg['source']}) ..."
        )
        raw_path = os.path.join(RAW_DIR, f"{series_id}.csv")
        raw = pd.read_csv(raw_path) if from_raw else fetch_raw_series(series_id, cfg, api_key)
        raw = align_series(raw, cfg)
        if not from_raw:
            raw.to_csv(raw_path, index=False)

        monthly = aggregate_monthly(raw, cfg)
        validate_release_alignment(series_id, raw, monthly, cfg)

        all_monthly_data[series_id] = monthly

    full_panel = pd.DataFrame(all_monthly_data).reindex(month_end_index())
    full_panel = full_panel[FINAL_PANEL_COLUMNS]
    full_panel.index.name = "date"
    validate_final_panel(full_panel)

    full_panel.to_csv(os.path.join(CLEAN_DIR, "fred_monthly_panel.csv"))
    print(f"Saved fred_monthly_panel.csv with all {len(full_panel.columns)} series")
    print_summary(full_panel, dropped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-raw", action="store_true",
                        help="Rebuild from existing raw CSVs without downloading or an API key.")
    try:
        main(from_raw=parser.parse_args().from_raw)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}")
