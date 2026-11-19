"""Crisis windows for the robustness checks (used by r1 and r2).

The windows are the NBER recessions. A forecast is dropped when the
month it forecasts (`target_date`) falls inside a window. Only the scored months change and nothing is
refitted.
"""
import pandas as pd

CRISES = {
    "GFC": ("2007-12", "2009-06"),    # first forecast is March 2008, so 2008-03 to 2009-06 is in sample
    "COVID": ("2020-02", "2020-04"),
}

# sample name : crises removed from the evaluation
SAMPLES = {
    "full": [],
    "ex_GFC": ["GFC"],
    "ex_COVID": ["COVID"],
    "ex_both": ["GFC", "COVID"],
}


def exclude(df, sample, date_col="target_date"):
    """Rows of `df` whose forecast month is outside the crises removed in `sample`."""
    month = pd.to_datetime(df[date_col]).dt.to_period("M")
    keep = pd.Series(True, index=df.index)
    for name in SAMPLES[sample]:
        start, end = CRISES[name]
        keep &= ~((month >= pd.Period(start)) & (month <= pd.Period(end)))
    return df[keep]
