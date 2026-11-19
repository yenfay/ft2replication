"""Forecast files and checks shared by the runner and portfolio notebooks."""

import numpy as np
import pandas as pd

FORECAST_FILES = {
    "Historical": "hist_rw_forecasts.csv",
    "Garch": "garch_forecasts.csv",
    "Elastic Net": "en_forecasts.csv",
    "XGBoost": "xgb_forecasts.csv",
}
ROLLING_FORECAST_FILES = {
    "en_rolling60_forecasts.csv": ["en_no_macro_roll60", "en_vix_only_roll60"],
}


def expected_forecast_months(actuals, train_start, train_months):
    """Holding months after the configured training window through the last RV month."""
    first = pd.Period(train_start, freq="M") + train_months
    last = pd.Timestamp(actuals["date"].max()).to_period("M")
    months = pd.period_range(first, last, freq="M").to_timestamp("M")
    if months.empty:
        raise ValueError("Realised variance does not cover the out-of-sample period")
    return months


def validate_forecast_frame(frame, actuals, models, tickers, months, source):
    """Require complete model/ETF/month coverage and actuals matching the RV input."""
    required = {"model", "etf", "date", "target_date", "actual_logRV", "actual_RV",
                "fc_log", "sigma2", "fc_level"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")
    if frame.empty or frame[["model", "etf", "date", "target_date"]].isna().any().any():
        raise ValueError(f"{source}: forecast keys must be present")
    if frame.duplicated(["model", "etf", "date"]).any():
        raise ValueError(f"{source}: duplicate model/ETF/origin keys")
    dates = pd.to_datetime(frame["date"], errors="raise")
    targets = pd.to_datetime(frame["target_date"], errors="raise")
    if not dates.eq(dates.dt.to_period("M").dt.to_timestamp("M")).all():
        raise ValueError(f"{source}: origins must be calendar month ends")
    if not targets.eq(dates + pd.offsets.MonthEnd(1)).all():
        raise ValueError(f"{source}: target_date must be the next month end")
    values = frame[["actual_logRV", "actual_RV", "fc_log", "fc_level"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (frame[["actual_RV", "fc_level"]] <= 0).any().any():
        raise ValueError(f"{source}: realised and forecast values must be finite and positive in levels")
    if np.isinf(frame["sigma2"].to_numpy(dtype=float)).any():
        raise ValueError(f"{source}: sigma2 cannot be infinite")

    reference = actuals.set_index(["etf", "date"])["rv_1m"]
    if not reference.index.is_unique:
        raise ValueError("Realised variance contains duplicate ETF/month keys")
    keys = pd.MultiIndex.from_arrays([frame["etf"], targets], names=["etf", "date"])
    observed = reference.reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError(f"{source}: target months are missing from realised variance or have invalid RV")
    if not np.allclose(frame["actual_RV"], observed, rtol=1e-9, atol=1e-12):
        raise ValueError(f"{source}: actual_RV does not match the current realised-variance input")
    if not np.allclose(frame["actual_logRV"], np.log(observed), rtol=1e-9, atol=1e-12):
        raise ValueError(f"{source}: actual_logRV does not match the current realised-variance input")

    selected = frame["model"].isin(models) & targets.ge(months.min())
    covered = pd.MultiIndex.from_arrays(
        [frame.loc[selected, "model"], frame.loc[selected, "etf"], targets[selected]],
        names=["model", "etf", "target_date"],
    )
    expected = pd.MultiIndex.from_product([models, tickers, months], names=covered.names)
    missing = expected.difference(covered)
    extra = covered.difference(expected)
    if len(missing) or len(extra):
        raise ValueError(f"{source}: incomplete model/ETF/month coverage "
                         f"({len(missing)} missing, {len(extra)} unexpected; "
                         f"first missing: {list(missing[:3])})")
