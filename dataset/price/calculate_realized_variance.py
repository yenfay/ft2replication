
from pathlib import Path
import numpy as np
import pandas as pd

clean_dir = Path(__file__).resolve().parent / "clean"

df = pd.read_csv(
    clean_dir / "daily_returns_panel_ver2.csv",
    parse_dates=["date"]
)

# Daily excess returns for the five cyclical ETFs.
tickers = ["XLI", "XLB", "XLE", "XLY", "XLF"]
ret_cols = [f"xret_{ticker}" for ticker in tickers]

# Monthly realised variance
monthly = (
    df.set_index("date")[ret_cols]
    .pow(2)
    .resample("ME")
    .sum(min_count=1)
)

# Ensure months with missing daily ETF returns are not treated as complete
counts = df.set_index("date")[ret_cols].resample("ME").count()
total_days = df.set_index("date").resample("ME").size()
monthly = monthly.where(counts.eq(total_days, axis=0))

# excluding partial January 2003 data
monthly = monthly.loc[monthly.index != pd.Timestamp("2003-01-31")]

# Reshape into ETF-month format
rv = (
    monthly.rename_axis("date")
    .reset_index()
    .melt(id_vars="date", var_name="etf", value_name="rv_1m")
)

rv["etf"] = rv["etf"].str.replace("xret_", "", regex=False)
rv = rv.sort_values(["etf", "date"])

# HAR-style predictors: 1,3 and 12 month realized variance
rv["log_rv_1m"] = np.log(rv["rv_1m"])

for h in [3, 12]:
    rv[f"log_rv_avg_{h}m"] = (
        rv.groupby("etf")["log_rv_1m"]
        .transform(lambda s: s.rolling(h, min_periods=h).mean())
    )

rv.to_csv(clean_dir / "realized_variance.csv", index=False)
