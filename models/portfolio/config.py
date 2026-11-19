"""Portfolio settings shared by the notebooks in models/portfolio.

The ETFs, sample, models and comparison blocks are in models/research_config.py. Change a setting
in either file, then rerun p1_build_portfolios.ipynb followed by p2 and p3.
"""

FORECAST_COL = "fc_level"  # log-normal corrected level forecast of next month's RV
K_CAP = 2                  # leverage cap on the base weight

# sigma2_bar specifications. "fixed": the 2003-03 to 2008-02 mean, the same every month.
# "expanding": the mean from 2003-03 through month t, for holding month t+1.
SPECS = ["fixed", "expanding"]
MAIN_SPEC = "fixed"        # the main results; "expanding" is the robustness check
