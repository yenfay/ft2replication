"""Shared code for the volatility-managed portfolio notebooks.

Used by portfolio_construction.ipynb (builds and evaluates the portfolios) and
forecast_vs_portfolio.ipynb (compares forecast accuracy with portfolio performance).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PERIODS_PER_YEAR = 12

# Relative to the notebooks in scripts/portfolio_construction.
# Absolute equivalent: C:\Documents\NUS\Y4S1\DSE4101\dse4101capstone_project
PROJECT_ROOT = Path("../..")
FORECAST_DIR = PROJECT_ROOT / "forecasts"
RETURNS_PATH = PROJECT_ROOT / "dataset" / "price" / "clean" / "daily_returns_panel_ver2.csv"
RESULTS_DIR = Path("results")
PORTFOLIO_DIR = RESULTS_DIR / "portfolios"   # written by 01_build_portfolios
EVAL_DIR = RESULTS_DIR / "eval"              # written by 02, 03 and 04

CONTROL = "control"   # name of the re-levering control portfolio (expanding spec only)

# One tidy forecast file per model family. Columns: model, etf, date, actual_logRV, actual_RV,
# fc_log, sigma2, fc_level. The `model` column holds the variant (e.g. en_full_macro).
FORECAST_FILES = {
    "Garch": "garch_forecasts_expanding_tidy.csv",
    "HAR OLS": "ols_forecasts_expanding_tidy.csv",
    "Elastic Net": "en_forecasts_expanding_tidy.csv",
    "XGBoost": "xgb_forecasts_expanding_tidy.csv",
}

CYCLICAL = ["XLB", "XLE", "XLF", "XLI", "XLY"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_monthly_returns(tickers, path=RETURNS_PATH):
    """Monthly excess returns and realised variance (sum of squared daily returns)."""
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    ret_daily = daily[[f"ret_{t}" for t in tickers]].rename(columns=lambda c: c.removeprefix("ret_"))

    ret_monthly = (1 + ret_daily).resample("ME").prod() - 1
    rf_monthly = (1 + daily["rf_daily"]).resample("ME").prod() - 1
    xret_monthly = ret_monthly.sub(rf_monthly, axis=0)
    rv_monthly = ret_daily.pow(2).resample("ME").sum()
    return xret_monthly, rv_monthly


def training_window(train_start, train_months):
    """Month-end index of the fixed training window and the first out-of-sample month."""
    train_idx = pd.period_range(train_start, periods=train_months, freq="M").to_timestamp("M")
    oos_start = train_idx.max() + pd.offsets.MonthEnd(1)
    return train_idx, oos_start


def load_forecast_files(selected, forecast_files=FORECAST_FILES, forecast_dir=FORECAST_DIR):
    """Read the tidy forecast file of each selected family and check the variants exist."""
    raw = {}
    for family, variants in selected.items():
        df = pd.read_csv(Path(forecast_dir) / forecast_files[family], parse_dates=["date"])
        unknown = set(variants) - set(df["model"].unique())
        assert not unknown, f"{family}: not in {forecast_files[family]}: {sorted(unknown)}"
        # `date` is the forecast origin t; the forecast is for holding month t+1
        df["holding_month"] = df["date"] + pd.offsets.MonthEnd(1)
        raw[family] = df
    return raw


def check_alignment(df, variant, rv_monthly):
    """Correlation of actual_RV with price-data RV of the same month t and of month t+1."""
    sub = df[df["model"] == variant]
    out = {}
    for label, months in [("same month (t)", sub["date"]), ("next month (t+1)", sub["holding_month"])]:
        rv = np.array([rv_monthly.at[m, e] if m in rv_monthly.index else np.nan
                       for m, e in zip(months, sub["etf"])])
        ok = ~np.isnan(rv)
        out[label] = np.corrcoef(sub["actual_RV"].to_numpy()[ok], rv[ok])[0, 1]
    return out


def forecast_wide(df, variant, column, tickers):
    """Wide table of one forecast column: rows = holding month t+1, columns = ETF."""
    sub = df[df["model"] == variant]
    return sub.pivot(index="holding_month", columns="etf", values=column)[tickers].sort_index()


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(sigma2_hat, xret, sigma2_bar, cap):
    """Capped volatility-managed weights, equal-weighted bucket, rescaled to BH volatility."""
    weights_base = np.minimum(sigma2_bar / sigma2_hat, cap)
    positions = weights_base * xret
    managed_unscaled = positions.mean(axis=1)
    bh = xret.mean(axis=1)
    c = bh.std() / managed_unscaled.std()
    return {
        "weights_base": weights_base,
        "positions": positions,
        "c": c,
        "managed": c * managed_unscaled,
        "weights_final": c * weights_base,                        # c x w(i,t)
        "bucket_weights": c * weights_base / xret.shape[1],       # weight held in each ETF
    }


def sigma2_bar_fixed(rv, train_idx):
    """Mean monthly realised variance per ETF over the fixed training window."""
    return rv.loc[train_idx].mean()


def sigma2_bar_expanding(rv, start, oos_idx):
    """Expanding mean RV: for holding month t+1, the mean from `start` through month t."""
    return rv.loc[start:].expanding().mean().shift(1).loc[oos_idx]


def build_control(sigma2_bar_exp, sigma2_bar_fix, xret, cap):
    """Re-levering control: weight min(sigma2_bar_t / sigma2_bar_fixed, cap), no forecast.

    build_portfolio with the fixed reference in place of the forecast, so the control has the same
    cap, equal weighting and rescale as the managed portfolios.
    """
    fixed_wide = pd.DataFrame(np.broadcast_to(sigma2_bar_fix[sigma2_bar_exp.columns].to_numpy(),
                                              sigma2_bar_exp.shape),
                              index=sigma2_bar_exp.index, columns=sigma2_bar_exp.columns)
    return build_portfolio(fixed_wide, xret, sigma2_bar_exp, cap)


def pct_capped(weights_base, cap):
    """% of months at the cap, per ETF."""
    return (weights_base >= cap).mean() * 100


def pct_capped_pooled(weights_base, cap):
    """% of ETF-months at the cap, pooled across ETFs."""
    return (weights_base >= cap).to_numpy().mean() * 100


# ---------------------------------------------------------------------------
# Saving and loading portfolios (one folder per sigma2_bar specification)
# ---------------------------------------------------------------------------

def _long(wide_by_model, value):
    """{model: months x ETFs} -> long DataFrame with columns model, date, etf, value."""
    frames = []
    for model, wide in wide_by_model.items():
        s = wide.stack()
        s.index.names = ["date", "etf"]
        frames.append(s.rename(value).reset_index().assign(model=model))
    return pd.concat(frames)[["model", "date", "etf", value]]


def save_portfolios(results, spec, xret, sigma2_bar, out_dir=PORTFOLIO_DIR):
    """Write every model's portfolio for one spec to results/portfolios/<spec>/.

    returns.csv (BH + each managed bucket, wide), positions.csv and weights.csv (base weights,
    long), c.csv, sigma2_bar.csv. xret.csv (monthly excess returns) is shared by both specs.
    """
    spec_dir = Path(out_dir) / spec
    spec_dir.mkdir(parents=True, exist_ok=True)
    xret.rename_axis("date").to_csv(Path(out_dir) / "xret.csv")

    returns = pd.DataFrame({"BH": xret.mean(axis=1)} | {m: r["managed"] for m, r in results.items()})
    returns.rename_axis("date").to_csv(spec_dir / "returns.csv")
    _long({m: r["positions"] for m, r in results.items()}, "position").to_csv(
        spec_dir / "positions.csv", index=False)
    _long({m: r["weights_base"] for m, r in results.items()}, "weight_base").to_csv(
        spec_dir / "weights.csv", index=False)
    pd.Series({m: r["c"] for m, r in results.items()}, name="c").rename_axis("model").to_csv(
        spec_dir / "c.csv")
    if isinstance(sigma2_bar, pd.Series):
        sigma2_bar = sigma2_bar.to_frame("sigma2_bar").T
    sigma2_bar.rename_axis("date").to_csv(spec_dir / "sigma2_bar.csv")


def load_portfolios(spec, in_dir=PORTFOLIO_DIR):
    """Read one spec's portfolios back into the dicts build_portfolio returns.

    Returns (results, xret): results maps model -> {weights_base, positions, c, managed,
    weights_final, bucket_weights}, xret is the months x ETFs excess-return table.
    """
    spec_dir = Path(in_dir) / spec
    xret = pd.read_csv(Path(in_dir) / "xret.csv", index_col="date", parse_dates=["date"])
    returns = pd.read_csv(spec_dir / "returns.csv", index_col="date", parse_dates=["date"])
    positions = pd.read_csv(spec_dir / "positions.csv", parse_dates=["date"])
    weights = pd.read_csv(spec_dir / "weights.csv", parse_dates=["date"])
    c = pd.read_csv(spec_dir / "c.csv", index_col="model")["c"]

    results = {}
    for model in returns.columns.drop("BH"):
        w = weights[weights["model"] == model].pivot(index="date", columns="etf", values="weight_base")
        p = positions[positions["model"] == model].pivot(index="date", columns="etf", values="position")
        w, p = w[xret.columns], p[xret.columns]
        results[model] = {
            "weights_base": w,
            "positions": p,
            "c": c[model],
            "managed": returns[model],
            "weights_final": c[model] * w,
            "bucket_weights": c[model] * w / xret.shape[1],
        }
    return results, xret


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def max_drawdown(r):
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def avg_turnover(weights):
    # weights: DataFrame (one column per holding) or Series; first month has no prior weight
    dw = weights.diff().abs().iloc[1:]
    return dw.sum(axis=1).mean() if isinstance(dw, pd.DataFrame) else dw.mean()


def performance_row(r, weights):
    return {
        "ann_mean": r.mean() * PERIODS_PER_YEAR,
        "ann_vol": r.std() * np.sqrt(PERIODS_PER_YEAR),
        "ann_sharpe": r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR),
        "max_drawdown": max_drawdown(r),
        "avg_turnover": avg_turnover(weights),
    }


def _sr_diff_and_grad(x, y):
    # Delta = SR_x - SR_y in terms of first and second moments
    mu_x, mu_y = x.mean(), y.mean()
    g_x, g_y = (x ** 2).mean(), (y ** 2).mean()
    v_x, v_y = g_x - mu_x ** 2, g_y - mu_y ** 2
    diff = mu_x / np.sqrt(v_x) - mu_y / np.sqrt(v_y)
    grad = np.array([g_x / v_x ** 1.5, -g_y / v_y ** 1.5,
                     -mu_x / (2 * v_x ** 1.5), mu_y / (2 * v_y ** 1.5)])
    return diff, grad


def _moments(x, y):
    return np.column_stack([x, y, x ** 2, y ** 2])


def _hac_cov(z, lags):
    z = z - z.mean(axis=0)
    T = len(z)
    psi = z.T @ z / T
    for j in range(1, lags + 1):
        gamma = z[j:].T @ z[:-j] / T
        psi += (1 - j / (lags + 1)) * (gamma + gamma.T)
    return psi


def ledoit_wolf_sharpe_test(x, y, block_size=5, n_boot=5000, seed=0):
    # Ledoit & Wolf (2008) studentised circular block bootstrap test of SR_x = SR_y
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = len(x)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    diff, grad = _sr_diff_and_grad(x, y)
    se = np.sqrt(grad @ _hac_cov(_moments(x, y), lags) @ grad / T)
    t_stat = diff / se

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_size))
    T_b = n_blocks * block_size
    offsets = np.arange(block_size)
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = ((starts[:, None] + offsets) % T).ravel()
        xb, yb = x[idx], y[idx]
        diff_b, grad_b = _sr_diff_and_grad(xb, yb)
        z = _moments(xb, yb)
        z = z - z.mean(axis=0)
        block_sums = z.reshape(n_blocks, block_size, 4).sum(axis=1) / np.sqrt(block_size)
        psi_b = block_sums.T @ block_sums / n_blocks
        se_b = np.sqrt(grad_b @ psi_b @ grad_b / T_b)
        t_boot[b] = (diff_b - diff) / se_b

    p_value = (np.sum(np.abs(t_boot) >= np.abs(t_stat)) + 1) / (n_boot + 1)
    return {
        "SR_managed": x.mean() / x.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR),
        "SR_benchmark": y.mean() / y.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR),
        "SR_diff": diff * np.sqrt(PERIODS_PER_YEAR),
        "t_stat": t_stat,
        "p_value": p_value,
    }


def alpha_regression(y, X):
    T = len(y)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    fit = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    row = {
        "alpha_monthly": fit.params["const"],
        "alpha_annual": fit.params["const"] * PERIODS_PER_YEAR,
        "alpha_t": fit.tvalues["const"],
        "alpha_p": fit.pvalues["const"],
    }
    for name in X.columns:
        row[f"beta_{name}"] = fit.params[name]
        row[f"beta_{name}_t"] = fit.tvalues[name]
    row.update({"R2": fit.rsquared, "NW_lags": lags, "T": T})
    return row


def evaluate_bucket(res, bh):
    """Performance, Ledoit-Wolf test vs BH and alpha on BH for one managed bucket."""
    lw = ledoit_wolf_sharpe_test(res["managed"], bh)
    al = alpha_regression(res["managed"], bh.to_frame("BH"))
    return {
        **performance_row(res["managed"], res["bucket_weights"]),
        "c": res["c"],
        "SR_diff_vs_BH": lw["SR_diff"],
        "LW_t": lw["t_stat"],
        "LW_p": lw["p_value"],
        "alpha_annual": al["alpha_annual"],
        "alpha_t": al["alpha_t"],
        "alpha_p": al["alpha_p"],
        "beta_BH": al["beta_BH"],
        "R2": al["R2"],
    }


def alpha_vs_bh_and_control(res, bh, control):
    """Two-factor alpha: managed = a + b1 BH + b2 control + e, Newey-West errors.

    a is what the forecast adds beyond holding the ETFs and beyond re-levering.
    """
    al = alpha_regression(res["managed"], pd.DataFrame({"BH": bh, "control": control["managed"]}))
    return {k: al[k] for k in ["alpha_annual", "alpha_t", "alpha_p", "beta_BH", "beta_control", "R2"]}


def compare_pair(res_a, res_b, xret):
    """Ledoit-Wolf test of SR(a) = SR(b): cyclical bucket and each ETF on its own."""
    rows = {"Cyclical": ledoit_wolf_sharpe_test(res_a["managed"], res_b["managed"])}
    managed_a, _ = per_etf_returns(res_a, xret)
    managed_b, _ = per_etf_returns(res_b, xret)
    for t in xret.columns:
        rows[t] = ledoit_wolf_sharpe_test(managed_a[t], managed_b[t])
    out = pd.DataFrame(rows).T.rename(columns={"SR_managed": "SR_a", "SR_benchmark": "SR_b",
                                               "t_stat": "LW_t", "p_value": "LW_p"})
    return out.rename_axis("ETF")


def per_etf_returns(res, xret):
    """Each ETF's managed series, rescaled to its own BH volatility: c_i x position_i."""
    c_i = xret.std() / res["positions"].std()
    return res["positions"] * c_i, c_i


def per_etf_table(res, xret):
    """Per-ETF managed series vs that ETF's own buy-and-hold excess return."""
    managed, c_i = per_etf_returns(res, xret)
    rows = {}
    for t in xret.columns:
        r_m, r_b = managed[t], xret[t]
        lw = ledoit_wolf_sharpe_test(r_m, r_b)
        al = alpha_regression(r_m, r_b.to_frame("BH"))
        rows[t] = {
            "c_i": c_i[t],
            "SR_managed": lw["SR_managed"],
            "SR_BH": lw["SR_benchmark"],
            "SR_diff": lw["SR_diff"],
            "LW_p": lw["p_value"],
            "alpha_annual": al["alpha_annual"],
            "alpha_t": al["alpha_t"],
            "maxDD_managed": max_drawdown(r_m),
            "maxDD_BH": max_drawdown(r_b),
            "avg_turnover": avg_turnover(c_i[t] * res["weights_base"][t]),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Forecast accuracy
# ---------------------------------------------------------------------------

def qlike(realized, forecast):
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1
