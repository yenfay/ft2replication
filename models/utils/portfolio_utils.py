"""Shared code for the notebooks in models/portfolio.

p1_build_portfolios builds the volatility-managed portfolios, p2_portfolio_eval evaluates them and
p3_forecast_vs_portfolio links forecast accuracy to portfolio performance. Paths are resolved from
this file's location, so the notebooks work from any working directory.
"""
from pathlib import Path
from .paths import PROJECT_ROOT, FORECAST_DIR, PORTFOLIO_RESULTS_DIR
from .forecast_data import FORECAST_FILES, expected_forecast_months, validate_forecast_frame

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats

PERIODS_PER_YEAR = 12

RETURNS_PATH = PROJECT_ROOT / "dataset" / "price" / "clean" / "daily_returns_panel_ver2.csv"
VIX_PATH = PROJECT_ROOT / "dataset" / "macro_variables" / "raw" / "VIXCLS.csv"
RESULTS_DIR = PORTFOLIO_RESULTS_DIR
PORTFOLIO_DIR = RESULTS_DIR / "portfolios"   # written by p1_build_portfolios
EVAL_DIR = RESULTS_DIR / "eval"              # written by p2_portfolio_eval and p3_forecast_vs_portfolio

CONTROL = "control"   # name of the re-levering control portfolio (expanding spec only)
BH = "BH"             # name of the buy-and-hold benchmark in the result tables

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_monthly_returns(tickers, path=RETURNS_PATH):
    """Monthly excess returns and RV of daily excess returns, matching model targets."""
    daily = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    ret_daily = daily[[f"ret_{t}" for t in tickers]].rename(columns=lambda c: c.removeprefix("ret_"))
    xret_daily = daily[[f"xret_{t}" for t in tickers]].rename(columns=lambda c: c.removeprefix("xret_"))
    if daily.index.hasnans or daily.index.has_duplicates:
        raise ValueError("Daily return dates must be present and unique")
    if not np.isfinite(daily[["rf_daily", *[f"ret_{t}" for t in tickers],
                                  *[f"xret_{t}" for t in tickers]]].to_numpy()).all():
        raise ValueError("Daily returns and risk-free rates must be finite")

    ret_monthly = (1 + ret_daily).resample("ME").prod() - 1
    rf_monthly = (1 + daily["rf_daily"]).resample("ME").prod() - 1
    xret_monthly = ret_monthly.sub(rf_monthly, axis=0)
    rv_monthly = xret_daily.pow(2).resample("ME").sum(min_count=1)
    return xret_monthly, rv_monthly


def training_window(train_start, train_months):
    """Month-end index of the fixed training window and the first out-of-sample month."""
    train_idx = pd.period_range(train_start, periods=train_months, freq="M").to_timestamp("M")
    oos_start = train_idx.max() + pd.offsets.MonthEnd(1)
    return train_idx, oos_start


def load_forecast_files(selected, forecast_files=FORECAST_FILES, forecast_dir=FORECAST_DIR):
    """Read the tidy forecast file of each selected family.

    Returns (raw, available). Every selected model must cover the configured sample.
    """
    from research_config import TICKERS, TRAIN_START, TRAIN_MONTHS

    actuals = pd.read_csv(PROJECT_ROOT / "dataset/price/clean/realized_variance.csv", parse_dates=["date"])
    months = expected_forecast_months(actuals, TRAIN_START, TRAIN_MONTHS)
    raw, available = {}, {}
    for family, variants in selected.items():
        df = pd.read_csv(Path(forecast_dir) / forecast_files[family], parse_dates=["date"])
        validate_forecast_frame(df, actuals, variants, TICKERS, months, forecast_files[family])
        # `date` is the forecast origin t; the forecast is for holding month t+1
        df["holding_month"] = df["date"] + pd.offsets.MonthEnd(1)
        raw[family] = df
        available[family] = list(variants)
    return raw, available


def forecast_wide(df, variant, column, tickers):
    """Wide table of one forecast column: rows = holding month t+1, columns = ETF."""
    sub = df[df["model"] == variant]
    return sub.pivot(index="holding_month", columns="etf", values=column)[tickers].sort_index()


def load_vix_monthly(path=VIX_PATH):
    """Monthly average of the daily VIX close, indexed by month end."""
    vix = pd.read_csv(path, parse_dates=["date"], na_values=["."]).set_index("date")["value"]
    return vix.astype(float).resample("ME").mean()


# ---------------------------------------------------------------------------
# Comparison blocks
# ---------------------------------------------------------------------------

def resolve_blocks(blocks, available):
    """The pairs in `blocks` whose two models are both in `available`.

    Returns (pairs, skipped): pairs has one row per runnable pair (block, pair, a, b, isolates),
    skipped one row per pair left out, with the models it is missing.
    """
    rows, skipped = [], []
    for block_id, block in blocks.items():
        for pair_id, a, b, isolates in block["pairs"]:
            missing = [m for m in (a, b) if m not in available]
            row = {"block": block_id, "pair": pair_id, "a": a, "b": b, "isolates": isolates}
            if missing:
                skipped.append(row | {"missing": ", ".join(missing)})
            else:
                rows.append(row)
    pairs = pd.DataFrame(rows).set_index("pair")
    skipped = pd.DataFrame(skipped, columns=["block", "pair", "a", "b", "isolates", "missing"])
    return pairs, skipped.set_index("pair")


def block_models(block, available):
    """The block's models that were built, in the block's order."""
    return [m for m in block["models"] if m in available]


def stars(p):
    """Significance stars: *** 1%, ** 5%, * 10%."""
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def outcome(gain, p, level=0.10):
    """'a better', 'a worse' or 'n.s.': the sign of `gain` if p < level."""
    if pd.isna(p) or p >= level:
        return "n.s."
    return "a better" if gain > 0 else "a worse"


def verdict(pair_id, a, b, gain, p, label="ΔSR", noun="portfolio"):
    """One line summarising a pair test; positive `gain` means a is better."""
    if p < 0.05:
        text = f"{a} gives a significantly {'better' if gain > 0 else 'worse'} {noun} than {b}"
    elif p < 0.10:
        text = f"{a} gives a weakly significantly {'better' if gain > 0 else 'worse'} {noun} than {b}"
    else:
        text = f"no significant difference between {a} and {b}"
    return f"{pair_id}: {text} ({label} {gain:+.3f}, p = {p:.3f})"


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(sigma2_hat, xret, sigma2_bar, cap):
    """Capped volatility-managed weights, equal-weighted bucket, rescaled to BH volatility."""
    if not sigma2_hat.index.equals(xret.index) or not sigma2_hat.columns.equals(xret.columns):
        raise ValueError("Forecast and return dates/ETFs must match before portfolio construction")
    if (not np.isfinite(sigma2_hat.to_numpy()).all() or (sigma2_hat <= 0).any().any()
            or not np.isfinite(xret.to_numpy()).all()):
        raise ValueError("Portfolio forecasts must be positive and finite; returns must be finite")
    weights_base = np.minimum(sigma2_bar / sigma2_hat, cap)
    positions = weights_base * xret
    managed_unscaled = positions.mean(axis=1)
    bh = xret.mean(axis=1)
    c = bh.std() / managed_unscaled.std()
    if not np.isfinite(c) or c <= 0:
        raise ValueError("Cannot volatility-scale a portfolio with zero or invalid volatility")
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


def weight_distance(res_a, res_b):
    """Mean |w_a - w_b| of the base weights over ETF-months: how far apart two models' bets are."""
    return (res_a["weights_base"] - res_b["weights_base"]).abs().to_numpy().mean()


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

    returns = pd.DataFrame({BH: xret.mean(axis=1)} | {m: r["managed"] for m, r in results.items()})
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
    for model in returns.columns.drop(BH):
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

def nw_lags(T):
    """Newey-West lag rule floor(4 (T/100)^(2/9)), used by every HAC estimator here."""
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


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
    lags = nw_lags(T)

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
        "SR_diff_se": se * np.sqrt(PERIODS_PER_YEAR),   # HAC standard error of SR_diff: t_stat = SR_diff / SR_diff_se
        "t_stat": t_stat,
        "p_value": p_value,
    }


def alpha_regression(y, X):
    T = len(y)
    lags = nw_lags(T)
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
                                               "SR_diff_se": "LW_se", "t_stat": "LW_t", "p_value": "LW_p"})
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
            "LW_t": lw["t_stat"],
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Categorical slots in fixed order; buy-and-hold and the historical benchmark are neutral greys.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BH_COLOR = "#0b0b0b"
BENCHMARK_COLOR = "#8a8985"
TEXT_SECONDARY = "#52514e"
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]   # negative (worse) - neutral - positive (better)


def style_axis(ax):
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def label_points(ax, xs, ys, labels, fontsize=7):
    """Label each point at the first spot around it that overlaps no point and no earlier label.

    Labels are placed in screen space, so call this after the limits and layout are final.
    """
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    r = 4 * fig.dpi / 72   # about a marker's radius, in pixels
    taken = [Bbox.from_extents(px - r, py - r, px + r, py + r)
             for px, py in ax.transData.transform(np.column_stack([xs, ys]))]
    spots = [(6, 0, "left", "center"), (-6, 0, "right", "center"), (0, 7, "center", "bottom"),
             (0, -7, "center", "top"), (5, 5, "left", "bottom"), (5, -5, "left", "top"),
             (-5, 5, "right", "bottom"), (-5, -5, "right", "top")]
    for x, y, text in zip(xs, ys, labels):
        for i, (dx, dy, ha, va) in enumerate(spots + spots[:1]):   # no free spot: fall back to the first
            ann = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                              fontsize=fontsize, color=TEXT_SECONDARY, zorder=4,
                              bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75))
            box = ann.get_window_extent(renderer)
            if i == len(spots) or not any(box.overlaps(t) for t in taken):
                break
            ann.remove()
        taken.append(box)


def plot_growth(returns, ax, title, benchmark=None):
    """Log-scale growth of $1 in excess returns. BH is black; `benchmark` is a dashed grey line."""
    colors = iter(SERIES_COLORS)
    for name, r in returns.items():
        if name == BH:
            style = dict(color=BH_COLOR, linewidth=2)
        elif name == benchmark:
            style = dict(color=BENCHMARK_COLOR, linewidth=1.5, linestyle="--")
        else:
            style = dict(color=next(colors), linewidth=1.5)
        ax.plot(r.index, (1 + r).cumprod(), label=name, **style)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("growth of $1 (log scale)", fontsize=8, color=TEXT_SECONDARY)
    style_axis(ax)
    ax.legend(fontsize=8, frameon=False, ncol=2)


def heatmap(values, pvalues, ax, title, center_label="ΔSR"):
    """Diverging heatmap of `values` (rows x columns), each cell labelled with its value and stars."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    v = values.to_numpy(float)
    lim = np.nanmax(np.abs(v)) or 1.0
    im = ax.imshow(v, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, f"{v[i, j]:+.2f}{stars(pvalues.iat[i, j])}", ha="center", va="center",
                        fontsize=8, color=BH_COLOR)
    ax.set_xticks(range(v.shape[1]), values.columns, fontsize=8)
    ax.set_yticks(range(v.shape[0]), values.index, fontsize=8)
    ax.tick_params(length=0, colors=TEXT_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(center_label, fontsize=8, color=TEXT_SECONDARY)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY)
    cbar.outline.set_visible(False)
