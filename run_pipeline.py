"""Reproduce data preparation, EDA, forecasts, evaluations, portfolios and robustness checks."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
DATA_STEPS = (
    "dataset/macro_variables/fetch_fred_data.py",
    "dataset/macro_variables/transform_macro_data.py",
    "dataset/price/clean_price_data_ver2.py",
    "dataset/price/calculate_realized_variance.py",
    "dataset/price/clean/eda_realized_volatility.ipynb",
)
BUILD_STEPS = (
    "models/forecasting/build/b0_hist_rw.ipynb",
    "models/forecasting/build/b1_garch_garchmidas.ipynb",
    "models/forecasting/build/b2_elastic_net.ipynb",
    "models/forecasting/build/b3_xgboost.ipynb",
)
DATA_CHECK_STEPS = (
    "dataset/macro_variables/check_unrevised_market_series.ipynb",
    "dataset/macro_variables/check_baa10y_vintageness.ipynb",
)
ANALYSIS_STEPS = (
    "models/forecasting/evaluation/e1_garch_family.ipynb",
    "models/forecasting/evaluation/e2_macro_value.ipynb",
    "models/forecasting/evaluation/e3_vix_effect_en.ipynb",
    "models/forecasting/evaluation/e4_vix_effect_xgb.ipynb",
    "models/forecasting/evaluation/e5_linear_vs_nonlinear.ipynb",
    "models/forecasting/evaluation/e6_naive_vs_all_models.ipynb",
    "models/forecasting/robustness/r1_crisis_en.ipynb",
    "models/forecasting/robustness/r2_crisis_naive_vs_garch_en.ipynb",
    "models/forecasting/robustness/r3_rolling_window_en.ipynb",
    "models/portfolio/p1_build_portfolios.ipynb",
    "models/portfolio/p2_portfolio_eval.ipynb",
    "models/portfolio/p3_forecast_vs_portfolio.ipynb",
    "models/portfolio/robustness/rp1_different_leverages.ipynb",
    "models/portfolio/robustness/rp2_timing_vs_relevering.ipynb",
)
SAVED_FORECASTS = (
    "models/forecasting/forecasts/hist_rw_forecasts.csv",
    "models/forecasting/forecasts/garch_forecasts.csv",
    "models/forecasting/forecasts/en_forecasts.csv",
    "models/forecasting/forecasts/xgb_forecasts.csv",
    "models/forecasting/robustness/forecasts/en_rolling60_forecasts.csv",
)

# Report sources and saved data.
# Fields: report item, source, section heading, saved data, coverage.
REPORT_RESULTS = (
    ("Table 1", "dataset/price/clean/eda_realized_volatility.ipynb", "Summary table",
     ("dataset/price/clean/realized_variance.csv",), "Report table display"),
    ("Figure 1", "dataset/price/clean/eda_realized_volatility.ipynb", "Monthly realised volatility (%)",
     (), "Volatility chart display"),
    ("Table 2", "models/forecasting/evaluation/e1_garch_family.ipynb", "By-sector percentage changes",
     ("models/forecasting/results/e1_evaluation.csv",), "Loss changes and p-values in separate displays/CSV"),
    ("Table 3", "models/forecasting/evaluation/e2_macro_value.ipynb", "By sector percentage changes",
     ("models/forecasting/results/e2_evaluation.csv",), "Loss changes and p-values in separate displays/CSV"),
    ("Table 4", "models/forecasting/evaluation/e3_vix_effect_en.ipynb", "By sector percentage changes",
     ("models/forecasting/results/e3_evaluation.csv",), "Loss changes and p-values in separate displays/CSV"),
    ("Table 5", "models/forecasting/evaluation/e5_linear_vs_nonlinear.ipynb", "Full-information report table",
     ("models/forecasting/results/e5_evaluation.csv",), "Report table display"),
    ("Table 6", "models/portfolio/p2_portfolio_eval.ipynb", "Part A: Cyclical bucket",
     ("models/portfolio/results/eval/vs_benchmarks_bucket_fixed.csv",), "Component performance table; growth also in D3"),
    ("Table A1", "dataset/macro_variables/fetch_fred_data.py", "SERIES_CONFIG",
     ("dataset/macro_variables/clean/fred_monthly_panel.csv",), "Source definitions, not a formatted report table"),
    ("Predictor exclusions / Table A1 transformations", "dataset/macro_variables/transform_macro_data.py",
     "DROP_NOW, FINAL_COLUMNS and LOG_DIFF_VARS",
     ("dataset/macro_variables/final_clean/final_macro_data.csv",), "Source definitions, not a formatted report table"),
    ("Table B1", "models/forecasting/build/b1_garch_garchmidas.ipynb", "Choice of the GARCH-MIDAS driver",
     ("models/forecasting/results/garch_midas_candidate_losses.csv",), "Candidate comparison displays and full-precision CSV"),
    ("Table B2 (Elastic Net)", "models/forecasting/build/b2_elastic_net.ipynb", "Recursive forecasts",
     (), "Settings in source, also in the opening code cell; no formatted report table"),
    ("Table B2 (XGBoost)", "models/forecasting/build/b3_xgboost.ipynb", "1. Settings",
     (), "Settings in source; no formatted report table"),
    ("Table C1", "models/forecasting/robustness/r1_crisis_en.ipynb", "Pooled results",
     ("models/forecasting/robustness/results/r1_crisis_evaluation.csv",), "Use pooled B3.2 rows across samples"),
    ("Table C2", "models/forecasting/robustness/r3_rolling_window_en.ipynb", "Pooled results",
     ("models/forecasting/robustness/results/r3_rolling_evaluation.csv",), "Use pooled expanding/rolling rows"),
    ("Table C3", "models/forecasting/build/b3_xgboost.ipynb", "7. Feature contributions",
     ("models/forecasting/results/xgb_shap_mean_abs.csv", "models/forecasting/results/xgb_shap_share_pct.csv"),
     "SHAP displays and CSVs"),
    ("Table C4", "models/forecasting/evaluation/e4_vix_effect_xgb.ipynb", "5. Percentage change in loss by comparison",
     ("models/forecasting/results/e4_evaluation.csv",), "Per-comparison loss-change displays and CSV"),
    ("Table D1", "models/portfolio/p1_build_portfolios.ipynb", "Step 2: The reference variance σ̄², two specifications",
     ("models/portfolio/results/portfolios/fixed/sigma2_bar.csv",
      "models/portfolio/results/portfolios/expanding/sigma2_bar.csv"), "Reference display and underlying data"),
    ("Table D2", "models/portfolio/p1_build_portfolios.ipynb", "Weight diagnostics",
     ("models/portfolio/results/portfolios/weight_diagnostics_pooled.csv",
      "models/portfolio/results/portfolios/fixed/c.csv", "models/portfolio/results/portfolios/expanding/c.csv"),
     "Component weight diagnostics and scaling factors"),
    ("Table D3", "models/portfolio/p2_portfolio_eval.ipynb", "Growth of $1 by period",
     ("models/portfolio/results/portfolios/fixed/returns.csv",), "Growth-by-period table display"),
    ("Table D4", "models/portfolio/p2_portfolio_eval.ipynb", "A.7 Bucket scoreboard",
     ("models/portfolio/results/eval/pairs.csv",), "Pair-test displays in A.1–A.6; bucket scoreboard and saved tests"),
    ("Table D5 / Figure 4", "models/portfolio/p3_forecast_vs_portfolio.ipynb", "3 Do better forecasters build better portfolios?",
     ("models/portfolio/results/eval/forecast_ranking.csv",
      "models/portfolio/results/eval/portfolio_ranking_fixed.csv",
      "models/portfolio/results/eval/rank_summary_fixed.csv"), "Rank chart and correlations; ranking tables in sections 1–2"),
    ("Table D6 / Table D7 / Figure 2", "models/portfolio/p2_portfolio_eval.ipynb", "Part B: Per-ETF portfolios",
     ("models/portfolio/results/eval/vs_benchmarks_per_etf_fixed.csv",),
     "Per-ETF heatmaps; full component tables in B.3 Full per-ETF tables"),
    ("Table D8", "models/portfolio/p2_portfolio_eval.ipynb", "B.3 Full per-ETF tables",
     ("models/portfolio/results/eval/vs_benchmarks_per_etf_fixed.csv",), "Use maxDD_managed and maxDD_BH fields"),
    ("Table D9", "models/portfolio/robustness/rp2_timing_vs_relevering.ipynb", "The expanding reference against the fixed one",
     ("models/portfolio/robustness/results/rp2_timing_vs_relevering.csv",
      "models/portfolio/results/eval/vs_benchmarks_bucket_expanding.csv"), "Reference comparison display and component CSVs"),
    ("Table D10", "models/portfolio/robustness/rp2_timing_vs_relevering.ipynb", "Timing or re-levering?",
     ("models/portfolio/robustness/results/rp2_timing_vs_relevering.csv",), "Spanning regression display and CSV"),
    ("Table D11", "models/portfolio/robustness/rp1_different_leverages.ipynb", "Against buy-and-hold, by cap",
     ("models/portfolio/robustness/results/rp1_different_leverages.csv",), "Leverage-cap displays and CSV"),
    ("Figure 3", "models/portfolio/p2_portfolio_eval.ipynb", "A.2 B1: GARCH family",
     ("models/portfolio/results/portfolios/fixed/returns.csv",), "GARCH-family growth chart display"),
)


def report_labels(filename):
    """Report items supplied by a particular pipeline source."""
    return [item for item, source, *_ in REPORT_RESULTS if source == filename]


def file_link(path, label=None):
    """Markdown link relative to the root runner notebooks, including paths with spaces."""
    path = Path(path)
    relative = os.path.relpath(path, ROOT) if path.is_absolute() else path.as_posix()
    return f"[{label or path.name}]({quote(relative, safe='/')})"


def report_index_markdown(run_dir=None):
    """Source map, optionally linked to the fresh executed copies from a run."""
    rows = ["## Report tables and figures", "",
            "Sources for report tables and figures. Executed notebooks contain the figures. "
            "Saved data are linked when available; skipped builders do not refresh their outputs.", "",
            "| Report item | Source and section | Saved data | Coverage |",
            "|---|---|---|---|"]
    for item, source, section, outputs, coverage in REPORT_RESULTS:
        location = file_link(source) + f" — {section}"
        if run_dir is not None and source.endswith(".ipynb"):
            executed = Path(run_dir) / "notebooks" / source
            location += "<br>" + (file_link(executed, "Fresh executed copy") if executed.is_file()
                                   else "Not executed in this run (e.g. forecasts reused)")
        elif run_dir is not None:
            log = Path(run_dir) / "logs" / (Path(source).stem + ".log")
            if log.is_file():
                location += "<br>" + file_link(log, "Run log")
        data_links = []
        skipped_builder = (run_dir is not None and source in BUILD_STEPS
                           and not (Path(run_dir) / "notebooks" / source).is_file())
        for path in outputs:
            if (ROOT / path).is_file():
                status = " (available; builder not run)" if skipped_builder else ""
                data_links.append(file_link(path, path.removeprefix("models/")) + status)
            else:
                data_links.append(f"`{path}` — unavailable; run `{source}`")
        data = "<br>".join(data_links) or "Notebook/source only"
        rows.append(f"| {item} | {location} | {data} | {coverage} |")
    return "\n".join(rows)


def show_step_outputs(filename, notebook, executed, log, display_mode):
    """Link every step and replay freshly executed outputs for the inline runner."""
    from IPython.display import Markdown, display

    links = [file_link(filename, "Source")]
    if executed is not None:
        links.append(file_link(executed, "Executed notebook"))
    if log is not None:
        links.append(file_link(log, "Script log"))
    display(Markdown(" | ".join(links)))
    if display_mode != "inline":
        return
    if log is not None:
        print(log.read_text(), end="")
    if notebook is None:
        return
    section = "Setup"
    shown_section = None
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            headings = [line.lstrip("# ") for line in cell.source.splitlines() if line.startswith("#")]
            if headings:
                section = headings[-1]
            continue
        if not cell.get("outputs"):
            continue
        if section != shown_section:
            labels = [item for item, source, heading, *_ in REPORT_RESULTS
                      if source == filename and heading.casefold() == section.casefold()]
            suffix = " — " + "; ".join(labels) if labels else ""
            display(Markdown(f"### {section}{suffix}"))
            shown_section = section
        for output in cell.outputs:
            if output.output_type == "stream":
                print(output.text, end="", file=sys.stderr if output.name == "stderr" else sys.stdout)
            elif output.output_type in ("display_data", "execute_result"):
                display(output.data, raw=True, metadata=output.get("metadata", {}))
            elif output.output_type == "error":
                print("\n".join(output.traceback), file=sys.stderr)


def pipeline_steps(reuse_forecasts=False, include_data_checks=False):
    steps = list(DATA_STEPS)
    if include_data_checks:
        steps[2:2] = DATA_CHECK_STEPS
    if not reuse_forecasts:
        steps.extend(BUILD_STEPS)
    return steps + list(ANALYSIS_STEPS)


def preflight(start, steps, reuse_forecasts=False):
    """Check inputs and dependencies before any research output is overwritten."""
    required = list(steps) + [
        "dataset/price/raw/sector_etf_prices_with_xlre.csv",
        "dataset/price/raw/risk_free_rate_returns.csv",
    ]
    spec = importlib.util.spec_from_file_location(
        "fred_pipeline", ROOT / "dataset/macro_variables/fetch_fred_data.py")
    fred = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fred)
    if start == "raw":
        required.extend(f"dataset/macro_variables/raw/{series}.csv"
                        for series in fred.FINAL_PANEL_COLUMNS)
    if reuse_forecasts:
        required.extend(SAVED_FORECASTS)
        required.append("models/forecasting/forecasts/garchmidas_theta_by_origin.csv")
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError("Required files are missing:\n" + "\n".join(missing))
    if start == "fetch" or any(step in steps for step in DATA_CHECK_STEPS):
        fred.get_fred_api_key()
    # Import compiled dependencies too: installed metadata alone does not establish usability.
    for module in ("nbformat", "nbclient", "ipykernel", "numba", "xgboost", "statsmodels.api"):
        __import__(module)


def validate_saved_forecasts():
    """Check main and rolling forecasts against the rebuilt RV before evaluation."""
    import pandas as pd
    from models.research_config import SELECTED, TICKERS, TRAIN_START, TRAIN_MONTHS
    from models.utils.forecast_data import (
        FORECAST_FILES, ROLLING_FORECAST_FILES, expected_forecast_months, validate_forecast_frame,
    )

    actuals = pd.read_csv(ROOT / "dataset/price/clean/realized_variance.csv", parse_dates=["date"])
    months = expected_forecast_months(actuals, TRAIN_START, TRAIN_MONTHS)
    forecast_dir = ROOT / "models/forecasting/forecasts"
    files = [(forecast_dir / FORECAST_FILES[family], variants)
             for family, variants in SELECTED.items()]
    files.extend((ROOT / "models/forecasting/robustness/forecasts" / filename, variants)
                 for filename, variants in ROLLING_FORECAST_FILES.items())
    for path, variants in files:
        frame = pd.read_csv(path)
        validate_forecast_frame(frame, actuals, variants, TICKERS, months, str(path))
    print(f"Validated {len(files)} forecast files: {months[0]:%Y-%m} to {months[-1]:%Y-%m}", flush=True)


def run_pipeline(*, start="raw", reuse_forecasts=False, use_fit_cache=True,
                 include_data_checks=False, run_dir=None, display_mode="console"):
    """Run the research workflow; return the directory of executed copies/logs.

    ``inline`` replays fresh notebook outputs; ``compact`` displays links only.
    Both notebook modes use the same steps and kernels as the console runner.
    """
    if start not in ("raw", "fetch"):
        raise ValueError("start must be 'raw' or 'fetch'")
    if display_mode not in ("console", "inline", "compact"):
        raise ValueError("display_mode must be 'console', 'inline' or 'compact'")
    if reuse_forecasts and not use_fit_cache:
        raise ValueError("Disabling the fit cache has no effect with reuse_forecasts")
    steps = pipeline_steps(reuse_forecasts, include_data_checks)
    preflight(start, steps, reuse_forecasts)
    import nbformat
    from nbclient import NotebookClient

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(run_dir) if run_dir is not None else ROOT / "supporting/pipeline_runs" / stamp
    run_dir = run_dir.resolve()
    # Keep executed notebooks from separate runs in separate directories.
    run_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, DSE4101_RECOMPUTE_FORECASTS="1", MPLBACKEND="Agg",
               DSE4101_USE_FIT_CACHE="1" if use_fit_cache else "0")
    if display_mode != "console":
        # Agg suppresses plt.show() figures in Jupyter; use the inline backend in child kernels.
        env["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    print(f"Executed notebooks and logs: {run_dir}", flush=True)
    if reuse_forecasts:
        print("Using saved forecasts; b0-b3 will NOT be refitted.", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="dse4101-kernel-") as temporary:
            spec = Path(temporary) / "kernels/dse4101-pipeline"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Pipeline Python", "language": "python",
            }))
            # Kernel discovery uses the parent process environment, not the kernel's env argument.
            previous_jupyter_path = os.environ.get("JUPYTER_PATH")
            os.environ["JUPYTER_PATH"] = temporary + os.pathsep + (previous_jupyter_path or "")
            try:
                for index, filename in enumerate(steps, 1):
                    if filename == ANALYSIS_STEPS[0]:
                        validate_saved_forecasts()
                    print(f"[{index}/{len(steps)}] {filename}", flush=True)
                    labels = report_labels(filename)
                    if labels:
                        print("Report: " + "; ".join(labels), flush=True)
                    notebook = None
                    log = None
                    executed = None
                    try:
                        path = ROOT / filename
                        if path.suffix == ".py":
                            command = [sys.executable, "-u", str(path)]
                            if filename == DATA_STEPS[0] and start == "raw":
                                command.append("--from-raw")
                            log = run_dir / "logs" / (Path(filename).stem + ".log")
                            log.parent.mkdir(parents=True, exist_ok=True)
                            with log.open("w") as stream:
                                subprocess.run(command, cwd=ROOT, env=env, check=True,
                                               stdout=stream, stderr=subprocess.STDOUT)
                        else:
                            notebook = nbformat.read(path, as_version=4)
                            # Remove historical displays before a run, including unexecuted cells
                            # in a failed copy. Only fresh outputs may be replayed or saved.
                            for cell in notebook.cells:
                                if cell.cell_type == "code":
                                    cell.outputs = []
                                    cell.execution_count = None
                            NotebookClient(
                                notebook, kernel_name="dse4101-pipeline", timeout=None,
                                allow_errors=False, ipython_hist_file=":memory:",
                            ).execute(cwd=str(ROOT), env=env)
                    finally:
                        if notebook is not None:
                            executed = run_dir / "notebooks" / filename
                            executed.parent.mkdir(parents=True, exist_ok=True)
                            nbformat.write(notebook, executed)
                        if display_mode != "console":
                            show_step_outputs(filename, notebook, executed, log, display_mode)
            finally:
                if previous_jupyter_path is None:
                    os.environ.pop("JUPYTER_PATH", None)
                else:
                    os.environ["JUPYTER_PATH"] = previous_jupyter_path
    except Exception as exc:
        failed = filename if "filename" in locals() else "kernel setup"
        raise RuntimeError(f"Failed: {failed}\n{exc}\nOutputs: {run_dir}") from exc
    print(f"Pipeline complete. Outputs: {run_dir}")
    if display_mode != "console":
        from IPython.display import Markdown, display
        display(Markdown(report_index_markdown(run_dir)))
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=("fetch", "raw"), required=True)
    parser.add_argument("--reuse-forecasts", action="store_true",
                        help="Skip b0-b3 and evaluate saved expanding/rolling forecasts; not a fresh model fit.")
    parser.add_argument("--no-fit-cache", action="store_true",
                        help="Disable GARCH fit-cache reads and writes for a fresh fit.")
    parser.add_argument("--include-data-checks", action="store_true",
                        help="Run both online FRED/ALFRED diagnostic notebooks; requires FRED_API_KEY.")
    parser.add_argument("--dry-run", action="store_true", help="Print the ordered files without executing or writing.")
    parser.add_argument("--run-dir", type=Path,
                        help="New directory for executed notebooks and script logs (default: supporting/pipeline_runs/<UTC>).")
    args = parser.parse_args(argv)
    if args.reuse_forecasts and args.no_fit_cache:
        parser.error("--no-fit-cache has no effect with --reuse-forecasts")
    if args.dry_run:
        steps = pipeline_steps(args.reuse_forecasts, args.include_data_checks)
        for index, filename in enumerate(steps, 1):
            print(f"[{index}/{len(steps)}] {filename}")
        return
    try:
        run_pipeline(start=args.start, reuse_forecasts=args.reuse_forecasts,
                     use_fit_cache=not args.no_fit_cache, include_data_checks=args.include_data_checks,
                     run_dir=args.run_dir)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
