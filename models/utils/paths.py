"""Paths shared by active model notebooks and scripts; rooted at this checkout."""
from pathlib import Path


def find_repo_root(start=None):
    """Find the checkout from an explicit path, or this module's own location."""
    path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if ((candidate / "models" / "research_config.py").is_file()
                and (candidate / "dataset").is_dir()):
            return candidate
    raise FileNotFoundError(f"Cannot locate project root from {path}")


PROJECT_ROOT = find_repo_root()
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
FORECAST_DIR = MODELS_DIR / "forecasting" / "forecasts"
FORECAST_RESULTS_DIR = MODELS_DIR / "forecasting" / "results"
PORTFOLIO_RESULTS_DIR = MODELS_DIR / "portfolio" / "results"
