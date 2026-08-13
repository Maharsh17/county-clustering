"""Configuration, loaded once from config/config.yaml.

Nothing here is tunable in code. Change config/config.yaml instead, which is
also what makes K arguable rather than hardcoded.
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())

FEATS = CONFIG["features"]          # column -> display name, order fixes every figure
COLS = list(FEATS)
SVI_ONLY = [c for c in COLS if c != "MOBILITY"]
K = CONFIG["k"]
SEED = CONFIG["random_state"]
NAMES = CONFIG["names"]
POP_FLOOR = CONFIG["population_floor"]
TEST_SIZE = CONFIG["test_size"]
CV_FOLDS = CONFIG["cv_folds"]
N_ESTIMATORS = CONFIG["n_estimators"]

DATA = ROOT / CONFIG["paths"]["data"]
FIGURES = ROOT / CONFIG["paths"]["figures"]
OUTPUTS = ROOT / CONFIG["paths"]["outputs"]

LABELS = [f"Type {i+1}\n{NAMES[i]}" for i in range(K)]
INK, MUT = "#0b0b0b", "#8a8a86"
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4"]
PAL_APP = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#008300", "#e87ba4"]
GEOJSON = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
SOURCE = "https://github.com/Maharsh17/county-clustering"
BASELINE_SIL = 0.155   # all 17 measures at K=4, from scripts/train.py
