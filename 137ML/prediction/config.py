"""
config.py — Central configuration for the Traffic Prediction module.

All paths, feature lists, model hyperparameters, and formula constants live here.
Other modules import from this file; nothing is hardcoded elsewhere.

CONGESTION FORMULA
------------------
    congestion = 1 - (v_current / v_freeflow)
    clipped to [0, 1]

Edge-case handling:
  * v_freeflow == 0 or missing: we use DEFAULT_FREEFLOW_SPEED (below).
    Rationale: a zero free-flow speed is a data quality issue; falling back to a
    domain-typical urban speed (50 km/h) is safer than producing NaN for the
    entire edge and losing its contribution to downstream optimisation.
    Document this assumption in any evaluation report that references congestion scores.
  * v_current > v_freeflow: congestion clips to 0 (no congestion, car moving faster
    than historical free-flow — possible during low-traffic periods).

LEAKAGE PREVENTION
------------------
Chronological split only (train / val / test by time boundary, never random shuffle).
Lag and rolling features are computed with strict pandas shift() so that row t only
ever sees data from t-k, t-k-1, ... Rows at the start of the window that would require
future data are dropped (NaN-dropped) from the training set.
"""

from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
ARTIFACT_DIR = BASE_DIR / "artifacts"

SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_traffic.csv"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_model.pkl"
GNN_MODEL_PATH = MODEL_DIR / "gnn_model.pt"
EVAL_RESULTS_PATH = ARTIFACT_DIR / "eval_results.json"

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "timestamp",
    "edge_id",
    "vehicle_count",
    "speed",
    "occupancy",
    "travel_time",
]

OPTIONAL_COLUMNS: List[str] = [
    "weather",
    "event_indicator",
    "day_of_week",
    "hour",
    "incident_flag",
]

TARGET_COLUMN = "travel_time"

# ---------------------------------------------------------------------------
# Prediction horizons (minutes)
# ---------------------------------------------------------------------------
HORIZONS: List[int] = [5, 10, 15]  # T+5, T+10, T+15

# ---------------------------------------------------------------------------
# Congestion formula constants
# ---------------------------------------------------------------------------
DEFAULT_FREEFLOW_SPEED: float = 50.0  # km/h fallback when per-edge value is missing

# ---------------------------------------------------------------------------
# Feature engineering knobs
# ---------------------------------------------------------------------------
# Lag windows (in time-steps, i.e., multiples of the data resolution).
# If data is 1-min granularity: lag=5 means speed 5 minutes ago.
LAG_WINDOWS: List[int] = [1, 5, 10, 15, 20]

# Rolling statistics window size (time-steps)
ROLLING_WINDOW: int = 10

# Cyclical time encoding: encode hour-of-day as sin/cos pair.
# Day-of-week encoded as one-hot (7 columns).
TIME_ENCODING = "cyclic_hour_onehot_dow"

# ---------------------------------------------------------------------------
# Train / val / test split fractions (chronological)
# ---------------------------------------------------------------------------
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# ---------------------------------------------------------------------------
# Baseline model hyperparameters
# ---------------------------------------------------------------------------
BASELINE_HYPERPARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 4,
        "n_jobs": -1,
        "random_state": 42,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "tree_method": "hist",
    },
    "lightgbm": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "verbosity": -1,
    },
}

# Which baseline backend to prefer (fallback chain: preferred -> rf)
PREFERRED_BASELINE: str = "lightgbm"  # "random_forest" | "xgboost" | "lightgbm"

# ---------------------------------------------------------------------------
# ST-GNN hyperparameters
# ---------------------------------------------------------------------------
GNN_HYPERPARAMS = {
    "hidden_dim": 64,
    "gnn_layers": 2,
    "temporal_hidden": 64,
    "dropout": 0.2,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "sequence_length": 12,   # number of past time-steps fed to GRU
}

# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
ANOMALY_Z_THRESHOLD: float = 2.0   # |z-score| above which is flagged as anomaly
ANOMALY_SEVERITY_CLIP: float = 1.0  # clamp severity to [0, ANOMALY_SEVERITY_CLIP]

# ---------------------------------------------------------------------------
# Synthetic data generation (demo only -- not for production)
# ---------------------------------------------------------------------------
SYNTHETIC_NUM_EDGES: int = 20
SYNTHETIC_DURATION_HOURS: int = 72   # 3 days
SYNTHETIC_FREQ_MINUTES: int = 1      # 1-minute resolution
SYNTHETIC_RANDOM_SEED: int = 42
