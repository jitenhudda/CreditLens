"""
config.py — Single source of truth for paths, constants, and settings.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Raw Home Credit Default Risk tables (all 8 files, loaded automatically)
# ---------------------------------------------------------------------------
RAW_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "installments_payments": "installments_payments.csv",
    "credit_card_balance": "credit_card_balance.csv",
}

# Artifacts produced by the pipeline
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.pkl")
MERGED_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "merged_sample.csv")

# ---------------------------------------------------------------------------
# Dataset schema (Home Credit Default Risk)
# ---------------------------------------------------------------------------
ID_COLUMN = "SK_ID_CURR"
BUREAU_ID_COLUMN = "SK_ID_BUREAU"
PREV_ID_COLUMN = "SK_ID_PREV"
TARGET_COLUMN = "TARGET"

# Sentinel used by Home Credit for "no employment record" — must become NaN.
DAYS_EMPLOYED_ANOMALY = 365243

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_K_FEATURES = 30
MAX_TRAIN_ROWS = 50_000       # caps the main application table before merging
MAX_AUX_ROWS = 2_000_000      # caps very large auxiliary tables (bureau_balance, installments, etc.)

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest"]

# Small, fast grids — enough to demonstrate tuning without a long runtime.
PARAM_GRIDS = {
    "logistic_regression": {"clf__C": [0.1, 1.0, 10.0]},
    "decision_tree": {"max_depth": [4, 8, 12]},
    "random_forest": {"n_estimators": [150, 300], "max_depth": [6, 10]},
}
