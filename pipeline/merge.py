"""
pipeline/merge.py — Loads all 8 Home Credit CSV files, validates them, and
merges them onto application_train using SK_ID_CURR with aggregation.

One generic aggregation function is reused for every auxiliary table
(bureau, bureau_balance, previous_application, POS_CASH_balance,
installments_payments, credit_card_balance) instead of writing bespoke
code per file — this keeps the logic easy to explain and to extend.
"""

import os
import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# 1. Load all 8 CSVs automatically
# ---------------------------------------------------------------------------
def load_raw_tables() -> tuple[dict, list]:
    """Reads every file listed in config.RAW_FILES from data/raw/.
    The two application_* files are always read in full; the much larger
    auxiliary tables are capped at config.MAX_AUX_ROWS rows so the pipeline
    finishes in a reasonable time on a laptop. Returns (tables, missing_files).
    """
    tables, missing = {}, []
    for key, filename in config.RAW_FILES.items():
        path = os.path.join(config.DATA_RAW_DIR, filename)
        if not os.path.exists(path):
            missing.append(filename)
            continue
        nrows = None if key.startswith("application") else config.MAX_AUX_ROWS
        tables[key] = pd.read_csv(path, nrows=nrows)
    return tables, missing


# ---------------------------------------------------------------------------
# 2. Validate — a lightweight data-quality report per table
# ---------------------------------------------------------------------------
def validate_tables(tables: dict) -> dict:
    report = {}
    for name, df in tables.items():
        report[name] = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_values": int(df.isnull().sum().sum()),
            "missing_pct": round(float(df.isnull().mean().mean() * 100), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "has_id_column": config.ID_COLUMN in df.columns,
        }
    return report


# ---------------------------------------------------------------------------
# 3. Generic aggregation: numeric -> mean/sum/max/min, categorical -> counts
# ---------------------------------------------------------------------------
def aggregate_table(df: pd.DataFrame, group_col: str, prefix: str,
                     drop_cols: tuple = ()) -> pd.DataFrame:
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != group_col]
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    pieces = []
    if numeric_cols:
        agg_numeric = df.groupby(group_col)[numeric_cols].agg(["mean", "sum", "max", "min"])
        agg_numeric.columns = [f"{prefix}_{col}_{stat}".upper() for col, stat in agg_numeric.columns]
        pieces.append(agg_numeric)

    if categorical_cols:
        dummies = pd.get_dummies(df[[group_col] + categorical_cols], columns=categorical_cols)
        agg_cat = dummies.groupby(group_col).sum()
        agg_cat.columns = [f"{prefix}_{col}".upper() for col in agg_cat.columns]
        pieces.append(agg_cat)

    count = df.groupby(group_col).size().rename(f"{prefix}_COUNT".upper())
    pieces.append(count)

    result = pieces[0]
    for piece in pieces[1:]:
        result = result.join(piece, how="outer")
    return result.reset_index()


# ---------------------------------------------------------------------------
# 4. Merge everything onto application_train via SK_ID_CURR
# ---------------------------------------------------------------------------
def merge_all_tables(tables: dict) -> tuple[pd.DataFrame, dict]:
    if "application_train" not in tables:
        raise FileNotFoundError(
            f"'{config.RAW_FILES['application_train']}' not found in {config.DATA_RAW_DIR}"
        )

    app = tables["application_train"].copy()
    overview = {name: tuple(df.shape) for name, df in tables.items()}
    overview["application_columns_before_merge"] = app.shape[1]

    # bureau_balance -> bureau (via SK_ID_BUREAU), then bureau -> application (via SK_ID_CURR)
    bureau = tables.get("bureau")
    bureau_balance = tables.get("bureau_balance")
    if bureau is not None and bureau_balance is not None:
        bb_agg = aggregate_table(bureau_balance, config.BUREAU_ID_COLUMN, "BB")
        bureau = bureau.merge(bb_agg, on=config.BUREAU_ID_COLUMN, how="left")
    if bureau is not None:
        bureau_agg = aggregate_table(bureau, config.ID_COLUMN, "BUREAU",
                                      drop_cols=(config.BUREAU_ID_COLUMN,))
        app = app.merge(bureau_agg, on=config.ID_COLUMN, how="left")

    # previous_application, POS_CASH_balance, installments_payments, credit_card_balance
    # all key off SK_ID_CURR directly (dropping the per-loan SK_ID_PREV id).
    aux_specs = [
        ("previous_application", "PREV"),
        ("pos_cash_balance", "POS"),
        ("installments_payments", "INSTAL"),
        ("credit_card_balance", "CC"),
    ]
    for table_key, prefix in aux_specs:
        table = tables.get(table_key)
        if table is not None:
            agg = aggregate_table(table, config.ID_COLUMN, prefix,
                                   drop_cols=(config.PREV_ID_COLUMN,))
            app = app.merge(agg, on=config.ID_COLUMN, how="left")

    overview["merged_shape"] = tuple(app.shape)
    return app, overview
