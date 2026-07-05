"""
utils.py — Common helper functions shared by run_pipeline.py and
pipeline/predict.py. Keeping this logic in one place means training
and inference can never drift apart.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

import config


# ---------------------------------------------------------------------------
# Console output helpers (used to print the required step-by-step log)
# ---------------------------------------------------------------------------
def print_step(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_pickle(obj, path: str):
    joblib.dump(obj, path)


def load_pickle(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Model helpers — logistic regression is wrapped in a Pipeline (StandardScaler
# + classifier) for stable convergence. These helpers let run_pipeline.py,
# pipeline/predict.py, and app.py all handle "plain model or Pipeline"
# the same way instead of duplicating the unwrap logic.
# ---------------------------------------------------------------------------
def get_estimator(model):
    """Returns the final estimator, unwrapping an sklearn Pipeline if needed."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def split_pipeline(model):
    """Returns (preprocessing_pipeline_or_None, final_estimator)."""
    if isinstance(model, Pipeline) and len(model.steps) > 1:
        return Pipeline(model.steps[:-1]), model.steps[-1][1]
    return None, get_estimator(model)


def get_feature_importance(model, feature_names) -> pd.Series:
    estimator = get_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.abs(estimator.coef_[0])
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def compute_shap_importance(model, background_df: pd.DataFrame, sample_df: pd.DataFrame):
    """Returns a feature -> mean(|SHAP value|) Series, handling both tree and
    linear models and normalizing the different array shapes SHAP can return
    (list-of-arrays, or a 3-D array with a class axis) into one 2-D matrix."""
    import shap

    pre, estimator = split_pipeline(model)
    if pre is not None:
        background_df = pd.DataFrame(pre.transform(background_df), columns=background_df.columns)
        sample_df = pd.DataFrame(pre.transform(sample_df), columns=sample_df.columns)

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.LinearExplainer(estimator, background_df)

    raw_values = explainer.shap_values(sample_df)
    values = _normalize_shap_values(raw_values)
    importance = pd.Series(np.abs(values).mean(axis=0), index=sample_df.columns).sort_values(ascending=False)
    return importance, explainer, pre


def _normalize_shap_values(raw_values) -> np.ndarray:
    """Collapses SHAP's various output shapes down to a plain (n_samples, n_features)
    array for the positive class, regardless of shap library version."""
    if isinstance(raw_values, list):
        values = raw_values[1] if len(raw_values) > 1 else raw_values[0]
    else:
        values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def explain_instance(model, background_df: pd.DataFrame, row_df: pd.DataFrame) -> pd.Series:
    """Local SHAP explanation for one row -> Series of feature -> signed SHAP value."""
    import shap

    pre, estimator = split_pipeline(model)
    bg, row = background_df, row_df
    if pre is not None:
        bg = pd.DataFrame(pre.transform(background_df), columns=background_df.columns)
        row = pd.DataFrame(pre.transform(row_df), columns=row_df.columns)

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
    else:
        explainer = shap.LinearExplainer(estimator, bg)

    values = _normalize_shap_values(explainer.shap_values(row))
    return pd.Series(values[0], index=row.columns)


# ---------------------------------------------------------------------------
# Cleaning (row/value-level fixes — safe to apply identically at train & predict)
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(config.DAYS_EMPLOYED_ANOMALY, np.nan)

    for col in ("DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"):
        if col in df.columns:
            df[col] = df[col].abs()

    # Duplicate rows and impossible negative amounts are dropped only if present.
    df = df.drop_duplicates()
    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"):
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Feature engineering — Income, Loan Amount, Existing Debt, Credit History,
# Payment History, Employment History, External Credit Scores.
# ---------------------------------------------------------------------------
ENGINEERED_FEATURE_NOTES = {
    "CREDIT_INCOME_RATIO": "Loan amount relative to annual income",
    "ANNUITY_INCOME_RATIO": "Monthly annuity burden relative to annual income",
    "CREDIT_TERM": "Annuity relative to total credit (implied loan term)",
    "GOODS_PRICE_CREDIT_RATIO": "Price of goods relative to credit granted",
    "AGE_YEARS": "Applicant age in years",
    "CREDIT_HISTORY_YEARS": "Years since ID document was published (credit-file age proxy)",
    "EMPLOYED_YEARS": "Years employed",
    "EMPLOYED_AGE_RATIO": "Employment length relative to age",
    "PAYMENT_RELIABILITY_SCORE": "1 - (social-circle defaults / observations), higher is better",
    "EXT_SOURCE_MEAN": "Mean of the three external credit bureau scores",
    "EXT_SOURCE_MAX": "Max of the three external credit bureau scores",
    "EXT_SOURCE_MIN": "Min of the three external credit bureau scores",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Income & Loan Amount ---
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(df.columns):
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    # --- Existing Debt (proxy: annuity burden relative to income) ---
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(df.columns):
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(df.columns):
        df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"} <= set(df.columns):
        df["GOODS_PRICE_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"].replace(0, np.nan)

    # --- Credit History (age of the credit file as a proxy) ---
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = df["DAYS_BIRTH"] / 365.25
    if "DAYS_ID_PUBLISH" in df.columns:
        df["CREDIT_HISTORY_YEARS"] = df["DAYS_ID_PUBLISH"] / 365.25

    # --- Employment History ---
    if "DAYS_EMPLOYED" in df.columns:
        df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"] / 365.25
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"} <= set(df.columns):
        df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

    # --- Payment History (proxy: social-circle default rate, if present) ---
    if {"DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"} <= set(df.columns):
        df["PAYMENT_RELIABILITY_SCORE"] = 1 - (
            df["DEF_30_CNT_SOCIAL_CIRCLE"] / (df["OBS_30_CNT_SOCIAL_CIRCLE"] + 1)
        )

    # --- External Credit Scores ---
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MAX"] = df[ext_cols].max(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)

    return df


# ---------------------------------------------------------------------------
# Preprocessor — one object, fit once during training, reused unchanged
# during single prediction. This is what gets saved as preprocessor.pkl
# ---------------------------------------------------------------------------
class Preprocessor:
    """Cleans, engineers, imputes, encodes, and aligns columns.
    fit() is called once on training data; transform() is called on any
    new data (a single customer row) and always produces the exact
    numeric matrix the model expects.
    """

    def __init__(self, max_missing_ratio: float = 0.5):
        self.max_missing_ratio = max_missing_ratio
        self.keep_columns_ = None
        self.numeric_medians_ = None
        self.categorical_modes_ = None
        self.categorical_columns_ = None
        self.dummy_columns_ = None   # full one-hot column set learned at fit time
        self.feature_names_ = None   # final selected feature list (set after feature selection)

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = clean_data(df)
        df = engineer_features(df)
        return df

    def fit(self, df: pd.DataFrame):
        df = self._base_transform(df)
        if config.TARGET_COLUMN in df.columns:
            df = df.drop(columns=[config.TARGET_COLUMN])
        if config.ID_COLUMN in df.columns:
            df = df.drop(columns=[config.ID_COLUMN])

        missing_ratio = df.isnull().mean()
        self.keep_columns_ = missing_ratio[missing_ratio <= self.max_missing_ratio].index.tolist()
        df = df[self.keep_columns_]

        self.categorical_columns_ = df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_columns = [c for c in df.columns if c not in self.categorical_columns_]

        self.numeric_medians_ = df[numeric_columns].median(numeric_only=True)
        self.categorical_modes_ = {
            c: (df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown")
            for c in self.categorical_columns_
        }

        df[numeric_columns] = df[numeric_columns].fillna(self.numeric_medians_)
        for c in self.categorical_columns_:
            df[c] = df[c].fillna(self.categorical_modes_[c])

        df = pd.get_dummies(df, columns=self.categorical_columns_, dummy_na=False)
        self.dummy_columns_ = df.columns.tolist()
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._base_transform(df)
        for col in (config.TARGET_COLUMN, config.ID_COLUMN):
            if col in df.columns:
                df = df.drop(columns=[col])

        df = df.reindex(columns=self.keep_columns_)

        numeric_columns = [c for c in self.keep_columns_ if c not in self.categorical_columns_]
        df[numeric_columns] = df[numeric_columns].fillna(self.numeric_medians_)
        for c in self.categorical_columns_:
            df[c] = df[c].fillna(self.categorical_modes_[c])

        df = pd.get_dummies(df, columns=self.categorical_columns_, dummy_na=False)
        df = df.reindex(columns=self.dummy_columns_, fill_value=0)

        if self.feature_names_ is not None:
            df = df.reindex(columns=self.feature_names_, fill_value=0)

        return df.apply(pd.to_numeric, errors="coerce").fillna(0)

    def set_selected_features(self, feature_names: list):
        """Called once by run_pipeline.py after feature selection so that
        every future transform() already returns the final, model-ready matrix."""
        self.feature_names_ = feature_names


# ---------------------------------------------------------------------------
# Dashboard support — lightweight, pickle-friendly summaries computed once
# during training and displayed by app.py (which never touches raw data).
# ---------------------------------------------------------------------------
def build_eda_summary(df: pd.DataFrame, target_col: str = config.TARGET_COLUMN) -> dict:
    numeric_df = df.select_dtypes(include=np.number)

    correlations = {}
    if target_col in numeric_df.columns:
        corr = numeric_df.corr(numeric_only=True)[target_col].drop(target_col)
        correlations = corr.abs().sort_values(ascending=False).head(15).round(4).to_dict()

    key_numeric_cols = [c for c in
                         ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AGE_YEARS", "EMPLOYED_YEARS")
                         if c in df.columns]
    numeric_summary = df[key_numeric_cols].describe().round(2).to_dict() if key_numeric_cols else {}

    categorical_distributions = {}
    for col in ("NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"):
        if col in df.columns:
            categorical_distributions[col] = df[col].value_counts(normalize=True).round(4).to_dict()

    target_distribution = df[target_col].value_counts(normalize=True).round(4).to_dict() \
        if target_col in df.columns else {}

    return {
        "top_correlations_with_target": correlations,
        "numeric_summary": numeric_summary,
        "categorical_distributions": categorical_distributions,
        "target_distribution": target_distribution,
    }


def build_feature_engineering_summary(n_before_merge, n_after_merge, n_after_engineering,
                                       n_after_encoding, n_selected) -> dict:
    return {
        "columns_original_application_table": n_before_merge,
        "columns_after_merging_aux_tables": n_after_merge,
        "columns_after_feature_engineering": n_after_engineering,
        "columns_after_encoding": n_after_encoding,
        "columns_after_feature_selection": n_selected,
        "engineered_features": ENGINEERED_FEATURE_NOTES,
    }
