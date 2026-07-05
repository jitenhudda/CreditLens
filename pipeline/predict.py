"""
pipeline/predict.py — Loads the trained artifacts and scores new data.

This module does NOT train anything. It assumes run_pipeline.py has already
produced models/best_model.pkl, preprocessor.pkl, feature_names.pkl, metrics.pkl.
"""

import os
import pandas as pd

import config
from utils import load_pickle


def load_artifacts():
    """Loads model + preprocessor + metrics once. app.py caches this call."""
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run `python run_pipeline.py` first."
        )
    model = load_pickle(config.MODEL_PATH)
    preprocessor = load_pickle(config.PREPROCESSOR_PATH)
    feature_names = load_pickle(config.FEATURE_NAMES_PATH)
    metrics = load_pickle(config.METRICS_PATH) if os.path.exists(config.METRICS_PATH) else {}
    return model, preprocessor, feature_names, metrics


def predict_dataframe(df: pd.DataFrame, model, preprocessor) -> pd.DataFrame:
    """Scores an in-memory DataFrame (one row or many)."""
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[:, 1]
    result = df.copy().reset_index(drop=True)
    result["default_probability"] = proba
    result["prediction"] = (proba >= 0.5).astype(int)
    result["risk_score"] = (proba * 1000).round(0).astype(int)
    return result


def predict_single(input_dict: dict, model, preprocessor) -> dict:
    df = pd.DataFrame([input_dict])
    result = predict_dataframe(df, model, preprocessor)
    return result.iloc[0].to_dict()
