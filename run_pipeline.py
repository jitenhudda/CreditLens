"""
run_pipeline.py — End-to-end CreditLens pipeline.

Run:
    python run_pipeline.py

Steps: load all 8 raw CSVs -> validate -> clean -> merge (SK_ID_CURR) ->
feature engineering -> preprocessing -> feature selection -> train
Logistic Regression / Decision Tree / Random Forest -> compare -> save
the best model + all dashboard artifacts into models/.
"""

import numpy as np
import pandas as pd
import subprocess
import sys
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import config
from pipeline.merge import load_raw_tables, validate_tables, merge_all_tables
from utils import (
    print_step, save_pickle, clean_data, Preprocessor,
    get_feature_importance, compute_shap_importance,
    build_eda_summary, build_feature_engineering_summary,
)


def main():
    # -----------------------------------------------------------------
    print_step("Loading Raw Data (8 CSV files)")
    tables, missing = load_raw_tables()
    if "application_train" not in tables:
        raise FileNotFoundError(
            f"application_train.csv not found in {config.DATA_RAW_DIR}. "
            "Place all 8 Home Credit CSV files there and re-run."
        )
    for name, df in tables.items():
        print(f"{name:<25} {df.shape[0]:>10,} rows  x  {df.shape[1]:>3} cols")
    if missing:
        print(f"Missing (skipped): {missing}")

    # -----------------------------------------------------------------
    print_step("Data Validation")
    data_quality = validate_tables(tables)
    for name, stats in data_quality.items():
        print(f"{name:<25} missing={stats['missing_pct']:>5.2f}%  "
              f"duplicates={stats['duplicate_rows']:<8} id_col={stats['has_id_column']}")

    # -----------------------------------------------------------------
    print_step("Merging Tables on SK_ID_CURR")
    merged_df, dataset_overview = merge_all_tables(tables)
    print(f"Merged shape: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} columns")
    n_after_merge = merged_df.shape[1]

    # -----------------------------------------------------------------
    print_step("Target Validation")
    assert config.TARGET_COLUMN in merged_df.columns, f"Missing required column: {config.TARGET_COLUMN}"
    missing_target = merged_df[config.TARGET_COLUMN].isnull().sum()
    merged_df = merged_df.dropna(subset=[config.TARGET_COLUMN])
    print(f"Rows with missing TARGET dropped: {missing_target}")
    print(f"Target balance -> Default: {merged_df[config.TARGET_COLUMN].mean()*100:.2f}% | "
          f"Non-default: {(1 - merged_df[config.TARGET_COLUMN].mean())*100:.2f}%")

    if len(merged_df) > config.MAX_TRAIN_ROWS:
        merged_df = merged_df.sample(config.MAX_TRAIN_ROWS, random_state=config.RANDOM_STATE)
        print(f"Sampled down to {config.MAX_TRAIN_ROWS:,} rows to keep training fast.")

    # -----------------------------------------------------------------
    print_step("Data Cleaning")
    before_rows = len(merged_df)
    cleaned_df = clean_data(merged_df)
    print(f"Duplicate rows removed: {before_rows - len(cleaned_df)}")
    print("Anomalous DAYS_EMPLOYED sentinel (365243) converted to NaN")
    print("DAYS_* columns normalized to positive values")

    y = cleaned_df[config.TARGET_COLUMN].astype(int)

    # -----------------------------------------------------------------
    print_step("Exploratory Data Analysis")
    eda_summary = build_eda_summary(cleaned_df)
    print("Top correlated features with TARGET:")
    print(pd.Series(eda_summary["top_correlations_with_target"]).head(10).to_string())

    # -----------------------------------------------------------------
    print_step("Missing Value Handling")
    print("Numeric columns will be imputed with median; categorical with mode")
    print("Columns with more than 50% missing values will be dropped")

    print_step("Feature Engineering")
    print("Engineered ratios/composites from: Income, Loan Amount, Existing Debt,")
    print("Credit History, Payment History, Employment History, External Credit Scores")
    n_after_engineering = cleaned_df.shape[1]  # Preprocessor engineers internally too; approximate count below

    preprocessor = Preprocessor(max_missing_ratio=0.5)
    X_full = preprocessor.fit(cleaned_df)
    n_after_encoding = X_full.shape[1]
    print(f"Feature matrix after cleaning + engineering + encoding: {X_full.shape[1]} columns")

    # -----------------------------------------------------------------
    print_step("Feature Selection")
    ranker = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1)
    ranker.fit(X_full, y)
    importances = pd.Series(ranker.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    top_k = min(config.TOP_K_FEATURES, len(importances))
    selected_features = importances.head(top_k).index.tolist()
    preprocessor.set_selected_features(selected_features)
    X = X_full[selected_features]
    print(f"Selected top {top_k} features by importance:")
    print(importances.head(10).round(4).to_string())

    feature_engineering_summary = build_feature_engineering_summary(
        n_before_merge=dataset_overview["application_columns_before_merge"],
        n_after_merge=n_after_merge,
        n_after_engineering=n_after_engineering,
        n_after_encoding=n_after_encoding,
        n_selected=top_k,
    )

    # -----------------------------------------------------------------
    print_step("Train/Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    # -----------------------------------------------------------------
    base_models = {}

    print_step("Training Logistic Regression")
    base_models["logistic_regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE)),
    ]).fit(X_train, y_train)
    print("Logistic Regression trained.")

    print_step("Training Decision Tree")
    base_models["decision_tree"] = DecisionTreeClassifier(
        max_depth=8, class_weight="balanced", random_state=config.RANDOM_STATE
    ).fit(X_train, y_train)
    print("Decision Tree trained.")

    print_step("Training Random Forest")
    base_models["random_forest"] = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=config.RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    print("Random Forest trained.")

    # -----------------------------------------------------------------
    print_step("Cross Validation")
    for name, model in base_models.items():
        scores = cross_val_score(model, X_train, y_train, cv=config.CV_FOLDS, scoring="roc_auc")
        print(f"{name:<20} mean ROC-AUC = {scores.mean():.4f}  (+/- {scores.std():.4f})")

    # -----------------------------------------------------------------
    print_step("Hyperparameter Tuning")
    tuned_models = {}
    for name, model in base_models.items():
        grid = config.PARAM_GRIDS[name]
        search = GridSearchCV(model, grid, cv=3, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        tuned_models[name] = search.best_estimator_
        print(f"{name:<20} best params = {search.best_params_}  best CV ROC-AUC = {search.best_score_:.4f}")

    # -----------------------------------------------------------------
    print_step("Model Comparison")
    results = {}
    for name, model in tuned_models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
    comparison_df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
    print(comparison_df.round(4).to_string())

    best_name = comparison_df.index[0]
    best_model = tuned_models[best_name]
    best_metrics = results[best_name]
    print(f"\nBest model selected: {best_name}")

    # -----------------------------------------------------------------
    print_step("Accuracy");   print(f"{best_metrics['accuracy']:.4f}")
    print_step("Precision");  print(f"{best_metrics['precision']:.4f}")
    print_step("Recall");     print(f"{best_metrics['recall']:.4f}")
    print_step("F1 Score");   print(f"{best_metrics['f1']:.4f}")
    print_step("ROC-AUC");    print(f"{best_metrics['roc_auc']:.4f}")

    # -----------------------------------------------------------------
    print_step("Confusion Matrix")
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    print(cm)

    # -----------------------------------------------------------------
    print_step("Feature Importance")
    feature_importance = get_feature_importance(best_model, X.columns)
    print(feature_importance.head(15).round(4).to_string())

    # -----------------------------------------------------------------
    print_step("SHAP Explainability")
    shap_importance = None
    try:
        sample = X_test.sample(min(200, len(X_test)), random_state=config.RANDOM_STATE)
        shap_importance, _, _ = compute_shap_importance(best_model, X_train, sample)
        print(shap_importance.head(10).round(4).to_string())
    except Exception as e:
        print(f"SHAP computation skipped ({e})")

    # -----------------------------------------------------------------
    print_step("Saving Artifacts")
    metrics_to_save = {
        "best_model_name": best_name,
        "best_metrics": best_metrics,
        "comparison": comparison_df.to_dict(orient="index"),
        "confusion_matrix": cm.tolist(),
        "feature_importance": feature_importance.to_dict(),
        "shap_importance": shap_importance.to_dict() if shap_importance is not None else {},
        "shap_background": X_train.sample(min(200, len(X_train)), random_state=config.RANDOM_STATE),
        "dataset_overview": dataset_overview,
        "data_quality": data_quality,
        "eda_summary": eda_summary,
        "feature_engineering_summary": feature_engineering_summary,
    }

    save_pickle(best_model, config.MODEL_PATH)
    save_pickle(preprocessor, config.PREPROCESSOR_PATH)
    save_pickle(selected_features, config.FEATURE_NAMES_PATH)
    save_pickle(metrics_to_save, config.METRICS_PATH)

    # A small merged-data sample is kept in outputs/ purely for manual inspection.
    cleaned_df.sample(min(500, len(cleaned_df)), random_state=config.RANDOM_STATE).to_csv(
        config.MERGED_SAMPLE_PATH, index=False
    )

    print(f"Saved: {config.MODEL_PATH}")
    print(f"Saved: {config.PREPROCESSOR_PATH}")
    print(f"Saved: {config.FEATURE_NAMES_PATH}")
    print(f"Saved: {config.METRICS_PATH}")
    print(f"Saved: {config.MERGED_SAMPLE_PATH}")


if __name__ == "__main__":
    try:
        main()

        print("\n" + "=" * 60)
        print("CreditLens Pipeline Completed Successfully")
        print("Launching Streamlit Dashboard...")
        print("=" * 60)

        streamlit_exe = shutil.which("streamlit")

        if streamlit_exe:
            subprocess.run([streamlit_exe, "run", "app.py"])
        else:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")

    except Exception as e:
        print(f"\nPipeline failed:\n{e}")