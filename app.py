"""
app.py — CreditLens Streamlit dashboard.

Run:
    streamlit run app.py

Loads only the artifacts saved by run_pipeline.py (cached, loaded once).
This app never retrains and never accepts a raw CSV upload — it is a
pure read-only view over models/*.pkl.
"""

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from pipeline.predict import load_artifacts, predict_single
from utils import explain_instance

st.set_page_config(page_title="CreditLens", page_icon="🏦", layout="wide")

NAVY, GOLD, GREEN, RED, AMBER, WHITE = "#0B1E3D", "#C9A24B", "#2ECC71", "#E74C3C", "#F5A623", "#F5F7FA"

st.markdown(f"""
<style>
.stApp {{ background: radial-gradient(circle at top left, #132A52 0%, {NAVY} 55%, #060F22 100%); color: {WHITE}; }}
section[data-testid="stSidebar"] {{ background: #081530; }}
section[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
h1, h2, h3 {{ color: {WHITE} !important; }}
div[data-testid="stMetric"] {{ background: #101F3A; border: 1px solid #22355C; border-radius: 10px; padding: 10px; }}
.stButton>button {{ background: {GOLD}; color: {NAVY}; font-weight: 700; border: none; border-radius: 8px; }}
.badge {{ display:inline-block; padding:6px 16px; border-radius:20px; font-weight:700; }}
.hero {{ text-align:center; padding: 36px 20px 28px 20px; }}
.hero h1 {{ font-size: 3rem; margin-bottom: 6px; }}
.hero p {{ color: #AEB9CF; font-size: 1.15rem; max-width: 760px; margin: 0 auto; }}
.kpi-card {{ background: linear-gradient(160deg, #101F3A 0%, #0C1A33 100%); border: 1px solid #22355C;
             border-radius: 14px; padding: 22px 10px; text-align:center; }}
.kpi-card .kpi-icon {{ font-size: 1.8rem; }}
.kpi-card .kpi-value {{ font-size: 1.6rem; font-weight: 800; color: {GOLD}; margin: 6px 0 2px 0; }}
.kpi-card .kpi-label {{ color: #AEB9CF; font-size: 0.9rem; }}
.info-card {{ background: #101F3A; border: 1px solid #22355C; border-radius: 14px; padding: 20px;
              height: 100%; }}
.info-card h4 {{ color: {WHITE} !important; margin-top: 0; }}
.info-card p, .info-card li {{ color: #C7D0E0; font-size: 0.92rem; }}
.workflow-wrap {{ display:flex; align-items:center; justify-content:center; flex-wrap:wrap; gap:8px; }}
.workflow-step {{ background: #101F3A; border: 1px solid {GOLD}; color: {WHITE}; border-radius: 10px;
                   padding: 10px 16px; font-weight:600; font-size: 0.9rem; }}
.workflow-arrow {{ color: {GOLD}; font-size: 1.2rem; }}
.tech-pill {{ display:inline-block; background:#101F3A; border:1px solid #22355C; color:{WHITE};
              border-radius: 20px; padding: 8px 18px; margin: 4px; font-weight:600; font-size:0.88rem; }}
.app-footer {{ margin-top: 48px; padding-top: 18px; border-top: 1px solid #22355C; text-align:center;
               color: #8896B0; font-size: 0.85rem; line-height: 1.6; }}
.app-footer strong {{ color: {WHITE}; }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def get_artifacts():
    return load_artifacts()


st.sidebar.markdown(f"## 🏦 CreditLens")
st.sidebar.caption("AI Credit Risk Intelligence Platform")

try:
    model, preprocessor, feature_names, metrics = get_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python run_pipeline.py` first, then relaunch the app.")
    st.stop()

st.sidebar.success(f"Model loaded: **{metrics.get('best_model_name', 'unknown')}**")
page = st.sidebar.radio("Navigate", [
    "Home", "Dataset Overview", "Data Quality", "EDA", "Feature Engineering Summary",
    "Model Performance", "SHAP Explainability", "Single Customer Prediction", "About Project",
])


def risk_band(p):
    if p < 0.10: return "Excellent", GREEN
    if p < 0.25: return "Good", "#8BC34A"
    if p < 0.45: return "Fair", AMBER
    if p < 0.65: return "Poor", "#E67E22"
    return "High Risk", RED


def gauge(value, title, max_value=100, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix, "font": {"color": WHITE}},
        title={"text": title, "font": {"color": WHITE, "size": 14}},
        gauge={"axis": {"range": [0, max_value]}, "bar": {"color": GOLD},
               "steps": [{"range": [0, max_value*.25], "color": GREEN}, {"range": [max_value*.25, max_value*.5], "color": "#8BC34A"},
                         {"range": [max_value*.5, max_value*.75], "color": AMBER}, {"range": [max_value*.75, max_value], "color": RED}]},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=230, margin=dict(l=20, r=20, t=50, b=10), font={"color": WHITE})
    return fig


def style(fig, height=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={"color": WHITE}, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ===========================================================================
# Home
# ===========================================================================
if page == "Home":
    st.markdown("""
    <div class="hero">
        <h1>🏦 CreditLens</h1>
        <p>An AI-powered Credit Risk Intelligence Platform that scores loan applicants
        using the full Home Credit Default Risk dataset — from raw data to an
        explainable, production-ready prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "📁", "8", "Datasets"),
        (k2, "👥", "246,009", "Customers"),
        (k3, "🧬", "608", "Features"),
        (k4, "🤖", "3", "ML Models"),
    ]
    for col, icon, value, label in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.subheader("How It Works")
    steps = ["Raw Data", "Data Cleaning", "Feature Engineering", "Model Training", "Prediction"]
    workflow_html = '<div class="workflow-wrap">'
    for i, step in enumerate(steps):
        workflow_html += f'<div class="workflow-step">{step}</div>'
        if i < len(steps) - 1:
            workflow_html += '<div class="workflow-arrow">➜</div>'
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)

    st.write("")
    st.subheader("Project Highlights")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("""
        <div class="info-card">
            <h4>🔗 End-to-End Pipeline</h4>
            <p>All 8 Home Credit tables are validated, aggregated, and merged on
            SK_ID_CURR into a single modeling-ready dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Explainable AI</h4>
            <p>Global and per-customer SHAP explanations make every prediction
            transparent and auditable.</p>
        </div>
        """, unsafe_allow_html=True)
    with h3:
        st.markdown("""
        <div class="info-card">
            <h4>⚡ Model Comparison</h4>
            <p>Logistic Regression, Decision Tree, and Random Forest are trained,
            tuned, and compared on Accuracy, Precision, Recall, F1, and ROC-AUC.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Technologies Used")
    tech = ["Python", "Pandas", "Scikit-learn", "Plotly", "Streamlit", "SHAP"]
    st.markdown("".join(f'<span class="tech-pill">{t}</span>' for t in tech), unsafe_allow_html=True)

    st.write("")
    st.write("")
    gcol1, gcol2, gcol3 = st.columns([1, 1, 1])
    with gcol2:
        if st.button("🚀 Get Started", use_container_width=True):
            st.info("👈 Use the sidebar on the left to explore the dashboard.")


# ===========================================================================
# Dataset Overview
# ===========================================================================
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    overview = metrics.get("dataset_overview", {})

    rows = []
    for key, filename in config.RAW_FILES.items():
        shape = overview.get(key)
        if shape:
            rows.append({"Table": key, "Rows": f"{shape[0]:,}", "Columns": shape[1]})
    if rows:
        st.subheader("Raw Tables Loaded (8 files)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    merged_shape = overview.get("merged_shape")
    c1, c2, c3 = st.columns(3)
    c1.metric("Application Columns (before merge)", overview.get("application_columns_before_merge", "-"))
    c2.metric("Merged Rows", f"{merged_shape[0]:,}" if merged_shape else "-")
    c3.metric("Merged Columns", merged_shape[1] if merged_shape else "-")

    st.caption("All 8 Home Credit tables (application_train, application_test, bureau, "
               "bureau_balance, previous_application, POS_CASH_balance, "
               "installments_payments, credit_card_balance) are aggregated to one row "
               "per SK_ID_CURR and merged onto application_train.")


# ===========================================================================
# Data Quality
# ===========================================================================
elif page == "Data Quality":
    st.title("Data Quality Report")
    dq = metrics.get("data_quality", {})
    if dq:
        dq_df = pd.DataFrame(dq).T
        dq_df.index.name = "Table"
        st.dataframe(dq_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Missing Values (%) by Table")
            fig = px.bar(dq_df.reset_index(), x="Table", y="missing_pct", color_discrete_sequence=[GOLD])
            st.plotly_chart(style(fig, 360), use_container_width=True)
        with col2:
            st.subheader("Duplicate Rows by Table")
            fig = px.bar(dq_df.reset_index(), x="Table", y="duplicate_rows", color_discrete_sequence=[RED])
            st.plotly_chart(style(fig, 360), use_container_width=True)
    else:
        st.info("No data-quality report found — retrain with `python run_pipeline.py`.")

    st.subheader("Cleaning Rules Applied")
    st.markdown("""
- `DAYS_EMPLOYED` sentinel value **365243** converted to missing
- `DAYS_*` columns normalized to positive values
- Exact duplicate rows removed
- Negative income / credit / annuity amounts treated as missing
- Columns with more than 50% missing values dropped before modeling
- Remaining numeric gaps filled with the median, categorical gaps with the mode
""")


# ===========================================================================
# EDA
# ===========================================================================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    eda = metrics.get("eda_summary", {})

    st.subheader("Target Distribution")
    target_dist = eda.get("target_distribution", {})
    if target_dist:
        labels = {0: "Non-Default", 1: "Default"}
        dist_df = pd.DataFrame({
            "Class": [labels.get(int(k), k) for k in target_dist.keys()],
            "Share": list(target_dist.values()),
        })
        fig = px.pie(dist_df, names="Class", values="Share", hole=0.5,
                     color_discrete_sequence=[GREEN, RED])
        st.plotly_chart(style(fig, 320), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Features Correlated with Default")
        corr = pd.Series(eda.get("top_correlations_with_target", {})).sort_values()
        if not corr.empty:
            fig = px.bar(x=corr.values, y=corr.index, orientation="h", color_discrete_sequence=[GOLD])
            st.plotly_chart(style(fig, 420), use_container_width=True)
    with col2:
        st.subheader("Categorical Distributions")
        cat_dist = eda.get("categorical_distributions", {})
        if cat_dist:
            chosen = st.selectbox("Column", list(cat_dist.keys()))
            series = pd.Series(cat_dist[chosen])
            fig = px.bar(x=series.index, y=series.values, color_discrete_sequence=[GOLD])
            st.plotly_chart(style(fig, 380), use_container_width=True)

    st.subheader("Key Numeric Columns — Summary Statistics")
    numeric_summary = eda.get("numeric_summary", {})
    if numeric_summary:
        st.dataframe(pd.DataFrame(numeric_summary), use_container_width=True)
    else:
        st.info("No EDA summary found — retrain with `python run_pipeline.py`.")


# ===========================================================================
# Feature Engineering Summary
# ===========================================================================
elif page == "Feature Engineering Summary":
    st.title("Feature Engineering Summary")
    fe = metrics.get("feature_engineering_summary", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Original App Columns", fe.get("columns_original_application_table", "-"))
    c2.metric("After Merging Aux Tables", fe.get("columns_after_merging_aux_tables", "-"))
    c3.metric("After Feature Engineering", fe.get("columns_after_feature_engineering", "-"))
    c4.metric("After Encoding", fe.get("columns_after_encoding", "-"))
    c5.metric("After Feature Selection", fe.get("columns_after_feature_selection", "-"))

    funnel_stages = ["Original", "After Merge", "After Engineering", "After Encoding", "Selected"]
    funnel_values = [
        fe.get("columns_original_application_table", 0), fe.get("columns_after_merging_aux_tables", 0),
        fe.get("columns_after_feature_engineering", 0), fe.get("columns_after_encoding", 0),
        fe.get("columns_after_feature_selection", 0),
    ]
    fig = go.Figure(go.Funnel(y=funnel_stages, x=funnel_values, marker={"color": GOLD}))
    st.plotly_chart(style(fig, 380), use_container_width=True)

    st.subheader("Engineered Features")
    engineered = fe.get("engineered_features", {})
    if engineered:
        st.dataframe(pd.DataFrame(list(engineered.items()), columns=["Feature", "Description"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No feature-engineering summary found — retrain with `python run_pipeline.py`.")


# ===========================================================================
# Model Performance
# ===========================================================================
elif page == "Model Performance":
    st.title("Model Performance")
    m = metrics.get("best_metrics", {})

    st.subheader("Model Leaderboard")
    comp = pd.DataFrame(metrics.get("comparison", {})).T
    st.dataframe(comp.round(4), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
    c2.metric("Precision", f"{m.get('precision', 0):.4f}")
    c3.metric("Recall", f"{m.get('recall', 0):.4f}")
    c4.metric("F1 Score", f"{m.get('f1', 0):.4f}")
    c5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: No Default", "Pred: Default"], y=["Actual: No Default", "Actual: Default"])
        st.plotly_chart(style(fig, 380), use_container_width=True)
    with col2:
        st.subheader("Feature Importance")
        fi = pd.Series(metrics.get("feature_importance", {})).sort_values(ascending=False).head(15)
        fig = px.bar(x=fi.values, y=fi.index, orientation="h", color_discrete_sequence=[GOLD])
        st.plotly_chart(style(fig, 380), use_container_width=True)


# ===========================================================================
# SHAP Explainability
# ===========================================================================
elif page == "SHAP Explainability":
    st.title("SHAP Explainability")

    st.subheader("Global Feature Importance (SHAP)")
    shap_imp = pd.Series(metrics.get("shap_importance", {})).sort_values(ascending=False).head(15)
    if not shap_imp.empty:
        fig = px.bar(x=shap_imp.values, y=shap_imp.index, orientation="h", color_discrete_sequence=[GOLD])
        st.plotly_chart(style(fig, 420), use_container_width=True)
    else:
        st.info("No SHAP data found — retrain with `python run_pipeline.py`.")

    st.subheader("Explain the Last Prediction")
    if "last_row" not in st.session_state:
        st.info("Run a prediction on **Single Customer Prediction** first to see its local explanation here.")
    else:
        background = metrics.get("shap_background")
        row_df = preprocessor.transform(pd.DataFrame([st.session_state["last_row"]]))
        if background is not None:
            contributions = explain_instance(model, background, row_df)
            top = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(15)
            top = top.sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation="h",
                        color=top.values > 0, color_discrete_map={True: RED, False: GREEN})
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 460), use_container_width=True)
            st.caption("Red pushes toward default, green pushes toward non-default.")
        else:
            st.info("No SHAP background sample found — retrain to enable local explanations.")


# ===========================================================================
# Single Customer Prediction
# ===========================================================================
elif page == "Single Customer Prediction":
    st.title("Single Customer Prediction")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            income = st.number_input("Annual Income", 10000.0, 5_000_000.0, 180000.0, step=5000.0)
            credit = st.number_input("Requested Credit Amount", 10000.0, 5_000_000.0, 500000.0, step=5000.0)
            annuity = st.number_input("Annuity (monthly)", 500.0, 200_000.0, 25000.0, step=500.0)
            goods_price = st.number_input("Goods Price", 0.0, 5_000_000.0, 480000.0, step=5000.0)
        with c2:
            age = st.slider("Age", 18, 75, 35)
            employed_years = st.slider("Years Employed", 0, 45, 6)
            fam_members = st.slider("Family Members", 1, 10, 2)
            children = st.slider("Number of Children", 0, 6, 0)
        with c3:
            gender = st.selectbox("Gender", ["F", "M"])
            education = st.selectbox("Education", ["Secondary / secondary special", "Higher education",
                                                     "Incomplete higher", "Lower secondary", "Academic degree"])
            income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant"])
            contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        c4, c5, c6 = st.columns(3)
        ext1 = c4.slider("External Score 1", 0.0, 1.0, 0.5)
        ext2 = c5.slider("External Score 2", 0.0, 1.0, 0.5)
        ext3 = c6.slider("External Score 3", 0.0, 1.0, 0.5)
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        row = {
            "SK_ID_CURR": 999999, "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods_price,
            "DAYS_BIRTH": -int(age * 365.25), "DAYS_EMPLOYED": -int(employed_years * 365.25),
            "DAYS_ID_PUBLISH": -2000, "CNT_FAM_MEMBERS": fam_members, "CNT_CHILDREN": children,
            "CODE_GENDER": gender, "NAME_EDUCATION_TYPE": education, "NAME_INCOME_TYPE": income_type,
            "NAME_CONTRACT_TYPE": contract_type, "NAME_FAMILY_STATUS": "Married",
            "NAME_HOUSING_TYPE": "House / apartment", "OBS_30_CNT_SOCIAL_CIRCLE": 2, "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "EXT_SOURCE_1": ext1, "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
        }
        t0 = time.time()
        result = predict_single(row, model, preprocessor)
        elapsed_ms = (time.time() - t0) * 1000
        st.session_state["last_row"] = row
        st.session_state["last_result"] = result

        st.caption(f"Scored in {elapsed_ms:.0f} ms")
        band, color = risk_band(result["default_probability"])
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(gauge(result["risk_score"], "AI Risk Score (0-1000)", 1000), use_container_width=True)
        with c2:
            st.plotly_chart(gauge(result["default_probability"]*100, "Default Probability", 100, "%"), use_container_width=True)
        with c3:
            st.markdown(f"### Risk Band\n<span class='badge' style='background:{color}33;color:{color};border:1px solid {color};'>{band}</span>",
                        unsafe_allow_html=True)
            decision = "APPROVE" if result["default_probability"] < 0.25 else ("MANUAL REVIEW" if result["default_probability"] < 0.65 else "DECLINE")
            dcolor = {"APPROVE": GREEN, "MANUAL REVIEW": AMBER, "DECLINE": RED}[decision]
            st.markdown(f"### Decision\n<span class='badge' style='background:{dcolor}33;color:{dcolor};border:1px solid {dcolor};'>{decision}</span>",
                        unsafe_allow_html=True)
    else:
        st.info("Fill in the applicant's details and click **Run Prediction**.")


# ===========================================================================
# About Project
# ===========================================================================
elif page == "About Project":
    st.title("About CreditLens")
    st.markdown("""
CreditLens is an AI credit-risk scoring platform built on all **8 tables**
of the Home Credit Default Risk dataset.

**Pipeline (`run_pipeline.py`):** load all 8 CSVs → validate → merge onto
`application_train` via `SK_ID_CURR` (bureau + bureau_balance,
previous_application, POS_CASH_balance, installments_payments,
credit_card_balance, each aggregated with mean/sum/max/min/count) → clean →
feature engineering (income, loan amount, existing debt, credit history,
payment history, employment history, external credit scores) → feature
selection → train/test split → train Logistic Regression, Decision Tree,
and Random Forest → cross-validation → hyperparameter tuning → compare on
Accuracy, Precision, Recall, F1, ROC-AUC → save the best model.

**Explainability:** global feature importance and SHAP values, plus
per-customer local SHAP explanations.

**Architecture:** `run_pipeline.py` does all training and saves every
artifact the dashboard needs (model, preprocessor, metrics, dataset
overview, data-quality report, EDA summary, feature-engineering summary)
into `models/metrics.pkl` and friends. `app.py` only loads those saved
artifacts (cached) — it never retrains and never accepts a raw file
upload. `pipeline/merge.py` handles loading + merging; `pipeline/predict.py`
handles single-customer inference.
""")
    st.subheader("Currently Loaded Model")
    st.json({
        "model": metrics.get("best_model_name"),
        "metrics": metrics.get("best_metrics"),
    })


# ===========================================================================
# Footer — shown on every page
# ===========================================================================
st.markdown("""
<div class="app-footer">
    ------------------------------------------------<br>
    <strong>CreditLens</strong> – AI Credit Risk Intelligence Platform<br>
    Developed by <strong>Jiten Hudda</strong><br>
    Powered by Python • Streamlit • Scikit-learn • Plotly<br>
    ------------------------------------------------
</div>
""", unsafe_allow_html=True)
