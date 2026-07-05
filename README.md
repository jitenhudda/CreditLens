# CreditLens

## AI-Powered Credit Risk Intelligence Platform

CreditLens is an end-to-end Machine Learning project designed to predict the probability of customer loan default using the Home Credit Default Risk dataset.

The project automates the complete machine learning pipeline—from loading multiple datasets and performing feature engineering to training, evaluating, and deploying the best-performing model through an interactive Streamlit dashboard.

Unlike traditional dashboards, CreditLens trains the model only once using a dedicated pipeline and loads the saved artifacts for fast and efficient analysis.

---

## Features

- End-to-End Machine Learning Pipeline
- Automatic Loading of 8 Home Credit CSV Files
- Data Validation & Cleaning
- Multi-Table Data Merging
- Feature Engineering & Feature Selection
- Machine Learning Model Comparison
- Automatic Best Model Selection
- SHAP Explainability
- Interactive Streamlit Dashboard
- Single Customer Credit Risk Prediction
- Modular and Professional Project Structure

---

# Workflow

```text
                     Home Credit Dataset
                            │
                            ▼
                 Load All 8 CSV Files
                            │
                            ▼
                   Data Validation
                            │
                            ▼
                     Data Cleaning
                            │
                            ▼
             Merge Multiple Data Tables
                            │
                            ▼
                 Feature Engineering
                            │
                            ▼
                  Feature Selection
                            │
                            ▼
                Data Preprocessing
                            │
                            ▼
          Train Machine Learning Models
                            │
        ┌──────────┬────────────┬───────────┐
        ▼          ▼            ▼
 Logistic Regression  Decision Tree  Random Forest
        └──────────┴────────────┴───────────┘
                            │
                            ▼
                  Model Evaluation
                            │
                            ▼
               Best Model Selection
                            │
                            ▼
                 SHAP Explainability
                            │
                            ▼
             Save Model & Artifacts
                            │
                            ▼
              Streamlit Dashboard
                            │
                            ▼
              Credit Risk Prediction
```

---

# Dataset

This project uses the **Home Credit Default Risk** dataset.

The pipeline automatically loads and processes all eight datasets:

- application_train.csv
- application_test.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- installments_payments.csv
- credit_card_balance.csv

All auxiliary datasets are aggregated and merged with the main application dataset using **SK_ID_CURR**.

---

# Machine Learning Pipeline

The complete pipeline performs the following operations automatically:

- Load all datasets
- Validate dataset integrity
- Handle missing values
- Remove duplicate records
- Merge multiple tables
- Create engineered features
- Encode categorical variables
- Scale numerical features
- Select important features
- Train multiple machine learning models
- Evaluate model performance
- Select the best-performing model
- Generate SHAP explanations
- Save trained artifacts for deployment

---

# Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

The best-performing model is automatically selected based on evaluation metrics.

---

# Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation

---

# Dashboard Modules

The interactive Streamlit dashboard includes:

- Home
- Dataset Overview
- Data Quality Report
- Exploratory Data Analysis (EDA)
- Feature Engineering Summary
- Model Performance
- SHAP Explainability
- Single Customer Prediction
- About Project

The dashboard loads only the saved model artifacts and **never retrains the model**, ensuring faster performance.

---

# Project Structure

```text
CreditLens
│
├── app.py
├── run_pipeline.py
├── config.py
├── utils.py
├── requirements.txt
├── README.md
│
├── pipeline/
│   ├── __init__.py
│   ├── merge.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
└── outputs/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/your-username/CreditLens.git
```

Navigate to the project directory

```bash
cd CreditLens
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Execute the Machine Learning Pipeline

```bash
python run_pipeline.py
```

This step will automatically:

- Load all datasets
- Validate and clean the data
- Merge multiple tables
- Perform feature engineering
- Train multiple ML models
- Evaluate model performance
- Select the best model
- Generate SHAP explainability
- Save all trained artifacts

---

## Step 2 — Launch the Dashboard

```bash
streamlit run app.py
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- SHAP
- Joblib

---

# Project Highlights

- End-to-End Machine Learning Solution
- Automated Data Processing Pipeline
- Multi-Table Feature Engineering
- Explainable AI with SHAP
- Interactive Business Dashboard
- Real-Time Credit Risk Prediction
- Professional Modular Architecture
- Internship Ready Project

---

# Future Improvements

- XGBoost & LightGBM Models
- Optuna Hyperparameter Optimization
- Batch Prediction Support
- REST API Integration
- Docker Deployment
- Cloud Deployment
- Model Monitoring Dashboard

---

# Author

**Jiten Hudda**

Machine Learning | Data Analytics | Python

---

# License

This project is developed for educational purposes and internship submission.