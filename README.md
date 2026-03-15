# Retention Risk Workbench

A production‑style machine learning project demonstrating how a churn prediction model can be deployed to an **interactive analytics dashboard** that provides actionable insights for business stakeholders.

This project focuses on the full lifecycle of a predictive model:

* feature engineering
* model training
* probability scoring
* explainability
* business metrics
* deployment into an interactive dashboard
* monitoring and drift signals

The goal is to demonstrate how machine learning models move from experimentation into **production‑ready decision support tools**.

If you want to showcase enterprise deployment readiness, see **Deployment & Ops** below for Docker, Kubernetes, managed endpoint options, monitoring/alerting, and auth hardening. These assets are provided for demos/portfolio only—they are not hardened for internet-facing production use.

---

# Project Overview

Customer or agent churn models are widely used in industries such as insurance, telecom, SaaS, and subscription businesses to prioritize retention interventions.

This project builds a churn prediction pipeline and surfaces results in a dashboard that answers questions like:

* Which customers are most likely to churn?
* Which high‑value customers are at risk?
* What factors drive churn risk?
* What is the expected revenue at risk?

The final output is an interactive dashboard that enables exploration of model predictions and explanations.

---

# Architecture

```
Raw dataset
     ↓
Data cleaning
     ↓
Feature engineering
     ↓
Train / test split
     ↓
Model training
     ↓
Probability scoring
     ↓
Explainability (feature importance + SHAP)
     ↓
Business metrics
     ↓
Interactive dashboard
     ↓
Monitoring + reporting
```

---

# Reproducible Evaluation & Splits

- Training now writes deterministic split artifacts to `artifacts/splits/`, including `test_ids.parquet` / `.csv` (stable row IDs) and `test_set.parquet` / `.csv` (full labeled slice).
- Evaluation automatically prefers those saved splits; if they are missing, it warns and falls back to recomputing from the split config.
- Split alignment uses stable IDs (e.g., `customerID` or `msno`) when present, and it validates feature columns and split parameters against the training metadata before scoring.
- All entrypoints load data via `src/utils/io.read_table`, so CSV and Parquet inputs are both supported.

---

# Repository Structure

```
agent_churn_dashboard/

README.md
requirements.txt
app.py

src/
    data.py
    features.py
    train.py
    scoring.py
    explain.py
    monitoring.py

data/
    raw/
    processed/

models/

reports/
    model_metrics.csv
    global_feature_importance.csv
    monitoring_snapshot.csv
```

---

# Dataset

The project can be run with several open‑source churn datasets.

Recommended options:

### IBM Telco Customer Churn

A widely used telecom churn dataset containing ~7,000 customer records and service features such as tenure, charges, and contract type.

Common features include:

* tenure
* monthly charges
* total charges
* contract type
* internet service
* payment method

Target variable:

```
churn
```

---

### Cell2Cell Telecom Churn

A larger telecom churn dataset containing ~70k customers and ~50+ features including usage patterns, billing, and demographics.

This dataset provides a more realistic scenario for building production‑style dashboards.

---

# Model

The starter implementation uses **logistic regression** due to its interpretability and stability for churn prediction.

Pipeline steps:

* missing value imputation
* categorical encoding
* feature scaling
* classification model

Model outputs:

```
churn_probability
```

Predictions are grouped into risk tiers:

| Risk Band | Probability |
| --------- | ----------- |
| Low       | < 0.20      |
| Moderate  | 0.20–0.50   |
| High      | 0.50–0.75   |
| Critical  | > 0.75      |

---

# Business Metrics

To make the model useful for decision‑making, additional metrics are calculated.

Examples:

**Revenue at risk**

```
revenue_at_risk = annual_value × churn_probability
```

**Expected intervention value**

```
expected_value = churn_probability × replacement_cost
```

These metrics help prioritize retention outreach.

---

# Dashboard

The project includes a **Streamlit dashboard** for interactive exploration.

Dashboard features include:

### Executive Summary

* number of customers
* average churn probability
* number of high‑risk accounts
* revenue at risk

### Risk Distribution

Histogram of churn probabilities.

### Customer Table

Sortable table of customers with the highest churn risk.

### Customer Inspection

Select a specific customer to view:

* prediction probability
* risk tier
* revenue at risk

### Model Drivers

Global feature importance visualization.

---

# Running the Project

## Install dependencies

```
pip install pandas numpy scikit-learn shap streamlit plotly joblib
```

---

## Train the model

### Telco (default)
```
python -m src.train --config configs/base.yaml
```

### KKBox (random split by default)
```
python -m src.train --config configs/kkbox.yaml
```

What this does:

* trains candidate models (logistic + optional XGBoost/LightGBM) and picks a champion
* logs metrics/artifacts to MLflow (SQLite at `mlruns/mlflow.db` by default)
* writes model artifacts to `artifacts/` (telco) or `artifacts/kkbox/`

---

## Launch the dashboard

```
streamlit run dashboard/app.py
```

Use the sidebar to pick which artifact set to view (e.g., `artifacts/` for telco, `artifacts/kkbox/` for KKBox). The dashboard reads matching prediction files from `data/predictions/` (or `data/predictions/kkbox/`).

---

# Example Output

Example churn probability distribution:

```
Mean churn probability: 0.27
High risk accounts: 14%
Critical risk accounts: 4%
```

---

# Monitoring

Production‑style monitoring metrics are included:

* number of entities scored
* average churn score
* percent high‑risk accounts
* percent critical accounts

These metrics help detect:

* model drift
* population changes
* scoring anomalies

### MLflow UI

```
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns
```

Or run the compose stack (includes MLflow at http://localhost:5050) from `infra/`:

```
cd infra
docker compose up --build
```

---

# Future Improvements

Possible enhancements include:

* gradient boosted models
* uplift modeling
* intervention simulation
* survival analysis
* data drift detection
* automated retraining pipeline

---

# Deployment & Ops

This repo now includes sample production scaffolding:

* Container: `infra/Dockerfile` builds the FastAPI scoring service; mount `artifacts/` with the trained model.
* Local stack: `infra/docker-compose.yml` runs the API, MLflow tracking server, Prometheus, and Grafana with alert rules in `infra/monitoring`.
* Kubernetes: `infra/k8s/deployment.yaml` includes Deployment + Service + HPA + TLS Ingress; plug in your image registry and secrets.
* Managed endpoints: guidelines in `infra/README.md` for AWS SageMaker, GCP Vertex AI, or Azure ML online endpoints.
* Security: API-key guard on `/predict` and `/batch-predict` (header `X-API-Key`); front with an API gateway for OIDC/JWT and rate limiting.
* Observability: `/metrics` exposes Prometheus metrics; Grafana dashboard/alerts can page on error-rate/latency; logs stream to your centralized sink.

See `infra/README.md` for commands and a CI/CD outline.

---

# Why This Project Exists

Many churn modeling examples stop at model training.

This repository demonstrates how to:

* operationalize model predictions
* expose insights to stakeholders
* build interactive analytics tools
* monitor model outputs

These capabilities are essential for **production machine learning systems**.

---

# Author

Liam Geron

Senior Data Scientist working on applied machine learning, NLP systems, and production AI architectures.
