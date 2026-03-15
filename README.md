# Retention Risk Workbench

Interactive churn risk pipeline that trains models, scores customers, and surfaces insights through a Streamlit dashboard and FastAPI service. This README documents the current repo layout and the commands you need to run it end to end.

---

## Quickstart (Telco demo)
- Create a virtual env and install deps:  
  `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Train the default telco model and write artifacts to `artifacts/`:  
  `python -m src.cli.train --config configs/base.yaml`
- Launch the dashboard pointing at those artifacts:  
  `streamlit run dashboard/app.py`

---

## Data inputs
- **Telco (default):** CSV at `data/raw/telco_churn.csv` (already staged in the repo). Target column defaults to `Churn`.
- **KKBox:** Raw tables expected in `data/raw/` (`train.csv`, `members.csv`, `transactions.csv`, `user_logs.csv`). Use the KKBox feature builder to create Parquet features before training:
  - Build features (full data): `python -m src.cli.kkbox_features`
  - Quick sample build: `python -m src.cli.kkbox_features --test-mode`
  - The training config expects `data/processed/kkbox_features_duckdb_cleaned_train.parquet`.

---

## Core workflows

- **Train a model**
  - Telco: `python -m src.cli.train --config configs/base.yaml`
  - KKBox: `python -m src.cli.train --config configs/kkbox.yaml`
  - Writes: `artifacts[/kkbox]/` with `model.pkl`, `model_metadata.json`, `figures/`, `splits/`, and `candidate_model_results.csv`. Metrics/logs go to MLflow (SQLite at `mlruns/mlflow.db` by default).

- **Hyperparameter tuning (optional)**
  - `python -m src.cli.tune --config configs/base.yaml`
  - Supports `tuning.algorithm` = `xgboost` or `lightgbm`; best params are saved to `artifacts/tuning/best_<algo>_params.json` and auto-applied on the next train.

- **Evaluate a saved model**
  - `python -m src.cli.evaluate --config configs/base.yaml --artifact-dir artifacts --report-dir reports`
  - Prefers saved test splits under `artifacts/splits/`; if missing, recomputes from the config (warns). Outputs ROC/PR/confusion matrix figures and `reports/executive_summary.md`.

- **Batch scoring / predictions**
  - `python -m src.cli.predict --config configs/base.yaml --input data/sample_scoring_input.csv`
  - Saves timestamped CSVs to `data/predictions/` (or `data/predictions/kkbox/` when dataset_type=kkbox) and updates `latest_predictions.csv`.

- **Interactive dashboard**
  - `streamlit run dashboard/app.py`
  - Sidebar lets you pick which artifact folder to load (root telco or subfolders like `artifacts/kkbox`). Expects matching predictions in `data/predictions[/<dataset>]`.

- **FastAPI scoring service (local)**
  - `uvicorn api.main:app --host 0.0.0.0 --port 8000`
  - Requires `artifacts/model.pkl` and `artifacts/model_metadata.json`. Protect endpoints with `API_KEY` env var (header `X-API-Key`). Exposes `/health`, `/model-info`, `/predict`, and `/batch-predict`; Prometheus metrics available at `/metrics`.

- **Local stack via Docker Compose (API + MLflow + Prometheus + Grafana + dashboard)**
  - `cd infra && docker compose up --build`
  - Mounts your local `artifacts/` and `mlruns/` into the containers. See `infra/README.md` for details and caveats.

---

## Repository map (top level)
```
README.md
configs/                  # YAML configs for datasets and training
data/                     # raw inputs, processed features, predictions
artifacts/                # trained models, figures, splits, tuning outputs
mlruns/                   # MLflow tracking (SQLite backend by default)
dashboard/                # Streamlit app (dashboard/app.py)
api/                      # FastAPI service (api/main.py)
src/
  cli/                    # CLI entrypoints (train, tune, evaluate, predict, kkbox_features)
  training/               # training loop, model selection, artifact writers
  kkbox/                  # KKBox feature engineering pipeline
  utils/                  # IO, metrics, split utilities, themes
  reporting/              # reporting helpers
notebooks/                # explorations and data prep
infra/                    # Dockerfiles, compose stack, k8s manifests
scripts/                  # quality_gate.sh
tests/                    # unit tests
```

---

## Artifacts produced by training
- `artifacts/model.pkl` — sklearn pipeline (preprocessor + model)
- `artifacts/model_metadata.json` — schema, feature columns, threshold, dataset info
- `artifacts/figures/` — ROC, PR, calibration, confusion matrix, SHAP
- `artifacts/splits/` — deterministic test IDs/rows (CSV and Parquet)
- `artifacts/candidate_model_results.csv` — metrics for all tried models
- `artifacts/tuning/` — Optuna trial table and best params (when tuning is run)

---

## Quality and tests
- Install dev tools: `pip install -r requirements-dev.txt`
- Run lightweight guardrail: `./scripts/quality_gate.sh`
  - Compiles `src/` and `tests/`, runs Ruff lint (errors only), and fails if junk files (pyc, __pycache__, stray artifacts) are present.

---

## Notes
- Python 3.10+ recommended for LightGBM/XGBoost wheels.
- KKBox feature build can be memory-heavy; use `--test-mode` for a sampled run when iterating locally.
- MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns`

