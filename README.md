# Build-Automated-Hyperparameter-Optimization-Pipeline-with-MLflow-Tracking

# Automated Hyperparameter Optimization Pipeline with MLflow Tracking

## Project Overview
This project implements a production-grade automated hyperparameter optimization pipeline using Optuna and MLflow.  
The pipeline systematically tunes an XGBoost regression model on the California Housing dataset using cross-validation, pruning strategies, and experiment tracking.

The goal of this project is to demonstrate professional MLOps practices including:
- Automated hyperparameter search
- Reproducible experiment tracking
- Parallel optimization
- Model evaluation and comparison
- Containerized execution using Docker

---

## Dataset
- **California Housing Dataset** (scikit-learn)
- 80/20 train-test split
- `random_state = 42` for full reproducibility

---

## Tools & Technologies
- Python 3.9+
- Optuna (hyperparameter optimization)
- MLflow (experiment tracking & model registry)
- XGBoost (regression model)
- Scikit-learn (data handling & metrics)
- Docker (containerization)
- Matplotlib / Seaborn (visualization)

---

## Optimization Details
- Model: XGBoost Regressor
- Hyperparameters tuned (7):
  - n_estimators
  - max_depth
  - learning_rate (log scale)
  - subsample
  - colsample_bytree
  - min_child_weight
  - gamma
- Cross-validation: 5-fold
- Optimization metric: Negative Mean Squared Error
- Number of trials: 100
- Pruning strategy: MedianPruner
- Parallel execution enabled
- Optuna storage backend: SQLite

---

## How to Run (Docker)

### Build the Docker image
```bash
docker build -t optuna-mlflow-pipeline .

### Run the Container
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline

