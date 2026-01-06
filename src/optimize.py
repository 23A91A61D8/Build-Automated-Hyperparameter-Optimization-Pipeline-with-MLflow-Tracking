import os
import time
import random
import json
import numpy as np
import mlflow
import optuna
import matplotlib.pyplot as plt

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances

from src.data_loader import load_and_split_data
from src.objective import objective
from src.evaluate import evaluate_best_model


def main():
    # -------------------------------
    # Set global random seeds
    # -------------------------------
    random.seed(42)
    np.random.seed(42)

    # -------------------------------
    # Output directories
    # -------------------------------
    os.makedirs("outputs", exist_ok=True)

    # -------------------------------
    # Load data
    # -------------------------------
    X_train, X_test, y_train, y_test = load_and_split_data()

    # -------------------------------
    # MLflow configuration
    # -------------------------------
    mlflow.set_tracking_uri("file:///app/outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    # -------------------------------
    # Optuna study configuration
    # -------------------------------
    storage = "sqlite:///optuna_study.db"

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=5
    )

    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True
    )

    start_time = time.time()

    # -------------------------------
    # Optimization loop
    # -------------------------------
    def mlflow_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            try:
                score = objective(trial, X_train, y_train)

                # Log hyperparameters
                for param_name, param_value in trial.params.items():
                    mlflow.log_param(param_name, param_value)

                # Log metrics
                mlflow.log_metric("cv_mse", -score)
                mlflow.log_metric("cv_rmse", np.sqrt(-score))
                mlflow.log_metric("trial_number", trial.number)

                mlflow.set_tag("trial_state", "COMPLETE")

                return score

            except optuna.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise

            except Exception:
                mlflow.set_tag("trial_state", "FAIL")
                raise

    study.optimize(
        mlflow_objective,
        n_trials=100,
        n_jobs=2
    )

    optimization_time = time.time() - start_time

    # -------------------------------
    # Save Optuna visualizations
    # -------------------------------
    history_fig = plot_optimization_history(study)
    importance_fig = plot_param_importances(study)

    history_path = "outputs/optimization_history.png"
    importance_path = "outputs/param_importance.png"

    history_fig.write_image(history_path)
    importance_fig.write_image(importance_path)

    # Log artifacts to MLflow
    mlflow.log_artifact(history_path)
    mlflow.log_artifact(importance_path)

    # -------------------------------
    # Evaluate best model
    # -------------------------------
    evaluate_best_model(
        study,
        X_train,
        X_test,
        y_train,
        y_test,
        optimization_time
    )


if __name__ == "__main__":
    main()
