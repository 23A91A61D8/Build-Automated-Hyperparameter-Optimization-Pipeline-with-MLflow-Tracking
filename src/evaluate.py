import json
import os
import numpy as np
import mlflow
import mlflow.xgboost

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_best_model(
    study,
    X_train,
    X_test,
    y_train,
    y_test,
    optimization_time
):
    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params

    # Train final model on full training data
    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred)

    # MLflow final run
    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("best_model", "true")

        # Log parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # Log trained model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model"
        )

    # Prepare results.json
    results = {
        "n_trials_completed": len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        ),
        "n_trials_pruned": len(
            [t for t in study.trials if t.state.name == "PRUNED"]
        ),
        "best_cv_rmse": np.sqrt(-best_trial.value),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": optimization_time
    }

    os.makedirs("outputs", exist_ok=True)
    results_path = os.path.join("outputs", "results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
