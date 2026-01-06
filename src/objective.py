import numpy as np
import random
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score


def objective(trial, X_train, y_train):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Define hyperparameter search space (EXACTLY 7)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.001, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 10
        ),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }

    # Create XGBoost regressor
    model = XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    # 5-fold cross-validation using negative MSE
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    # Return mean negative MSE
    return cv_scores.mean()
