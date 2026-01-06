import os
import numpy as np
import random
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size=0.2, random_state=42):
    # Set seeds for reproducibility
    np.random.seed(random_state)
    random.seed(random_state)

    data_dir = "outputs"
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, "california_housing.csv")

    # Try loading local cached dataset first
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        X = df.drop("MedHouseVal", axis=1)
        y = df["MedHouseVal"]
    else:
        try:
            data = fetch_california_housing(as_frame=True)
            df = pd.concat(
                [data.data, data.target.rename("MedHouseVal")],
                axis=1
            )
            df.to_csv(data_path, index=False)
            X = data.data
            y = data.target
        except Exception as e:
            raise RuntimeError(
                "California Housing dataset download failed.\n"
                "This is due to network restrictions (HTTP 403).\n\n"
                "SOLUTION:\n"
                "1. Connect to a different network (mobile hotspot).\n"
                "2. Run load_and_split_data() ONCE.\n"
                "3. A local CSV will be created in outputs/.\n"
                "4. After that, all runs (including Docker) will work offline.\n\n"
                f"Original error: {e}"
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
