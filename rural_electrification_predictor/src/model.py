"""
File: model.py
Purpose: Train and evaluate a linear regression model for electrification prediction.
Author: Your Name
Date: February 26, 2026
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate(
    X_train_scaled,
    X_test_scaled,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
):
    """
    Train a linear regression model and evaluate predictive performance.

    Args:
        X_train_scaled: Scaled training features.
        X_test_scaled: Scaled testing features.
        X_train (pd.DataFrame): Unscaled training features.
        X_test (pd.DataFrame): Unscaled testing features.
        y_train (pd.Series): Training target values.
        y_test (pd.Series): Testing target values.

    Returns:
        tuple: Trained model and predicted test values (model, y_pred).
    """
    # Step 1: Train a linear regression model on scaled feature inputs.
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Step 2: Print intercept and coefficients to inspect learned linear weights.
    print("\nModel Parameters:")
    print(f"Intercept: {model.intercept_:.4f}")
    print("Coefficients:")
    print(f"  distance_from_grid_km: {model.coef_[0]:.4f}")
    print(f"  pop_density: {model.coef_[1]:.4f}")

    # Step 3: Predict electrification percentages for the test feature set.
    y_pred = model.predict(X_test_scaled)

    # Step 4: Compute common regression metrics and print numeric values.
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\nEvaluation Metrics:")
    print(f"R^2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Step 5: Print a small metric interpretation table for quick readability.
    results_table = pd.DataFrame(
        {
            "metric": ["R^2", "MAE", "RMSE"],
            "value": [round(r2, 4), round(mae, 4), round(rmse, 4)],
            "meaning": [
                "Explained variance ratio (higher is better).",
                "Average absolute error in electrification percentage points.",
                "Root mean squared error, penalizing larger mistakes.",
            ],
        }
    )
    print("\nMetric Explanation Table:")
    print(results_table.to_string(index=False))

    # Step 6: Build and print first 10 predictions with per-row error values.
    comparison_df = pd.DataFrame(
        {
            "actual_pct": y_test.values,
            "predicted_pct": y_pred,
        },
        index=X_test.index,
    )
    comparison_df["error"] = comparison_df["actual_pct"] - comparison_df["predicted_pct"]
    prediction_table = (
        comparison_df.head(10).reset_index().rename(columns={"index": "village_index"})
    )
    print("\nSample Prediction Table (First 10 Rows):")
    print(prediction_table.to_string(index=False))
    return model, y_pred
