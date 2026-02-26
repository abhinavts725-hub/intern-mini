"""
File: visualization.py
Purpose: Create and save result plots for model evaluation.
Author: Your Name
Date: February 26, 2026
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred,
    output_dir: str,
) -> None:
    """
    Generate and save model result plots.

    Args:
        X_test (pd.DataFrame): Unscaled test features.
        y_test (pd.Series): Actual electrification percentages.
        y_pred: Predicted electrification percentages from the model.
        output_dir (str): Directory where output images will be saved.

    Returns:
        None: Saves plot files and prints confirmations.
    """
    # Ensure the output directory exists before writing image files.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Compare actual and predicted electrification with perfect-fit reference.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color="green", alpha=0.7, label="Predicted vs Actual")
    min_val = min(float(y_test.min()), float(np.min(y_pred)))
    max_val = max(float(y_test.max()), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")
    plt.title("Actual vs Predicted Electrification")
    plt.xlabel("Actual Electrification (%)")
    plt.ylabel("Predicted Electrification (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    actual_vs_predicted_path = output_path / "actual_vs_predicted.png"
    plt.savefig(actual_vs_predicted_path, dpi=300)
    plt.close()
    print(f"Plot saved: {actual_vs_predicted_path}")

    # Plot 2: Show regression behavior against distance from the grid.
    distance = X_test["distance_from_grid_km"].values
    order = np.argsort(distance)
    sorted_distance = distance[order]
    sorted_actual = y_test.values[order]
    sorted_predicted = np.array(y_pred)[order]
    plt.figure(figsize=(10, 6))
    plt.scatter(distance, y_test, color="steelblue", alpha=0.7, label="Actual")
    plt.plot(sorted_distance, sorted_predicted, color="red", linewidth=2, label="Predicted")
    plt.title("Regression Line: Distance vs Electrification")
    plt.xlabel("Distance from Grid (km)")
    plt.ylabel("Electrification (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    regression_result_path = output_path / "regression_result.png"
    plt.savefig(regression_result_path, dpi=300)
    plt.close()
    print(f"Plot saved: {regression_result_path}")
