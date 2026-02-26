"""
File: main.py
Purpose: Execute the full Bangalore Rural Electrification Predictor pipeline end-to-end.
Author: Your Name
Date: February 26, 2026
"""

from pathlib import Path

from src.data_cleaning import clean_dataset
from src.data_generation import generate_dataset
from src.eda import run_eda
from src.model import train_and_evaluate
from src.preprocessing import preprocess
from src.visualization import plot_results


def main() -> None:
    """
    Run all project steps in sequence from data generation to visualization.

    Returns:
        None: Executes the pipeline and prints progress/results.
    """
    # Print startup banner for the project pipeline.
    print("=" * 60)
    print("  Bangalore Rural Electrification Predictor")
    print("  Mini Project Pipeline")
    print("=" * 60)

    # Define all required input and output paths relative to project root.
    project_root = Path(__file__).resolve().parent
    raw_data_path = project_root / "data" / "raw" / "data.csv"
    processed_data_path = project_root / "data" / "processed" / "rural_electrification_clean.csv"
    output_dir = project_root / "outputs"

    # Step 1: Generate raw synthetic village dataset.
    generate_dataset(str(raw_data_path))

    # Step 2: Clean and validate the raw dataset for modeling.
    cleaned_df = clean_dataset(str(raw_data_path), str(processed_data_path))

    # Step 3: Run exploratory data  analysis and visualize relationships.
    run_eda(cleaned_df)

    # Step 4: Preprocess cleaned data into train/test splits and scaled arrays.
    X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler = preprocess(
        cleaned_df
    )
    _ = scaler

    # Step 5: Train linear regression and evaluate prediction quality.
    model, y_pred = train_and_evaluate(
        X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test
    )
    _ = model

    # Step 6: Save output plots for final result interpretation.
    plot_results(X_test, y_test, y_pred, str(output_dir))


if __name__ == "__main__":
    main()
