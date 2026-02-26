"""
File: preprocessing.py
Purpose: Split and scale data for model training and evaluation.
Author: Your Name
Date: February 26, 2026
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(
    df: pd.DataFrame,
) -> Tuple:
    """
    Prepare model features and target with train-test split and scaling.

    Args:
        df (pd.DataFrame): Cleaned dataset for machine learning preparation.

    Returns:
        tuple: X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler
    """
    # Step 1: Select model features tied to grid distance and village density.
    X = df[["distance_from_grid_km", "pop_density"]]

    # Step 2: Select target variable as electrification percentage.
    y = df["electrification_pct"]

    # Step 3: Split data into training and testing sets with fixed reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Fit scaler on train data only, then transform train and test features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Print sample counts for transparent training/evaluation setup.
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    return X_train_scaled, X_test_scaled, X_train, X_test, y_train, y_test, scaler
