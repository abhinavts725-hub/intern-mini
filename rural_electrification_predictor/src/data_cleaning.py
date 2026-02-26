"""
File: data_cleaning.py
Purpose: Clean and validate the raw Bangalore electrification dataset.
Author: Your Name
Date: February 26, 2026
"""

from pathlib import Path

import numpy as np
import pandas as pd


def clean_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Clean the raw electrification dataset and save the processed file.

    Args:
        input_path (str): Path to the raw CSV file.
        output_path (str): Path to save the cleaned CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame after validation and transformations.
    """
    # Step 1: Load CSV from disk into a DataFrame.
    df = pd.read_csv(input_path)

    # Step 2: Print missing-value counts before any cleaning operations.
    print("Missing values before cleaning:")
    print(df.isna().sum())

    # Step 3: Drop rows where village_name or electrification_pct is missing.
    df = df.dropna(subset=["village_name", "electrification_pct"]).copy()

    # Step 4: Fill remaining numeric nulls with each column median.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Step 5: Remove duplicate village names and keep the first appearance.
    df = df.drop_duplicates(subset=["village_name"], keep="first").copy()

    # Step 6: Enforce consistent data types for model-readiness.
    df["village_name"] = df["village_name"].astype(str)
    df["population"] = df["population"].round().astype(int)
    float_cols = ["area_sqkm", "pop_density", "distance_from_grid_km", "electrification_pct"]
    for col in float_cols:
        df[col] = df[col].astype(float)

    # Step 7: Flag and print population outliers outside expected bounds.
    population_outliers = df[(df["population"] < 500) | (df["population"] > 100000)]
    print("Population outliers (<500 or >100000):")
    print(population_outliers if not population_outliers.empty else "None found")

    # Step 8: Flag and print area outliers outside expected bounds.
    area_outliers = df[(df["area_sqkm"] < 1) | (df["area_sqkm"] > 100)]
    print("Area outliers (<1 or >100):")
    print(area_outliers if not area_outliers.empty else "None found")

    # Step 9: Remove rows where population density is invalid or non-positive.
    df = df[df["pop_density"] > 0].copy()

    # Step 10: Clip electrification percentages into a realistic cleaned range.
    df["electrification_pct"] = df["electrification_pct"].clip(30, 98)

    # Step 11: Clip distance from grid to reasonable physical bounds.
    df["distance_from_grid_km"] = df["distance_from_grid_km"].clip(1, 100)

    # Step 12: Save the cleaned dataset to the processed path.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    # Step 13: Print final shape and completion confirmation message.
    print(f"Final cleaned dataset shape: {df.shape}")
    print(f"Cleaned dataset saved to: {output_file}")
    return df
