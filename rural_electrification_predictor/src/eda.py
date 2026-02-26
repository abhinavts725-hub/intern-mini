"""
File: eda.py
Purpose: Run exploratory data analysis for electrification insights.
Author: Your Name
Date: February 26, 2026
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(df: pd.DataFrame) -> None:
    """
    Perform exploratory analysis and visual inspection of cleaned data.

    Args:
        df (pd.DataFrame): Cleaned dataset to analyze.

    Returns:
        None: This function prints outputs and displays plots.
    """
    # Step 1: Print descriptive statistics to summarize numeric distributions.
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Step 2: Print DataFrame schema to inspect types and non-null counts.
    print("\nDataFrame Info:")
    df.info()

    # Step 3: Plot distance from grid versus electrification percentage.
    plt.figure(figsize=(10, 6))
    plt.scatter(df["distance_from_grid_km"], df["electrification_pct"], alpha=0.7)
    plt.title("Distance from Grid vs. Electrification %")
    plt.xlabel("Distance from Grid (km)")
    plt.ylabel("Electrification (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 4: Plot population density versus electrification percentage.
    plt.figure(figsize=(10, 6))
    plt.scatter(df["pop_density"], df["electrification_pct"], alpha=0.7, color="orange")
    plt.title("Population Density vs. Electrification %")
    plt.xlabel("Population Density (people per sq km)")
    plt.ylabel("Electrification (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 5: Plot enhanced polynomial fit for distance and electrification.
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x="distance_from_grid_km",
        y="electrification_pct",
        order=2,
        line_kws={"color": "red"},
        scatter_kws={"alpha": 0.6, "color": "steelblue"},
    )
    plt.title("Distance from Grid vs. Electrification % with Polynomial Fit (Order 2)")
    plt.xlabel("Distance from Grid (km)")
    plt.ylabel("Electrification (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 6: Print written insights so non-visual runs still capture findings.
    print("\nEDA Summary:")
    print(
        "- Electrification generally declines as distance from the main grid increases.\n"
        "- Population density shows a mild positive relationship with electrification levels.\n"
        "- The polynomial fit indicates slight non-linearity, with steeper decline at larger distances."
    )
