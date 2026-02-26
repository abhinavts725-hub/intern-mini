# Bangalore Rural Electrification Predictor

## Project Overview

This mini project simulates village-level electrification conditions for Bangalore Rural and nearby urban fringe areas. It builds a synthetic dataset, cleans it, explores patterns, trains a linear regression model, and saves visual outputs. The goal is to demonstrate an end-to-end data science workflow that connects distance-to-grid and population density with electrification outcomes.

## Dataset

The dataset is synthetically generated for exactly 200 villages. Each row represents one village and includes:

- `village_name`: Name of the village.
- `population`: Simulated village population (2,000 to 16,000).
- `area_sqkm`: Simulated village area in square kilometers (10 to 30).
- `pop_density`: Population density (`population / area_sqkm`).
- `distance_from_grid_km`: Distance to nearest electrical grid access point (4.5 to 48 km, shuffled).
- `electrification_pct`: Simulated electrification percentage computed by formula and clipped to realistic limits.

Formula used during generation:

`electrification_pct = 100 - (1.2 * distance_km) + (0.015 * pop_density) + noise(-5, +5)`

The generated value is clipped between `30` and `100` during generation and then clipped to `30` to `98` during cleaning.

- EXPLANATION :
  Due to lack of publicly available village-level electrification datasets with required features, a synthetic dataset was generated using realistic ranges inspired by rural infrastructure data. This dataset enables demonstration of machine learning techniques for electrification prediction.

## Folder Structure

```text
bangalore_electrification_predictor/
|
|-- data/
|   |-- raw/
|   |   `-- data.csv
|   `-- processed/
|       `-- bangalore_electrification_clean.csv
|
|-- src/
|   |-- __init__.py
|   |-- data_generation.py
|   |-- data_cleaning.py
|   |-- eda.py
|   |-- preprocessing.py
|   |-- model.py
|   `-- visualization.py
|
|-- outputs/
|   |-- actual_vs_predicted.png
|   `-- regression_result.png
|
|-- main.py
|-- requirements.txt
`-- README.md
```

## How to Run

1. Open a terminal in the project root.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the complete pipeline:
   `python main.py`

## Pipeline Steps

1. **Data Generation**: Creates a 200-row synthetic village dataset.
2. **Data Cleaning**: Handles missing values, duplicates, data types, and clipping rules.
3. **EDA**: Prints dataset summaries and displays scatter plots/regression fit.
4. **Preprocessing**: Selects features, splits train/test sets, and scales features.
5. **Model Training**: Fits a linear regression model and reports R^2, MAE, RMSE.
6. **Visualization**: Saves actual-vs-predicted and distance-vs-regression plots.

## Model

The model is **Linear Regression** from scikit-learn.

- Inputs (features):
  - `distance_from_grid_km`
  - `pop_density`
- Output (target):
  - `electrification_pct`
- Metrics:
  - **R^2**: Fraction of target variance explained by the model.
  - **MAE**: Average absolute prediction error (percentage points).
  - **RMSE**: Error magnitude that penalizes larger mistakes.

## Formula Used

`electrification% = 100 - (1.2 * distance_km) + (0.015 * pop_density) + noise`

Interpretation:

- Distance has a strong negative weight (`-1.2`): farther villages tend to have lower electrification.
- Population density has a mild positive weight (`+0.015`): denser villages tend to get slightly better electrification.
- Random noise models real-world uncertainty and unobserved factors.

## Results

After running `main.py`, the script prints model quality values:

- **R^2** indicates how well the model captures variation in electrification.
- **MAE** indicates average prediction error in percentage points.
- **RMSE** indicates error severity while giving more weight to larger misses.

These outputs help interpret whether the synthetic relationship is predictable and stable.

## Disclaimer

This dataset is synthetically generated for academic and mini project demonstration purposes only. It is not an official government dataset and should not be used for policy decisions.
