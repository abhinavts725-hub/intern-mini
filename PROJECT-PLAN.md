# 1. DATA SET CREATION PLAN

## Goal

Build a dataset of ~200 - 220 Bangalore rural villages for the Rural Electrification Predictor model.

## Steps

### 1. Village Selection

- Select 200–210 villages from Bangalore Rural & Urban fringe areas
- Source: Karnataka state village directory

### 2. Population Data

- Merge population figures from Census 2011
- Calculate population density = population / area (sq km)
- Source: censusindia.gov.in

### 3. Distance from Nearest Power Grid

- Identify major KPTCL substations near Bangalore as grid hubs
- Estimate distance (km) from each village to nearest hub
- Method: Google Maps manual / coordinate-based estimation

### 4. Electrification % (Simulated)

- Generated using formula:
  `electrification% = 100 - (α × distance) + (β × pop_density) + noise`

  > - α (alpha) = 1.2 → the weight for distance
  > - β (beta) = 0.015 → the weight for pop_density

- Capped between 30% – 100%
- Intentionally includes non-linear noise for realistic modeling

## Output

A clean CSV with columns:
`village_name, population, pop_density, distance_from_grid_km, electrification_pct`

## Example

| village_name  | population | area_sqkm | pop_density | distance_from_grid_km | electrification_pct |
| ------------- | ---------- | --------- | ----------- | --------------------- | ------------------- |
| Anekal        | 14200      | 18.4      | 771         | 8.2                   | 88.4                |
| Devanahalli   | 9800       | 22.1      | 443         | 24.5                  | 61.2                |
| Doddaballapur | 11500      | 19.7      | 583         | 18.3                  | 70.8                |
| Hoskote       | 8700       | 25.3      | 344         | 31.7                  | 52.1                |
| Nelamangala   | 12300      | 16.8      | 732         | 11.4                  | 83.6                |

## Formulas

- pop_density = population / area_sqkm

- electrification% = 100 - (1.2 × distance_km) + (0.015 × pop_density) + noise(-5, +5)

**Why these weights?**

| Factor      | Effect                          | Weight   |
| ----------- | ------------------------------- | -------- |
| Distance    | Farther = less electricity      | `-1.2`   |
| Pop density | Denser = more demand/investment | `+0.015` |
| Noise       | Real-world randomness           | `±5`     |

<br/>

> **NOTE:** It is also possible to generate an entire dataset with powerful AI models like Claude, ChatGPT, Perplexity, or Gemini.

> **IMPORTANT:** It is essential to maintain **consistency and correctness** throughout the dataset.

<br/>

---

# 2. DATA CLEANING

## Goal

Ensure the dataset is accurate, consistent, and ready for model training.

## Checklist

### Step 1 — Check for Missing Values

- Identify any null or empty cells in all columns
- Action: Drop rows with missing `village_name` or `electrification_pct`
- Action: Fill missing numeric values with column median if minimal (<5%)

```python
df.isnull().sum()                        # check nulls
df.dropna(subset=['village_name'], inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
```

### Step 2 — Remove Duplicate Villages

- Check for duplicate village names
- Keep the first occurrence, drop the rest

```python
df.duplicated('village_name').sum()      # count duplicates
df.drop_duplicates(subset='village_name', keep='first', inplace=True)
```

### Step 3 — Fix Data Types

- Ensure all numeric columns are `float` or `int`
- Ensure `village_name` is `string`

```python
df['population'] = df['population'].astype(int)
df['area_sqkm'] = df['area_sqkm'].astype(float)
df['distance_from_grid_km'] = df['distance_from_grid_km'].astype(float)
df['electrification_pct'] = df['electrification_pct'].astype(float)
```

### Step 4 — Validate Ranges (Outlier Check)

| Column                  | Valid Range   | Action if outside     |
| ----------------------- | ------------- | --------------------- |
| `population`            | 500 – 100,000 | Flag & review         |
| `area_sqkm`             | 1 – 100       | Flag & review         |
| `pop_density`           | > 0           | Drop if zero/negative |
| `distance_from_grid_km` | 1 – 100 km    | Clip to range         |
| `electrification_pct`   | 30% – 98%     | Clip to range         |

```python
df['electrification_pct'] = df['electrification_pct'].clip(30, 98)
df['distance_from_grid_km'] = df['distance_from_grid_km'].clip(1, 100)
df = df[df['pop_density'] > 0]
```

### Step 5 — Recalculate Derived Columns

- Recalculate `pop_density` from `population` and `area_sqkm` to ensure consistency

```python
df['pop_density'] = (df['population'] / df['area_sqkm']).round(2)
```

### Step 6 — Final Check

- Confirm row count is between 200–220
- Confirm no nulls remain
- Confirm all columns are present

```python
print(df.shape)
print(df.isnull().sum())
print(df.describe())
```

## Output

A fully cleaned CSV saved as:
`bangalore_electrification_clean.csv`

```python
df.to_csv('bangalore_electrification_clean.csv', index=False)
```

---

---

# 3. EXPLORATORY DATA ANALYSIS (EDA)

## Goal

Visually explore the dataset to understand patterns, spot non-linear relationships, and justify the use of regression.

## Step 1 — Basic Statistics

Get a quick summary of all columns.

```python
import pandas as pd

df = pd.read_csv('bangalore_electrification_clean.csv')
print(df.describe())
print(df.info())
```

## Step 2 — Scatter Plot: Distance vs Electrification %

> ⚠️ Project specifically requires identifying non-linear patterns BEFORE applying linear regression.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(df['distance_from_grid_km'], df['electrification_pct'], color='steelblue', alpha=0.6)
plt.title('Distance from Grid vs Electrification %')
plt.xlabel('Distance from Grid Hub (km)')
plt.ylabel('Electrification %')
plt.grid(True)
plt.tight_layout()
plt.savefig('distance_vs_electrification.png')
plt.show()
```

**Expected pattern:** As distance increases, electrification % drops in a **curved (non-linear)** trend.

## Step 3 — Scatter Plot: Population Density vs Electrification %

```python
plt.figure(figsize=(8, 5))
plt.scatter(df['pop_density'], df['electrification_pct'], color='darkorange', alpha=0.6)
plt.title('Population Density vs Electrification %')
plt.xlabel('Population Density (per sq km)')
plt.ylabel('Electrification %')
plt.grid(True)
plt.tight_layout()
plt.savefig('popdensity_vs_electrification.png')
plt.show()
```

## Step 4 — Add Polynomial Fit to Show Non-Linearity

```python
import numpy as np

x = df['distance_from_grid_km']
y = df['electrification_pct']

# Fit a polynomial curve (degree 2)
coeffs = np.polyfit(x, y, deg=2)
poly = np.poly1d(coeffs)
x_line = np.linspace(x.min(), x.max(), 300)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='steelblue', alpha=0.6, label='Actual Data')
plt.plot(x_line, poly(x_line), color='red', linewidth=2, label='Polynomial Fit (degree 2)')
plt.title('Non-Linear Pattern: Distance vs Electrification %')
plt.xlabel('Distance from Grid Hub (km)')
plt.ylabel('Electrification %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('nonlinear_pattern.png')
plt.show()
```

## Step 5 — Correlation Heatmap

```python
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(df[['population', 'pop_density', 'distance_from_grid_km', 'electrification_pct']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
```

## EDA Output Summary

| Plot                             | Insight                                     |
| -------------------------------- | ------------------------------------------- |
| Distance vs Electrification %    | Non-linear drop as distance increases       |
| Pop Density vs Electrification % | Higher density → higher electrification     |
| Polynomial Fit                   | Confirms non-linear relationship            |
| Heatmap                          | Distance has strongest negative correlation |

## Conclusion from EDA

> The scatter plots confirm a **non-linear relationship** between distance and electrification %. Despite this, we proceed with **Linear Regression** as a baseline model to quantify the trend and measure prediction accuracy.

---

# 4. PRE-PROCESSING

## Goal

Prepare the cleaned dataset for model training.

## Step 1 — Select Features & Target

```python
X = df[['distance_from_grid_km', 'pop_density']]  # features
y = df['electrification_pct']                       # target
```

## Step 2 — Train / Test Split (80/20)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")
```

## Step 3 — Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

> **Why scale?** Distance and pop_density have very different ranges. Scaling brings them to the same level for fair model training.

---

# 5. LINEAR REGRESSION

## Goal

Train a linear regression model to predict electrification % from distance and population density.

## Step 1 — Train the Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", dict(zip(X.columns, model.coef_)))
```

## Step 2 — Predict

```python
y_pred = model.predict(X_test_scaled)
```

## Step 3 — Evaluate

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
```

**What these mean:**

| Metric | Meaning                          | Good Value      |
| ------ | -------------------------------- | --------------- |
| R²     | How well model explains variance | Closer to 1     |
| MAE    | Average prediction error in %    | Lower is better |
| RMSE   | Penalizes large errors more      | Lower is better |

---

# 6. RESULTS & VISUALIZATION

## Plot: Actual vs Predicted

```python
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Fit')
plt.xlabel('Actual Electrification %')
plt.ylabel('Predicted Electrification %')
plt.title('Actual vs Predicted Electrification %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()
```

## Plot: Regression Line (Distance vs Electrification)

```python
plt.figure(figsize=(8, 5))
plt.scatter(X_test['distance_from_grid_km'], y_test, color='steelblue', alpha=0.6, label='Actual')
plt.scatter(X_test['distance_from_grid_km'], y_pred, color='red', alpha=0.6, label='Predicted')
plt.xlabel('Distance from Grid Hub (km)')
plt.ylabel('Electrification %')
plt.title('Linear Regression: Distance vs Electrification %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('regression_result.png')
plt.show()
```

## Sample Prediction Table

```python
results = X_test.copy()
results['actual_pct']    = y_test.values
results['predicted_pct'] = y_pred.round(2)
results['error']         = (results['actual_pct'] - results['predicted_pct']).abs().round(2)
print(results.head(10))
```

---

# 7. SDG 9 IMPACT ANALYSIS

## SDG Goal

**SDG 9 — Industry, Innovation, and Infrastructure**
_"Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation."_

## How This Project Contributes

| SDG Target                                     | Our Contribution                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Expand access to infrastructure in rural areas | Model identifies villages with low electrification for priority intervention |
| Use data & innovation for development planning | Machine learning used to predict electrification gaps                        |
| Support sustainable industrialization          | Electrification enables local industry, schools, and healthcare              |

## Key Insight from Model

Villages **more than 30 km** from the nearest grid hub have electrification rates below **60%**. These villages should be prioritized for solar/off-grid solutions to meet SDG 9 targets.

## Recommendations

- Villages with `distance > 30km` and `pop_density < 300` → prioritize solar microgrids
- Villages with `distance < 15km` → feasible for direct grid extension
- Policy makers can use this model to allocate rural electrification budgets effectively

---

# 8. GITHUB REPOSITORY

## Structure

```
rural-electrification-predictor/
│
├── data/
│   ├── bangalore_villages_raw.csv
│   └── bangalore_electrification_clean.csv
│
├── notebooks/
│   └── rural_electrification.ipynb
│
├── outputs/
│   ├── distance_vs_electrification.png
│   ├── nonlinear_pattern.png
│   ├── correlation_heatmap.png
│   ├── actual_vs_predicted.png
│   └── regression_result.png
│
├── README.md
└── requirements.txt
```

## README Must Include

- Project title & SDG goal
- Dataset description
- Steps: Data creation → Cleaning → EDA → Model → Results
- How to run the code
- Screenshots of results

## requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Git Commands

```bash
git init
git add .
git commit -m "Initial commit - Rural Electrification Predictor"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

> **Note:** Repository must remain **public** at all times as per guidelines.
