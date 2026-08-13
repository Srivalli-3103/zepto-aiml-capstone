# Analytics and Machine Learning Submodule

This directory holds the end-to-end data science workflow for profiling, cleaning, analyzing, and modeling the Titanic dataset.

## Installation & Environment Activation

```bash
cd analytics
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
```

## Running the Pipelines

The tasks are split into two ordered execution scripts/notebooks inside VS Code:
1. **`01_eda.ipynb`** (Profiles data, executes threshold-based cleaning, and plots the visual data story).
2. **`02_modeling.ipynb`** (Loads cleaned data, constructs ML pipelines, tunes hyperparameters, and executes the regression side-task).

---

## Unified Business Intelligence Model Comparison Matrix

The performance arrays below separate classification and regression models into distinct groups to ensure metrics are evaluated on their correct native scales.

### Classification Models Performance Metrics

| Target Model Name | Accuracy | Precision | Recall | F1-Score | Area Under ROC (AUC) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8034 | 0.7714 | 0.7397 | 0.7552 | 0.8524 |
| **Decision Tree** (Depth=4) | 0.8146 | 0.8254 | 0.7123 | 0.7647 | 0.8285 |
| **Random Forest** (Depth=5) | 0.8258 | 0.8500 | 0.6986 | 0.7669 | 0.8651 |

### Linear Regression Model Performance Metrics (Target Feature: `fare`)

| Dependent Variable Target | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R-Squared ($R^2$) | Adjusted R-Squared ($R_{adj}^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Passenger Fare Cost** | 18.4215 | 34.6180 | 0.3845 | 0.3621 |

---

## Final Production Deployment Recommendation

Based on the comprehensive performance matrices above, we recommend deploying the **Random Forest** pipeline into production. It achieves the top-tier **Accuracy of 82.58%** and demonstrates the strongest discriminative capability across unseen data with an **AUC of 0.8651**. Furthermore, it handles false positive risks exceptionally well by delivering a **Precision of 85.00%** (the highest among all evaluated architectures). This ensures highly stable predictions when facing raw, un-preprocessed new entries via the exported pipeline asset file (`titanic_production_pipeline.joblib`).

---

## Detailed Task Summaries and Analytical Design Decisions

### 1. Missing Value Threshold Strategies
* **Analysis Percentages:** `deck` (77.22%), `age` (19.87%), `embarked` (0.22%).
* **Applied Rules:** `deck` was removed entirely because it exceeded the 30% reliability threshold. Rows with missing `embarked` values were dropped because the missingness was below the 5% threshold. `age` was preserved for median imputation within the training pipeline folds.

### 2. Univariate Distribution and Outliers
* **Outlier Volume Counts:** Age contains **15** outliers beyond the whiskers; Fare contains **114** extreme outliers.
* **Skewness Profile:** Fare metrics demonstrate high right-skewness ($Mean: 32.20 > Median: 14.45 > Mode: 8.05$). This indicates a heavy pricing tail driven by a small number of luxury first-class tickets.

### 3. Strongest Structural Correlations
* **Pair 1: Fare & Pclass (-0.548):** Strong negative connection, showing that ticket prices increased significantly for upper-class cabins.
* **Pair 2: SibSp & Parch (0.414):** Confirms structural grouping behavior, showing that passengers generally traveled in family cohorts.

### 4. Class Imbalance Strategies
* **Baseline F1:** 0.762 | **Class Weighted F1:** 0.758 | **SMOTE Pipeline F1:** 0.771.
* **Conclusion:** Applying SMOTE exclusively within the training fold produced the highest F1-Score. It safely expands minority class representations without leaking data into the validation splits.

### 5. Regression Residual Analysis
* The structural residual plot shows a clear fan-shaped expansion of error terms as predicted fare costs increase. This non-random error behavior confirms the presence of **heteroscedasticity**, meaning that pricing variances scale up significantly for luxury tier accommodations.
