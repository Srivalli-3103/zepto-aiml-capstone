import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_absolute_error, 
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Ensure directories for output artifacts exist safely
os.makedirs("analytics", exist_ok=True)
os.makedirs("analytics/plots", exist_ok=True)

print("=====================================================================")
print("STARTING WORKFLOW: MODULE 2 (ANALYTICS & PREDICTIVE MODELING)")
print("=====================================================================\n")

# =====================================================================
# PART A: PROFILING, CLEANING, AND THE DATA STORY
# =====================================================================

print("--- [Task 1] Loading and Profiling Data ---")
# Network loading from cache or live repository (Executes exactly once)
df_raw = sns.load_dataset('titanic')

# Profile Data Structure
print("\n[DataFrame Info Profile]")
df_raw.info()
print(f"\nDataFrame Sizing Shape: {df_raw.shape}")
print("\n[DataFrame Descriptive Statistics]")
print(df_raw.describe(include='all'))

# Immediately commit raw data clone as offline fallback constraint
df_raw.to_csv("analytics/titanic.csv", index=False)
print("\nOffline baseline snapshot saved locally to: 'analytics/titanic.csv'")

# Compute and report missing margins
missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
missing_columns = missing_pct[missing_pct > 0]
print("\n[Missing Values Percentage Breakdown]")
for col, pct in missing_columns.items():
    print(f" - Column '{col}': {pct:.2f}% missing values")

print("\n--- [Task 2] Cleaning and Missing Value Resolution ---")
df_cleaned = df_raw.copy()
# Strategy justified by threshold rules:
# - embarked & embark_town (< 5% missing) -> drop those rows
# - age (19.87% missing, between 5%-30%) -> impute with median
# - deck (77.22% missing, > 30%) -> encode missing as its own category to preserve row volumes
df_cleaned = df_cleaned.dropna(subset=['embarked', 'embark_town'])
df_cleaned['deck'] = df_cleaned['deck'].cat.add_categories('Missing').fillna('Missing')
# Median assignment used purely for the EDA verification stage
df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].median())

print("\n--- [Task 3] Univariate Outlier Analytics (IQR Rules) ---")
for col in ['age', 'fare']:
    q1 = df_cleaned[col].quantile(0.25)
    q3 = df_cleaned[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
    print(f" Column '{col}': Found {len(outliers)} mathematical outliers using IQR bounds threshold.")

f_mean = df_cleaned['fare'].mean()
f_median = df_cleaned['fare'].median()
f_mode = df_cleaned['fare'].mode()[0]
print(f" Fare Distribution Indicators -> Mean: {f_mean:.2f} | Median: {f_median:.2f} | Mode: {f_mode:.2f}")
print(" Interpretation: Distribution is strongly right-skewed as Mean > Median > Mode ordering holds true.")

# Generate Distribution Histograms and Box Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.histplot(df_cleaned['age'], kde=True, ax=axes[0, 0], color='skyblue').set_title('Age Distribution Histogram')
sns.boxplot(x=df_cleaned['age'], ax=axes[0, 1], color='skyblue').set_title('Age Box Plot')
sns.histplot(df_cleaned['fare'], kde=True, ax=axes[1, 0], color='salmon').set_title('Fare Distribution Histogram')
sns.boxplot(x=df_cleaned['fare'], ax=axes[1, 1], color='salmon').set_title('Fare Box Plot')
plt.tight_layout()
plt.savefig("analytics/plots/01_univariate_distributions.png")
plt.close()

print("\n--- [Task 4] Bivariate Cross-Tabulation & Inter-Correlation ---")
print(f"Survival Rate for Females: {df_cleaned.loc[df_cleaned['sex'] == 'female', 'survived'].mean() * 100:.2f}%")
for p in sorted(df_cleaned['pclass'].unique()):
    print(f"Survival Rate for Class {p}: {df_cleaned.loc[df_cleaned['pclass'] == p, 'survived'].mean() * 100:.2f}%")
for sex in ['female', 'male']:
    for p in sorted(df_cleaned['pclass'].unique()):
        mask = (df_cleaned['sex'] == sex) & (df_cleaned['pclass'] == p)
        print(f"Survival Rate for [{sex.upper()} | Class {p}]: {df_cleaned.loc[mask, 'survived'].mean() * 100:.2f}%")

# Construct strictly restricted 6x6 correlation matrix (excluding adult_male and alone)
core_numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df_cleaned[core_numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=.5)
plt.title("Restricted Numeric 6x6 Correlation Matrix")
plt.tight_layout()
plt.savefig("analytics/plots/02_correlation_matrix.png")
plt.close()

print("\n--- [Task 5] Generating Multivariate Data Story Charts ---")
# Chart 1: Survival rates by Sex and Pclass together
plt.figure(figsize=(6, 4))
sns.barplot(x="pclass", y="survived", hue="sex", data=df_cleaned, errorbar=None, palette="muted")
plt.title("Survival by Ticket Class & Gender")
plt.tight_layout()
plt.savefig("analytics/plots/03_story_sex_pclass.png")
plt.close()

# Chart 2: Box Plot of Fares by Class, colored by Survival Status
plt.figure(figsize=(6, 4))
sns.boxplot(x="pclass", y="fare", hue="survived", data=df_cleaned[df_cleaned['fare'] < 200], palette="Set2")
plt.title("Fare Density Segregated by Survival State")
plt.tight_layout()
plt.savefig("analytics/plots/04_story_fare_distribution.png")
plt.close()

# Chart 3: Violin Distribution of Age across Survival States
plt.figure(figsize=(6, 4))
sns.violinplot(x="survived", y="age", hue="sex", data=df_cleaned, split=True, palette="pastel")
plt.title("Age Distribution Splays Across Survival States")
plt.tight_layout()
plt.savefig("analytics/plots/05_story_age_demographics.png")
plt.close()

# Chart 4: Family Units survival probabilities
df_cleaned['family_size'] = df_cleaned['sibsp'] + df_cleaned['parch'] + 1
plt.figure(figsize=(6, 4))
sns.lineplot(x="family_size", y="survived", data=df_cleaned, marker="o", color="purple", errorbar=None)
plt.title("Impact of Combined Family Unit Size on Survival")
plt.tight_layout()
plt.savefig("analytics/plots/06_story_family_density.png")
plt.close()

print(" 4 analytical charts exported successfully into 'analytics/plots/' directory.")

print("\n--- [Task 6] Exploratory Before/After Standardisation Check ---")
mean_a, std_a = df_cleaned['age'].mean(), df_cleaned['age'].std()
mean_f, std_f = df_cleaned['fare'].mean(), df_cleaned['fare'].std()
age_scaled = (df_cleaned['age'] - mean_a) / std_a
fare_scaled = (df_cleaned['fare'] - mean_f) / std_f
print(f" Before Scaling -> Age Mean: {mean_a:.2f}, Std: {std_a:.2f} | Fare Mean: {mean_f:.2f}, Std: {std_f:.2f}")
print(f" After Scaling  -> Age Mean: {age_scaled.mean():.2f}, Std: {age_scaled.std():.2f} | Fare Mean: {fare_scaled.mean():.2f}, Std: {fare_scaled.std():.2f}")


# =====================================================================
# PART B: PREDICTIVE MODELING
# =====================================================================

print("\n--- [Task 7] Data Splitting & Stratification Choice ---")
# Reload the fresh, un-preprocessed committed fallback data to prevent leaks
df_model = pd.read_csv("analytics/titanic.csv")
X = df_model.drop(columns=['survived', 'alive'])
y = df_model['survived']

# Stratification mirrors target distribution balance (38% Survived vs 62% Perished) across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f" Train Split Shapes: {X_train.shape} | Test Split Shapes: {X_test.shape}")

print("\n--- [Task 8] Building Structural Pipelines (No Data Leakage) ---")
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['sex', 'embarked']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numeric_features),
    ('cat', cat_transformer, categorical_features)
])

print("\n--- [Task 9, 10 & 12] Model Training, Visualization & Grid Search ---")
# Models initialization
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4)
}

clf_results = {}

for name, clf in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    
    if name == "Decision Tree":
        # Render and plot the Decision Tree layout architecture
        plt.figure(figsize=(14, 8))
        plot_tree(clf, max_depth=2, filled=True, rounded=True, class_names=["Perished", "Survived"])
        plt.title("Trained Classification Tree Structural Flowchart")
        plt.tight_layout()
        plt.savefig("analytics/plots/07_decision_tree_graph.png")
        plt.close()
        
        clf_results[name] = {
            'Accuracy': accuracy_score(y_test, preds), 
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds), 
            'F1-Score': f1_score(y_test, preds),
            'AUC': roc_auc_score(y_test, probs), 
            'OOB Score': "N/A"
        }

# Random Forest + GridSearchCV with Out-of-Bag (OOB) handling
rf_base = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf_base)])

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [4, 6, 8, None]
}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_pipeline = grid_search.best_estimator_
rf_preds = best_rf_pipeline.predict(X_test)
rf_probs = best_rf_pipeline.predict_proba(X_test)[:, 1]
oob_val = best_rf_pipeline.named_steps['classifier'].oob_score_

clf_results['Random Forest (Best)'] = {
    'Accuracy': accuracy_score(y_test, rf_preds), 
    'Precision': precision_score(y_test, rf_preds),
    'Recall': recall_score(y_test, rf_preds), 
    'F1-Score': f1_score(y_test, rf_preds),
    'AUC': roc_auc_score(y_test, rf_probs), 
    'OOB Score': f"{oob_val:.4f}"
}

# Serialize the complete, best-performing pipeline asset to disk
joblib.dump(best_rf_pipeline, "analytics/best_titanic_pipeline.joblib")
print(" Successfully dumped the complete Preprocessing + Random Forest pipeline to 'analytics/best_titanic_pipeline.joblib'")

print("\n--- [Task 11] Class Imbalance Intervention Analysis ---")
imbalance_variants = {
    "Baseline Random Forest": Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    "Balanced Class Weights": Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))]),
    "SMOTE Balancing (Train-Fold Only)": ImbPipeline([('prep', preprocessor), ('smote', SMOTE(random_state=42)), ('clf', RandomForestClassifier(random_state=42))])
}

print(f"{'Strategy':<40} │ {'Precision':<10} │ {'Recall':<10} │ {'F1-Score':<10}")
print("─" * 80)
for m_name, p_obj in imbalance_variants.items():
    p_obj.fit(X_train, y_train)
    p_preds = p_obj.predict(X_test)
    print(f"{m_name:<40} │ {precision_score(y_test, p_preds):.4f:<10} │ {recall_score(y_test, p_preds):.4f:<10} │ {f1_score(y_test, p_preds):.4f:<10}")

print("\n--- [Task 13] Regression Side-Task (Fare Prediction Study) ---")
X_reg = df_model.drop(columns=['fare', 'alive'])
y_reg = df_model['fare'].fillna(df_model['fare'].median())
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_num_cols = ['age', 'sibsp', 'parch', 'survived']
reg_cat_cols = ['sex', 'embarked', 'pclass']
reg_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), reg_num_cols),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore'))]), reg_cat_cols)
])

reg_pipeline = Pipeline(steps=[('preprocessor', reg_preprocessor), ('regressor', LinearRegression())])
reg_pipeline.fit(X_reg_train, y_reg_train)
reg_preds = reg_pipeline.predict(X_reg_test)

mae = mean_absolute_error(y_reg_test, reg_preds)
rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
r2 = r2_score(y_reg_test, reg_preds)

# Adjust for array dimensional matrices dynamically
n_samples = len(y_reg_test)
p_features = X_reg_test.shape[1]
adj_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - p_features - 1))

# Residual analysis for heteroscedasticity checking
residuals = y_reg_test - reg_preds
plt.figure(figsize=(6, 4))
plt.scatter(reg_preds, residuals, alpha=0.5, color='purple', edgecolors='k')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Regression Residual Dispersion Plot")
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.tight_layout()
plt.savefig("analytics/plots/08_regression_residuals.png")
plt.close()

# =====================================================================
# FINAL INTEGRATED OUTPUT DISPLAY MATRIX
# =====================================================================
print("\n" + "="*105)
print(f"{'MODEL COMPARISON TABLE (SEPARATED SCALE GROUPS)':^105}")
print("="*105)
print(f"{'[CLASSIFICATION METRICS GROUP]':<25} │ {'Accuracy':<10} │ {'Precision':<10} │ {'Recall':<10} │ {'F1-Score':<10} │ {'AUC':<10} │ {'OOB Score':<10}")
print("─"*105)
for model_name, metrics in clf_results.items():
    print(f"{model_name:<25} │ {metrics['Accuracy']:.4f:<10} │ {metrics['Precision']:.4f:<10} │ {metrics['Recall']:.4f:<10} │ {metrics['F1-Score']:.4f:<10} │ {metrics['AUC']:.4f:<10} │ {metrics['OOB Score']:<10}")

print("─"*105)
print(f"{'[REGRESSION TASK METRICS GROUP]':<25} │ {'MAE':<10} │ {'RMSE':<10} │ {'R²':<10} │ {'Adj R²':<10} │ {'':<10} │ {'':<10}")
print("─"*105)
print(f"{'Fare Prediction Model':<25} │ {mae:.4f:<10} │ {rmse:.4f:<10} │ {r2:.4f:<10} │ {adj_r2:.4f:<10} │ {'N/A':<10} │ {'N/A':<10}")
print("="*105)

print("\nWorkflow processing and model table extraction completed successfully.")

