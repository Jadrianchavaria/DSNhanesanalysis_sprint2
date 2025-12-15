import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# 1. Load data
df = pd.read_csv("cleaned_data_safe.csv", low_memory=False)

# 2. Create target column 
# 1 = early metabolic dysfunction 
# 0 = no early metabolic dysfunction
df = df[df["LBXSGL"].notna()].copy()
df["early_metabolic_dysfunction"] = (df["LBXSGL"] >= 100).astype(int)


feature_cols = [
    "RIDAGEYR",   # age
    "RIAGENDR",   # sex
    "BMXBMI",     # BMI
    "BMXWAIST",   # waist circumference
    "LBXTC",      # total cholesterol
    "LBDLDL",     # LDL
    "LBDHDD",     # HDL
    "LBXSIR",     # insulin
    "BPXSY1",     # systolic BP
    "BPXDI1",     # diastolic BP
]

# Only keep columns that actually exist in the file
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df["early_metabolic_dysfunction"]

# 4. Handle missing values and make sure everything is numeric
X = X.select_dtypes(include="number")
X = X.fillna(X.median())

# 5. Train/test split (70% train, 30% test, stratified by label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

def evaluate(name, y_true, y_pred, y_proba=None):
    """Print basic metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    if auc is not None:
        print(f"AUC      : {auc:.3f}")

    return acc, prec, rec, auc

# 6. Baseline model: always predict the majority class from the TRAIN set
majority_class = y_train.value_counts().idxmax()
y_pred_baseline = [majority_class] * len(y_test)
baseline_metrics = evaluate("Baseline (majority class)", y_test, y_pred_baseline)

# 7. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
lr_metrics = evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr)

# 8. Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
rf_metrics = evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf)

# 9. XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
xgb_metrics = evaluate("XGBoost", y_test, y_pred_xgb, y_proba_xgb)

print("\nDone training and evaluating all models.")


#   FEATURE IMPORTANCE + VISUALIZATIONS


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# RANDOM FOREST FEATURE IMPORTANCE


# Put features + importance into a DataFrame
rf_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False).head(10)

# Print top features
print("\nTop 10 Random Forest Features:")
print(rf_importance)

# ---- Plot RF Feature Importance ----
plt.figure(figsize=(10,5))
plt.barh(rf_importance["feature"], rf_importance["importance"])
plt.xlabel("Importance Score")
plt.title("Top 10 Random Forest Features")
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()   



# XGBOOST FEATURE IMPORTANCE

# Put features + importance into a DataFrame
xgb_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb.feature_importances_
}).sort_values(by="importance", ascending=False).head(10)

# Print top features
print("\nTop 10 XGBoost Features:")
print(xgb_importance)

# - XGB Feature Importance 
plt.figure(figsize=(10,5))
plt.barh(xgb_importance["feature"], xgb_importance["importance"])
plt.xlabel("Importance Score")
plt.title("Top 10 XGBoost Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()   


# MODEL COMPARISON BAR CHART

import matplotlib.pyplot as plt

# Model names and accuracies
model_names = ["Baseline", "Logistic Regression", "Random Forest", "XGBoost"]
accuracies = [0.515, 0.677, 0.720, 0.769]

plt.figure(figsize=(8,5))
plt.bar(model_names, accuracies)

plt.title("Model Comparison: Accuracy Scores")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)  # accuracy range

# Show values on top of each bar
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.show()
