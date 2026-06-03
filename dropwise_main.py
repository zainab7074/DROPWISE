"""
DROPWISE: Dissolved Oxygen Real-time Prediction with Intelligent Sensor-based Evaluation
=========================================================================================
A machine learning framework for classifying dissolved oxygen (DO) levels
in freshwater aquaculture ecosystems using IoT sensor data.

Models: Random Forest, LightGBM, CatBoost, Logistic Regression
Validation: 15-fold TimeSeriesSplit (time-aware cross-validation)
Dataset: IoT Monitoring of Water Quality and Tilapia (Kaggle)

Authors: Ritu Chauhan, Zainab Sarfi, Dhananjay Singh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD AND CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("DROPWISE — Loading and Cleaning Data")
print("=" * 60)

df = pd.read_csv("Data_Water.csv")

# Parse datetime and sort chronologically (required for TimeSeriesSplit)
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values("datetime").reset_index(drop=True)

# Standardise column names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("/", "", regex=False)
    .str.replace("%", "percent")
    .str.replace("°", "deg", regex=False)
)

# Remove duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

print(f"✅ Columns after cleaning: {df.columns.tolist()}")
print(f"✅ Dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: CREATE TARGET VARIABLE (DO Category)
# ─────────────────────────────────────────────────────────────────────────────

# Identify the dissolved oxygen column
do_col = next((c for c in df.columns if "dissolved_oxygen" in c), None)
if do_col is None:
    raise ValueError("Could not find dissolved oxygen column. Check your CSV.")

print(f"\n✅ Using DO column: '{do_col}'")

# Bin into Low / Medium / High per aquaculture thresholds
#   Low:    DO ≤ 6.5 mg/L  (stressful for tilapia)
#   Medium: 6.5 < DO ≤ 7.5 mg/L
#   High:   DO > 7.5 mg/L  (optimal)
df["do_category"] = pd.cut(
    df[do_col],
    bins=[0, 6.5, 7.5, 100],
    labels=["Low", "Medium", "High"]
)

df = df.dropna(subset=["do_category"]).reset_index(drop=True)

le_target = LabelEncoder()
df["do_label"] = le_target.fit_transform(df["do_category"])  # Low=0, Medium=1, High=2

print("\n✅ DO category distribution:")
print(df["do_category"].value_counts())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: FEATURE SELECTION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "average_fish_weight_g",
    "survival_rate_percent",
    "disease_occurrence_cases",
    "temperature_degc",
    "precipitation_inches",
    "ph",
    "turbidity_ntu",
]

# Keep only features that exist in the dataset
FEATURES = [f for f in FEATURES if f in df.columns]
print(f"\n✅ Features used: {FEATURES}")

df = df.dropna(subset=FEATURES).reset_index(drop=True)

X = df[FEATURES].copy()
y = df["do_label"].values

# Normalise features to [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

print(f"✅ Final dataset size: {X_scaled.shape[0]} rows, {X_scaled.shape[1]} features")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DROPWISE — Exploratory Data Analysis")
print("=" * 60)

# 4a. Pairplot of key water quality parameters
pairplot_cols = [c for c in [do_col, "temperature_degc", "turbidity_ntu", "ph"] if c in df.columns]
if len(pairplot_cols) >= 2:
    sns.pairplot(df[pairplot_cols].dropna())
    plt.suptitle("Pairplot: Key Water Quality Parameters", y=1.02)
    plt.tight_layout()
    plt.savefig("figure_pairplot.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: figure_pairplot.png")

# 4b. Correlation heatmap
corr_features = FEATURES + [do_col]
corr_features = [c for c in corr_features if c in df.columns]
plt.figure(figsize=(10, 8))
sns.heatmap(df[corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap of Water Quality Features")
plt.tight_layout()
plt.savefig("figure_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: figure_correlation_heatmap.png")

# 4c. Rolling average time series of DO
if "datetime" in df.columns and do_col in df.columns:
    df_plot = df[["datetime", do_col]].dropna().copy()
    df_plot["do_rolling_24h"] = df_plot[do_col].rolling(window=24).mean()
    plt.figure(figsize=(14, 5))
    plt.plot(df_plot["datetime"], df_plot[do_col], alpha=0.4, label="DO (raw)", color="steelblue")
    plt.plot(df_plot["datetime"], df_plot["do_rolling_24h"], color="red", linewidth=2, label="24-hr Rolling Avg")
    plt.title("Dissolved Oxygen Over Time (with 24-hr Rolling Average)")
    plt.xlabel("Date")
    plt.ylabel("DO (mg/L)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure_do_timeseries.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: figure_do_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: MODEL TRAINING WITH 15-FOLD TIME-AWARE CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DROPWISE — Model Training (15-fold TimeSeriesSplit)")
print("=" * 60)

tscv = TimeSeriesSplit(n_splits=15)

MODELS = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(random_seed=42, verbose=0),
    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="ovr", random_state=42),
}

results_summary = []
fold_accuracies = {name: [] for name in MODELS}

for model_name, model in MODELS.items():
    print(f"\n🔍 Training: {model_name}")
    fold_accs = []
    fold_f1s = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)

    mean_acc = np.mean(fold_accs)
    mean_f1 = np.mean(fold_f1s)
    fold_accuracies[model_name] = fold_accs

    print(f"  Avg Accuracy : {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"  Avg F1 Score : {mean_f1:.4f}")

    results_summary.append({
        "Model": model_name,
        "Accuracy": f"{mean_acc*100:.2f}%",
        "F1-Score": round(mean_f1, 4),
    })

# Print summary table
print("\n" + "=" * 60)
print("DROPWISE — Model Comparison (15-fold Time-Aware CV)")
print("=" * 60)
results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: TRAIN FINAL RANDOM FOREST ON FULL DATA & SAVE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DROPWISE — Training Final Random Forest Model")
print("=" * 60)

best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_scaled, y)

joblib.dump(best_model, "rf_model_timeaware.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_target, "label_encoder.pkl")

print("✅ Saved: rf_model_timeaware.pkl")
print("✅ Saved: scaler.pkl")
print("✅ Saved: label_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: EVALUATION PLOTS (Confusion Matrix, ROC, Feature Importance, Fold-Wise)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DROPWISE — Generating Evaluation Plots")
print("=" * 60)

# Use last fold split for evaluation plots
train_idx_final, test_idx_final = list(tscv.split(X_scaled))[-1]
X_train_f, X_test_f = X_scaled.iloc[train_idx_final], X_scaled.iloc[test_idx_final]
y_train_f, y_test_f = y[train_idx_final], y[test_idx_final]

best_model.fit(X_train_f, y_train_f)
y_pred_f = best_model.predict(X_test_f)

class_names = le_target.classes_

# Confusion Matrix
cm = confusion_matrix(y_test_f, y_pred_f, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues")
plt.title("Random Forest — Confusion Matrix")
plt.tight_layout()
plt.savefig("figure_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: figure_confusion_matrix.png")

# ROC Curves (One-vs-Rest)
y_test_bin = label_binarize(y_test_f, classes=[0, 1, 2])
y_proba = best_model.predict_proba(X_test_f)

plt.figure(figsize=(9, 6))
colors = ["steelblue", "darkorange", "green"]
for i, (cls, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Random Forest (One-vs-Rest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: figure_roc_curves.png")

# Feature Importance
importances = best_model.feature_importances_
feat_df = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=True)
plt.figure(figsize=(8, 5))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="Blues_d")
plt.title("Random Forest — Feature Importance")
plt.tight_layout()
plt.savefig("figure_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: figure_feature_importance.png")

# Fold-wise accuracy plot
plt.figure(figsize=(12, 5))
for model_name, accs in fold_accuracies.items():
    plt.plot(range(1, len(accs) + 1), accs, marker="o", label=model_name)
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Fold-wise Accuracy — 15-fold Time-Aware Cross-Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figure_foldwise_accuracy.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: figure_foldwise_accuracy.png")

print("\n" + "=" * 60)
print("DROPWISE — Pipeline Complete ✅")
print("Run `python dropwise_app.py` to launch the Gradio interface.")
print("=" * 60)
