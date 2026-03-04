"""
EIE Impact Classification — XGBoost Model
==========================================
Simple, clean pipeline: Load → Prepare → Train → Evaluate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# =============================================
# STEP 1: LOAD THE DATASET
# =============================================

# Kaggle path (use this on Kaggle)
# df = pd.read_csv("/kaggle/input/environmental-impact-assessment-eie-dataset/eie_unified.csv")

# Local path (use this on your machine)
df = pd.read_csv("eie_unified.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nColumn types:\n{df.dtypes}")

# =============================================
# STEP 2: DATA PREPARATION
# =============================================

# Our target variable
target = "importance_relative"

# These columns are intermediate results computed FROM the target
# Including them would be "cheating" (data leakage)
columns_to_drop = [
    target,               # the target itself
    "importance",         # directly determines importance_relative
    "sensitivity",        # intermediate calculation
    "impact_apprehende",  # intermediate calculation
    "value_initial",      # intermediate calculation
    "intensity_class",    # intermediate calculation
]

# Separate features (X) and target (y)
y_raw = df[target]
X = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

print(f"\nFeatures: {list(X.columns)}")
print(f"Target classes: {y_raw.unique()}")

# =============================================
# STEP 3: ENCODING (text → numbers)
# =============================================

# Encode the target: Mineure=0, Moyenne=2, Majeure=1 (alphabetical)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

print(f"\nLabel mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# Encode categorical feature columns (phase, component, etendue, duree)
feature_encoders = {}
for col in X.select_dtypes(include="object").columns:
    feature_encoders[col] = LabelEncoder()
    X[col] = feature_encoders[col].fit_transform(X[col])
    print(f"  Encoded '{col}': {dict(zip(feature_encoders[col].classes_, range(len(feature_encoders[col].classes_))))}")

print(f"\nFinal X shape: {X.shape}")
print(f"Final y shape: {y.shape}")

# =============================================
# STEP 4: TRAIN / TEST SPLIT
# =============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80% train, 20% test
    random_state=42,     # reproducible results
    stratify=y           # keep class proportions equal in both sets
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# =============================================
# STEP 5: TRAIN THE XGBOOST MODEL
# =============================================

model = XGBClassifier(
    n_estimators=300,      # number of trees
    max_depth=8,           # max tree depth
    learning_rate=0.1,     # step size
    eval_metric="mlogloss",
    random_state=42,
    verbosity=0            # suppress training logs
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# =============================================
# STEP 6: TESTING & EVALUATION
# =============================================

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =============================================
# STEP 7: RESULTS VISUALIZATION
# =============================================

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, ax=axes[0])
axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")

# --- Feature Importance ---
importances = model.feature_importances_
feature_names = list(X.columns)
sorted_idx = np.argsort(importances)  # ascending order

axes[1].barh(range(len(sorted_idx)), importances[sorted_idx],
             color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx))))
axes[1].set_yticks(range(len(sorted_idx)))
axes[1].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[1].set_title("Feature Importance", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("xgboost_results.png", dpi=150)
plt.show()
print("\n📊 Chart saved: xgboost_results.png")

# --- Print top 5 features ---
print("\nTop 5 Most Important Features:")
for rank, idx in enumerate(np.argsort(importances)[::-1][:5], 1):
    print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")
