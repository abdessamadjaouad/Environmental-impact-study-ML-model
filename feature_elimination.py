"""
EIE Feature Elimination — Per-Group Analysis
==============================================
For each component group:
  1. Train XGBoost with all parameters
  2. Rank by feature importance
  3. Iteratively remove least important features
  4. Find minimum set at ≥95% accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv("eie_per_parameter.csv")
print(f"Dataset: {df.shape}")

# Groups and their parameter columns
GROUPS = {
    "eau": [c for c in df.columns if c.startswith("eau_") and not c.startswith("target")],
    "sol": [c for c in df.columns if c.startswith("sol_") and not c.startswith("target")],
    "air": [c for c in df.columns if c.startswith("air_") and not c.startswith("target")],
    "population": [c for c in df.columns if c.startswith("population_") and not c.startswith("target")],
    "sante": [c for c in df.columns if c.startswith("sante_") and not c.startswith("target")],
}

# Shared features (etendue + duree affect all groups)
SHARED = ["etendue", "duree"]


# ============================================================================
# ANALYSIS PER GROUP
# ============================================================================

def analyze_group(group_name, feature_cols, target_col, threshold=0.95):
    """
    Full analysis for one group:
    1. Train with all features → baseline accuracy
    2. Rank features by importance
    3. Remove features one by one until accuracy drops below threshold
    """
    print(f"\n{'='*60}")
    print(f"  GROUP: {group_name.upper()}")
    print(f"{'='*60}")

    # Prepare data
    all_features = feature_cols + SHARED
    X = df[all_features].copy()
    y = df[target_col].copy()

    # Encode categoricals
    le_etendue = LabelEncoder()
    le_duree = LabelEncoder()
    X["etendue"] = le_etendue.fit_transform(X["etendue"])
    X["duree"] = le_duree.fit_transform(X["duree"])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Step 1: Baseline with ALL features ---
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n  Baseline: {len(feature_names)} features → {baseline_acc:.4f} accuracy")

    # --- Step 2: Feature importance ranking ---
    importances = model.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Feature importance ranking:")
    for i, (name, imp) in enumerate(ranked, 1):
        bar = "█" * int(imp * 100)
        print(f"    {i:2d}. {name:40s} {imp:.4f} {bar}")

    # --- Step 3: Iterative elimination ---
    print(f"\n  --- Iterative elimination (threshold={threshold:.0%}) ---")

    current_features = [name for name, _ in ranked]  # start with all, sorted by importance
    results = []

    # Record baseline
    results.append({
        "n_features": len(current_features),
        "removed": "-",
        "accuracy": baseline_acc,
        "features": list(current_features),
    })

    # Remove from the LEAST important end
    while len(current_features) > 2:  # keep at least etendue + duree
        # Remove the least important feature
        removed = current_features.pop()

        # Retrain
        X_sub = df[current_features].copy()
        for col in ["etendue", "duree"]:
            if col in X_sub.columns:
                X_sub[col] = LabelEncoder().fit_transform(X_sub[col])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub.values, y, test_size=0.2, random_state=42, stratify=y
        )

        m = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )
        m.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, m.predict(X_te))

        results.append({
            "n_features": len(current_features),
            "removed": removed,
            "accuracy": acc,
            "features": list(current_features),
        })

        status = "✅" if acc >= threshold else "❌"
        print(f"    {status} {len(current_features):2d} features (removed {removed:35s}) → {acc:.4f}")

        # Stop if we drop below threshold
        if acc < threshold:
            # Put the feature back — it was needed
            current_features.append(removed)
            print(f"\n  ⚡ Stopping: removing '{removed}' drops below {threshold:.0%}")
            break

    # Final result
    min_set = None
    min_acc = 0
    for r in reversed(results):
        if r["accuracy"] >= threshold:
            min_set = r["features"]
            min_acc = r["accuracy"]
            break

    # If no result met threshold, use all features
    if min_set is None:
        min_set = feature_names
        min_acc = baseline_acc
        print(f"\n  ⚠️ Baseline ({baseline_acc:.4f}) already below {threshold:.0%}, keeping all features")

    param_features = [f for f in min_set if f not in SHARED]
    shared_in_set = [f for f in min_set if f in SHARED]

    print(f"\n  RESULT: {len(param_features)} params + {len(shared_in_set)} shared = {len(min_set)} total features")
    print(f"  Accuracy: {min_acc:.4f}")
    print(f"  Kept parameters:")
    for f in param_features:
        print(f"    ✅ {f}")
    print(f"  Removed: {len(feature_cols) - len(param_features)} parameters")

    return {
        "group": group_name,
        "baseline_features": len(feature_cols) + len(SHARED),
        "baseline_accuracy": baseline_acc,
        "minimum_features": min_set,
        "minimum_param_count": len(param_features),
        "minimum_accuracy": min_acc,
        "removed_count": len(feature_cols) - len(param_features),
        "elimination_history": results,
        "importance_ranking": ranked,
    }


# ============================================================================
# RUN ALL GROUPS
# ============================================================================

all_results = {}

for group_name, feature_cols in GROUPS.items():
    target_col = f"target_{group_name}"
    result = analyze_group(group_name, feature_cols, target_col)
    all_results[group_name] = result

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n\n{'='*60}")
print("  FINAL SUMMARY")
print(f"{'='*60}")
print(f"\n  {'Group':<15} {'Before':>8} {'After':>8} {'Removed':>8} {'Accuracy':>10}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

total_before = 0
total_after = 0

for name, r in all_results.items():
    before = len([c for c in GROUPS[name]])
    after = r["minimum_param_count"]
    removed = r["removed_count"]
    acc = r["minimum_accuracy"]
    total_before += before
    total_after += after
    print(f"  {name:<15} {before:>8} {after:>8} {removed:>8} {acc:>10.4f}")

# Add special groups (can't be reduced)
print(f"  {'paysage':<15} {'1':>8} {'1':>8} {'0':>8} {'N/A':>10}")
print(f"  {'infrastructure':<15} {'1':>8} {'1':>8} {'0':>8} {'N/A':>10}")
print(f"  {'emploi':<15} {'1':>8} {'1':>8} {'0':>8} {'N/A':>10}")

total_before += 3
total_after += 3

print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'TOTAL':<15} {total_before:>8} {total_after:>8} {total_before - total_after:>8}")
print(f"\n  Reduction: {total_before} → {total_after} parameters ({(1 - total_after/total_before)*100:.0f}% fewer inputs!)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save the minimum feature set for each group
min_features = {}
for name, r in all_results.items():
    min_features[name] = {
        "features": r["minimum_features"],
        "accuracy": r["minimum_accuracy"],
    }

import json
with open("minimum_features.json", "w") as f:
    json.dump(min_features, f, indent=2)

print(f"\n💾 Feature sets saved to: minimum_features.json")

# ============================================================================
# PLOT: Elimination curves
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(all_results.items()):
    ax = axes[idx]
    history = r["elimination_history"]

    n_feats = [h["n_features"] for h in history]
    accs = [h["accuracy"] for h in history]

    ax.plot(n_feats, accs, "b-o", markersize=4)
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95% threshold")
    ax.axvline(x=len(r["minimum_features"]), color="g", linestyle="--", alpha=0.7, label=f"Min: {len(r['minimum_features'])}")

    ax.set_title(f"{name.upper()}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.85, 1.01)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# Hide unused subplot
if len(all_results) < len(axes):
    axes[-1].set_visible(False)

plt.suptitle("Feature Elimination Curves — Accuracy vs # Features", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("elimination_curves.png", dpi=150)
print(f"📊 Plot saved: elimination_curves.png")
