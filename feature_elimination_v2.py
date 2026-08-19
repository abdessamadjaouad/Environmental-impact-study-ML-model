"""
EIE Feature Elimination v2 — By Parameter (Row)
=================================================
Eliminates ENTIRE PARAMETERS (rows), not individual score columns.

Each parameter = one row the user fills in Excel = both score_m + score_r.
When we remove "temperature", the user no longer needs to fill that row at all.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv("eie_per_parameter.csv")
print(f"Dataset: {df.shape}")

# Define parameter ROWS per group (each row = score_m + score_r)
PARAM_ROWS = {
    "eau": [
        "temperature", "ph", "turbidite", "conductivite", "dbo5", "dco",
        "oxygene_dissous", "nitrates", "nitrites", "ammoniac", "phosphore",
        "azote", "plomb", "cadmium", "chrome", "cuivre", "zinc", "nickel",
        "mercure", "arsenic", "hydrocarbures",
    ],
    "sol": [
        "ph", "permeabilite", "matiere_organique", "carbone_organique",
        "plomb", "cadmium", "mercure", "arsenic", "chrome", "cuivre",
        "zinc", "nickel", "azote", "phosphore",
    ],
    "air": [
        "poussieres", "pm10", "pm25", "so2", "nox", "co", "ozone",
    ],
    "population": [
        "radiation_eoliennes", "radiation_cables", "radiation_onduleurs", "qualite_vie",
    ],
    "sante": [
        "poussieres", "risques_electriques", "securite",
    ],
}

SHARED = ["etendue", "duree"]


def get_columns_for_param(group, param):
    """Get both score columns for one parameter row."""
    return [f"{group}_{param}_score_m", f"{group}_{param}_score_r"]


def get_all_columns(group, params):
    """Get all score columns for a list of parameters."""
    cols = []
    for p in params:
        cols.extend(get_columns_for_param(group, p))
    return cols


# ============================================================================
# PARAMETER-LEVEL ELIMINATION
# ============================================================================

def analyze_group(group_name, param_list, target_col, threshold=0.95):
    """
    For each group:
    1. Train with ALL parameters → baseline
    2. Get parameter-level importance (sum of score_m + score_r importance)
    3. Remove entire parameters until accuracy drops below threshold
    """
    print(f"\n{'='*60}")
    print(f"  {group_name.upper()} — {len(param_list)} parameters (rows)")
    print(f"{'='*60}")

    # All feature columns for this group
    all_param_cols = get_all_columns(group_name, param_list)
    all_features = all_param_cols + SHARED
    feature_names = list(all_features)

    # Prepare data
    X = df[all_features].copy()
    y = df[target_col].copy()

    le_etendue = LabelEncoder()
    le_duree = LabelEncoder()
    X["etendue"] = le_etendue.fit_transform(X["etendue"])
    X["duree"] = le_duree.fit_transform(X["duree"])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Baseline ---
    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n  Baseline: {len(param_list)} params → {baseline_acc:.4f} accuracy")

    # --- Parameter-level importance (sum score_m + score_r) ---
    importances = dict(zip(feature_names, model.feature_importances_))

    param_importance = {}
    for param in param_list:
        cols = get_columns_for_param(group_name, param)
        total_imp = sum(importances.get(c, 0) for c in cols)
        param_importance[param] = total_imp

    # Sort by importance (descending)
    ranked_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n  Parameter importance (combined score_m + score_r):")
    for i, (param, imp) in enumerate(ranked_params, 1):
        bar = "█" * int(imp * 50)
        print(f"    {i:2d}. {param:30s} {imp:.4f} {bar}")

    # --- Iterative elimination: remove entire parameters ---
    print(f"\n  --- Removing parameters (threshold={threshold:.0%}) ---")

    current_params = [p for p, _ in ranked_params]  # most important first
    results = [{"n_params": len(current_params), "removed": "-",
                "accuracy": baseline_acc, "params": list(current_params)}]

    while len(current_params) > 1:
        # Remove the LEAST important parameter
        removed = current_params.pop()

        # Rebuild feature set
        remaining_cols = get_all_columns(group_name, current_params) + SHARED
        X_sub = df[remaining_cols].copy()
        for col in SHARED:
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

        results.append({"n_params": len(current_params), "removed": removed,
                        "accuracy": acc, "params": list(current_params)})

        status = "✅" if acc >= threshold else "❌"
        print(f"    {status} {len(current_params):2d} params (removed {removed:25s}) → {acc:.4f}")

        if acc < threshold:
            current_params.append(removed)
            print(f"\n  ⚡ Stopping: '{removed}' is needed!")
            break

    # Find minimum set
    min_set = None
    min_acc = 0
    for r in reversed(results):
        if r["accuracy"] >= threshold:
            min_set = r["params"]
            min_acc = r["accuracy"]
            break

    if min_set is None:
        min_set = param_list
        min_acc = baseline_acc

    removed_params = [p for p in param_list if p not in min_set]

    print(f"\n  ✅ RESULT: {len(param_list)} → {len(min_set)} parameters")
    print(f"  Accuracy: {min_acc:.4f}")
    print(f"  Kept ({len(min_set)}):")
    for p in min_set:
        print(f"    ✅ {p}")
    if removed_params:
        print(f"  Removed ({len(removed_params)}):")
        for p in removed_params:
            print(f"    ❌ {p}")

    return {
        "group": group_name,
        "total_params": len(param_list),
        "kept_params": min_set,
        "removed_params": removed_params,
        "kept_count": len(min_set),
        "removed_count": len(removed_params),
        "baseline_accuracy": baseline_acc,
        "final_accuracy": min_acc,
        "history": results,
        "ranking": ranked_params,
    }


# ============================================================================
# RUN ALL
# ============================================================================

all_results = {}
for group_name, params in PARAM_ROWS.items():
    target = f"target_{group_name}"
    all_results[group_name] = analyze_group(group_name, params, target)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'='*60}")
print("  FINAL SUMMARY — Parameters the user needs to fill")
print(f"{'='*60}")

total_before = 0
total_after = 0

print(f"\n  {'Group':<15} {'Before':>8} {'After':>8} {'Removed':>8} {'Accuracy':>10}")
print(f"  {'-'*51}")

for name, r in all_results.items():
    total_before += r["total_params"]
    total_after += r["kept_count"]
    print(f"  {name:<15} {r['total_params']:>8} {r['kept_count']:>8} {r['removed_count']:>8} {r['final_accuracy']:>10.4f}")

print(f"  {'paysage':<15} {'1':>8} {'1':>8} {'0':>8} {'binary':>10}")
print(f"  {'infrastructure':<15} {'1':>8} {'1':>8} {'0':>8} {'binary':>10}")
print(f"  {'emploi':<15} {'1':>8} {'1':>8} {'0':>8} {'binary':>10}")
total_before += 3
total_after += 3

print(f"  {'-'*51}")
print(f"  {'TOTAL':<15} {total_before:>8} {total_after:>8} {total_before - total_after:>8}")
print(f"\n  🎉 Reduction: {total_before} → {total_after} rows ({(1 - total_after/total_before)*100:.0f}% fewer!)")

# ============================================================================
# SAVE
# ============================================================================

import json
output = {}
for name, r in all_results.items():
    output[name] = {
        "kept": r["kept_params"],
        "removed": r["removed_params"],
        "accuracy": r["final_accuracy"],
    }
json.dump(output, open("minimum_parameters.json", "w"), indent=2)
print(f"\n💾 Saved: minimum_parameters.json")

# ============================================================================
# PLOT
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(all_results.items()):
    ax = axes[idx]
    history = r["history"]

    n_params = [h["n_params"] for h in history]
    accs = [h["accuracy"] for h in history]

    ax.plot(n_params, accs, "b-o", markersize=5)
    ax.axhline(y=0.95, color="r", linestyle="--", alpha=0.7, label="95% threshold")
    ax.axvline(x=r["kept_count"], color="g", linestyle="--", alpha=0.7,
               label=f"Min: {r['kept_count']} params")

    ax.set_title(f"{name.upper()} ({r['total_params']}→{r['kept_count']})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of parameters")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.80, 1.01)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

axes[-1].set_visible(False)

plt.suptitle("Parameter Elimination — How many rows can we remove?",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("parameter_elimination.png", dpi=150)
print(f"📊 Saved: parameter_elimination.png")
