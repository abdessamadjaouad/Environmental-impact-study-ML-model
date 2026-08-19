"""
Sentinel-Based Feature Elimination on Correlated Data
======================================================
For each group:
1. Train with all parameters → baseline
2. Keep 1 sentinel per cluster → test accuracy
3. Add back sentinels until ≥95% accuracy
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv("eie_correlated.csv")
print(f"Dataset: {df.shape}")

# Clusters (same as generator)
CLUSTERS = {
    "eau": {
        "physical":     ["temperature", "ph", "turbidite", "conductivite"],
        "organic":      ["dbo5", "dco", "oxygene_dissous"],
        "nutrients":    ["nitrates", "nitrites", "ammoniac", "phosphore", "azote"],
        "heavy_metals": ["plomb", "cadmium", "chrome", "cuivre", "zinc", "nickel", "mercure", "arsenic"],
        "chemical":     ["hydrocarbures"],
    },
    "sol": {
        "physical":     ["ph", "permeabilite"],
        "organic":      ["matiere_organique", "carbone_organique"],
        "heavy_metals": ["plomb", "cadmium", "mercure", "arsenic", "chrome", "cuivre", "zinc", "nickel"],
        "nutrients":    ["azote", "phosphore"],
    },
    "air": {
        "particles":    ["poussieres", "pm10", "pm25"],
        "gases":        ["so2", "nox", "co", "ozone"],
    },
    "population": {
        "radiation":    ["radiation_eoliennes", "radiation_cables", "radiation_onduleurs"],
        "quality":      ["qualite_vie"],
    },
    "sante": {
        "all":          ["poussieres", "risques_electriques", "securite"],
    },
}

SHARED = ["etendue", "duree"]

def get_cols(group, params):
    cols = []
    for p in params:
        cols.append(f"{group}_{p}_score_m")
        cols.append(f"{group}_{p}_score_r")
    return cols


def train_and_eval(group_name, param_list, target_col):
    """Train XGBoost and return accuracy."""
    cols = get_cols(group_name, param_list) + SHARED
    X = df[cols].copy()
    y = df[target_col].copy()

    for c in SHARED:
        X[c] = LabelEncoder().fit_transform(X[c])

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42, verbosity=0
    )
    model.fit(X_tr, y_tr)
    return accuracy_score(y_te, model.predict(X_te))


# ============================================================================
# ANALYSIS PER GROUP
# ============================================================================

all_results = {}

for group_name, clusters in CLUSTERS.items():
    target_col = f"target_{group_name}"
    all_params = [p for cl in clusters.values() for p in cl]

    print(f"\n{'='*60}")
    print(f"  {group_name.upper()} — {len(all_params)} params in {len(clusters)} clusters")
    print(f"{'='*60}")

    # --- Baseline: all params ---
    baseline_acc = train_and_eval(group_name, all_params, target_col)
    print(f"  Baseline ({len(all_params)} params): {baseline_acc:.4f}")

    # --- Sentinel set: 1 param per cluster ---
    sentinels = {}
    for cluster_name, params in clusters.items():
        # Pick the first param as sentinel (they're correlated, any should work)
        sentinels[cluster_name] = [params[0]]
    
    sentinel_params = [p for sl in sentinels.values() for p in sl]
    sentinel_acc = train_and_eval(group_name, sentinel_params, target_col)
    print(f"  Sentinels ({len(sentinel_params)} params, 1/cluster): {sentinel_acc:.4f}")

    # --- If below 95%, try 2 sentinels per cluster ---
    if sentinel_acc < 0.95:
        sentinels_2 = {}
        for cluster_name, params in clusters.items():
            sentinels_2[cluster_name] = params[:min(2, len(params))]
        
        sentinel_2_params = [p for sl in sentinels_2.values() for p in sl]
        sentinel_2_acc = train_and_eval(group_name, sentinel_2_params, target_col)
        print(f"  Sentinels ({len(sentinel_2_params)} params, 2/cluster): {sentinel_2_acc:.4f}")

        # --- If still below 95%, iteratively add from biggest clusters ---
        if sentinel_2_acc < 0.95:
            current = list(sentinel_2_params)
            remaining = [p for p in all_params if p not in current]
            
            print(f"\n  Fine-tuning: adding params until ≥95%...")
            best_acc = sentinel_2_acc

            while remaining and best_acc < 0.95:
                best_param = None
                best_new_acc = 0

                for param in remaining:
                    test_set = current + [param]
                    acc = train_and_eval(group_name, test_set, target_col)
                    if acc > best_new_acc:
                        best_new_acc = acc
                        best_param = param

                current.append(best_param)
                remaining.remove(best_param)
                best_acc = best_new_acc
                status = "✅" if best_acc >= 0.95 else "📈"
                print(f"    {status} +{best_param:25s} → {len(current)} params → {best_acc:.4f}")

            final_params = current
            final_acc = best_acc
        else:
            final_params = sentinel_2_params
            final_acc = sentinel_2_acc
    else:
        final_params = sentinel_params
        final_acc = sentinel_acc

    removed = [p for p in all_params if p not in final_params]

    print(f"\n  ✅ RESULT: {len(all_params)} → {len(final_params)} params (removed {len(removed)})")
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Kept: {final_params}")
    if removed:
        print(f"  Removed: {removed}")

    all_results[group_name] = {
        "all_params": len(all_params),
        "kept_params": final_params,
        "removed_params": removed,
        "kept_count": len(final_params),
        "baseline_accuracy": baseline_acc,
        "final_accuracy": final_acc,
    }


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n\n{'='*60}")
print("  FINAL SUMMARY")
print(f"{'='*60}")

print(f"\n  {'Group':<15} {'Before':>8} {'After':>8} {'Cut':>6} {'Accuracy':>10}")
print(f"  {'-'*52}")

total_b = 0
total_a = 0

for name, r in all_results.items():
    total_b += r["all_params"]
    total_a += r["kept_count"]
    print(f"  {name:<15} {r['all_params']:>8} {r['kept_count']:>8} {r['all_params']-r['kept_count']:>6} {r['final_accuracy']:>10.4f}")

print(f"  {'paysage':<15} {'1':>8} {'1':>8} {'0':>6}")
print(f"  {'infrastructure':<15} {'1':>8} {'1':>8} {'0':>6}")
print(f"  {'emploi':<15} {'1':>8} {'1':>8} {'0':>6}")
total_b += 3; total_a += 3

print(f"  {'-'*52}")
print(f"  {'TOTAL':<15} {total_b:>8} {total_a:>8} {total_b-total_a:>6}")
print(f"\n  🎉 {total_b} → {total_a} rows ({(1 - total_a/total_b)*100:.0f}% reduction!)")

# Save
output = {"groups": {}}
for name, r in all_results.items():
    output["groups"][name] = {
        "kept": r["kept_params"],
        "removed": r["removed_params"],
        "accuracy": r["final_accuracy"],
    }
output["total_original"] = total_b
output["total_reduced"] = total_a

json.dump(output, open("sentinel_parameters.json", "w"), indent=2, ensure_ascii=False)
print(f"\n💾 Saved: sentinel_parameters.json")
