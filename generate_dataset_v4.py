"""
EIE Dataset Generator v3 — Correlated Scores
===============================================
Parameters within the same environmental cluster are correlated.
When one heavy metal spikes, others in the cluster tend to spike too.

This reflects reality: pollution sources affect groups of params together.
"""

import numpy as np
import pandas as pd
from eie_calculator_v2 import calculate_from_scores, calculate_paysage, calculate_emploi, calculate_infrastructure

# ============================================================================
# CORRELATION CLUSTERS
# ============================================================================
# Within a cluster: if one param gets score 2, others are likely 1 or 2.
# Between clusters: independent.

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

ETENDUE_OPTIONS = ["Ponctuelle", "Locale", "Régionale", "Nationale"]
DUREE_OPTIONS = ["Courte", "Moyenne", "Longue"]


def generate_cluster_scores(n_params, cluster_correlation=0.7):
    """
    Generate correlated scores (0/1/2) for a cluster of parameters.

    Method:
    1. Pick a "cluster state" (0, 1, or 2) — the dominant condition
    2. Each param in the cluster follows the cluster state with probability
       `cluster_correlation`, or deviates randomly otherwise

    This means: if the cluster state is 2 (bad), most params will be 2,
    some might be 1 or 0, giving realistic within-cluster correlation.
    """
    # Base distribution for cluster state: 55/25/20 (same as before)
    cluster_state = np.random.choice([0, 1, 2], p=[0.55, 0.25, 0.20])

    scores = []
    for _ in range(n_params):
        if np.random.random() < cluster_correlation:
            # Follow cluster state
            scores.append(cluster_state)
        else:
            # Deviate: pick a nearby score
            if cluster_state == 0:
                scores.append(np.random.choice([0, 1], p=[0.7, 0.3]))
            elif cluster_state == 1:
                scores.append(np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3]))
            else:  # 2
                scores.append(np.random.choice([1, 2], p=[0.3, 0.7]))

    return scores


def generate_dataset(n_rows=100000, seed=42, correlation=0.7):
    """Generate dataset with correlated parameter scores."""
    np.random.seed(seed)
    print(f"Generating {n_rows:,} rows (correlation={correlation})...")

    data = {}

    # --- Generate correlated scores per cluster ---
    for group_name, clusters in CLUSTERS.items():
        for cluster_name, params in clusters.items():
            # Pre-generate for all rows
            for i_row in range(n_rows):
                # Generate correlated measured scores
                m_scores = generate_cluster_scores(len(params), correlation)
                # Generate correlated rejection scores (same cluster, independent draw)
                r_scores = generate_cluster_scores(len(params), correlation)

                for j, param in enumerate(params):
                    col_m = f"{group_name}_{param}_score_m"
                    col_r = f"{group_name}_{param}_score_r"
                    if col_m not in data:
                        data[col_m] = []
                        data[col_r] = []
                    data[col_m].append(m_scores[j])
                    data[col_r].append(r_scores[j])

    # --- Special rows ---
    data["paysage_modification"] = np.random.choice([0, 2], size=n_rows).tolist()
    data["infrastructure_score"] = np.random.choice([0, 1, 2], size=n_rows, p=[0.55, 0.25, 0.20]).tolist()
    data["emploi_score"] = np.random.choice([0, 1, 2], size=n_rows, p=[0.55, 0.25, 0.20]).tolist()

    # --- Categoricals ---
    data["etendue"] = np.random.choice(ETENDUE_OPTIONS, size=n_rows).tolist()
    data["duree"] = np.random.choice(DUREE_OPTIONS, size=n_rows).tolist()

    df = pd.DataFrame(data)
    print(f"  Features: {len(df.columns)} columns")

    # --- Compute 8 targets ---
    print("  Computing targets...")

    group_params = {}
    for group_name, clusters in CLUSTERS.items():
        group_params[group_name] = []
        for params in clusters.values():
            group_params[group_name].extend(params)

    targets = {f"target_{g}": [] for g in CLUSTERS.keys()}
    targets["target_paysage"] = []
    targets["target_infrastructure"] = []
    targets["target_emploi"] = []

    for i in range(n_rows):
        etendue = df.at[i, "etendue"]
        duree = df.at[i, "duree"]

        for group_name, params in group_params.items():
            m_scores = [df.at[i, f"{group_name}_{p}_score_m"] for p in params]
            r_scores = [df.at[i, f"{group_name}_{p}_score_r"] for p in params]
            result = calculate_from_scores(m_scores, r_scores, etendue, duree)
            targets[f"target_{group_name}"].append(result["importance_relative"])

        targets["target_paysage"].append(calculate_paysage(df.at[i, "paysage_modification"]))
        targets["target_infrastructure"].append(
            calculate_infrastructure(df.at[i, "infrastructure_score"], etendue, duree))
        targets["target_emploi"].append(calculate_emploi(df.at[i, "emploi_score"]))

        if (i + 1) % 20000 == 0:
            print(f"    {i + 1:,} / {n_rows:,} done")

    for col, values in targets.items():
        df[col] = values

    print(f"\n✅ Dataset complete: {df.shape}")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_rows=100000)
    df.to_csv("eie_correlated.csv", index=False)
    print(f"💾 Saved: eie_correlated.csv")

    # Show target distributions
    print(f"\n--- Target distributions ---")
    for col in [c for c in df.columns if c.startswith("target_")]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())

    # Verify correlations exist
    print(f"\n--- Correlation check (Eau heavy metals, measured scores) ---")
    hm_cols = [f"eau_{p}_score_m" for p in ["plomb", "cadmium", "chrome", "cuivre", "zinc", "nickel", "mercure", "arsenic"]]
    corr = df[hm_cols].corr()
    print(f"Average pairwise correlation: {corr.values[np.triu_indices_from(corr.values, k=1)].mean():.3f}")
    print(f"(Should be ~0.4-0.7 if clustering works)")
