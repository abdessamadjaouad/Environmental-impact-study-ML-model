"""
EIE Dataset Generator v2 — Per-Parameter
==========================================
Generates a dataset where each row = one complete filled Excel sheet.

Structure:
  103 feature columns (49 params × 2 scores + 3 specials + 2 categoricals)
  8 target columns (one verdict per component group)

Score distribution: 55% score 0, 25% score 1, 20% score 2
"""

import numpy as np
import pandas as pd
from eie_calculator_v2 import calculate_from_scores, calculate_paysage, calculate_emploi, calculate_infrastructure

# ============================================================================
# ALL 49 STANDARD PARAMETERS (name, range_min, range_max)
# Grouped by component
# ============================================================================

GROUPS = {
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

ETENDUE_OPTIONS = ["Ponctuelle", "Locale", "Régionale", "Nationale"]
DUREE_OPTIONS = ["Courte", "Moyenne", "Longue"]

# Score probabilities: 55% normal, 25% slight, 20% major
SCORE_PROBS = [0.55, 0.25, 0.20]


# ============================================================================
# GENERATOR
# ============================================================================

def generate_dataset(n_rows=100000, seed=42):
    """Generate the full dataset."""
    np.random.seed(seed)

    print(f"Generating {n_rows:,} rows...")

    # Pre-generate all scores for all parameters
    data = {}

    # --- Standard parameters (49 × 2 = 98 columns) ---
    for group_name, params in GROUPS.items():
        for param in params:
            col_m = f"{group_name}_{param}_score_m"
            col_r = f"{group_name}_{param}_score_r"
            data[col_m] = np.random.choice([0, 1, 2], size=n_rows, p=SCORE_PROBS)
            data[col_r] = np.random.choice([0, 1, 2], size=n_rows, p=SCORE_PROBS)

    # --- Special rows (3 columns) ---
    data["paysage_modification"] = np.random.choice([0, 2], size=n_rows)
    data["infrastructure_score"] = np.random.choice([0, 1, 2], size=n_rows, p=SCORE_PROBS)
    data["emploi_score"] = np.random.choice([0, 1, 2], size=n_rows, p=SCORE_PROBS)

    # --- Categoricals (2 columns) ---
    data["etendue"] = np.random.choice(ETENDUE_OPTIONS, size=n_rows)
    data["duree"] = np.random.choice(DUREE_OPTIONS, size=n_rows)

    df = pd.DataFrame(data)
    print(f"  Features: {len(df.columns)} columns")

    # --- Compute 8 targets ---
    print("  Computing targets...")

    targets = {
        "target_eau": [],
        "target_sol": [],
        "target_air": [],
        "target_population": [],
        "target_sante": [],
        "target_paysage": [],
        "target_infrastructure": [],
        "target_emploi": [],
    }

    for i in range(n_rows):
        etendue = df.at[i, "etendue"]
        duree = df.at[i, "duree"]

        # Standard groups: collect scores → calculate verdict
        for group_name, params in GROUPS.items():
            m_scores = [df.at[i, f"{group_name}_{p}_score_m"] for p in params]
            r_scores = [df.at[i, f"{group_name}_{p}_score_r"] for p in params]
            result = calculate_from_scores(m_scores, r_scores, etendue, duree)
            targets[f"target_{group_name}"].append(result["importance_relative"])

        # Special groups
        targets["target_paysage"].append(
            calculate_paysage(df.at[i, "paysage_modification"]))
        targets["target_infrastructure"].append(
            calculate_infrastructure(df.at[i, "infrastructure_score"], etendue, duree))
        targets["target_emploi"].append(
            calculate_emploi(df.at[i, "emploi_score"]))

        if (i + 1) % 20000 == 0:
            print(f"    {i + 1:,} / {n_rows:,} done")

    # Add targets to dataframe
    for col, values in targets.items():
        df[col] = values

    print(f"\n✅ Dataset complete: {df.shape}")
    return df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    df = generate_dataset(n_rows=100000)

    # Save
    df.to_csv("eie_per_parameter.csv", index=False)
    print(f"\n💾 Saved: eie_per_parameter.csv")

    # Show summary
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Feature columns: {df.shape[1] - 8}")
    print(f"Target columns: 8")

    print(f"\n--- Target distributions ---")
    for col in [c for c in df.columns if c.startswith("target_")]:
        print(f"\n{col}:")
        print(df[col].value_counts().to_string())
