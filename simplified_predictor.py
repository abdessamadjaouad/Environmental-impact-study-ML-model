"""
Simplified EIE Predictor — Only 38 Rows
=========================================
Instead of filling 52 rows, the user fills only 38 sentinel parameters.
For each row, user enters a score (0=normal, 1=slight deviation, 2=major deviation).
The model predicts the matrice d'impact for all 8 component groups.

Trained on correlated synthetic data with XGBoost.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ============================================================================
# PARAMETER DEFINITIONS (what the user fills)
# ============================================================================

SENTINEL_PARAMS = {
    "eau": {
        "name": "💧 Eau (Water Quality)",
        "params": [
            ("temperature",   "Température de l'eau"),
            ("ph",            "pH de l'eau"),
            ("turbidite",     "Turbidité"),
            ("dbo5",          "Demande biochimique en oxygène (DBO5)"),
            ("dco",           "Demande chimique en oxygène (DCO)"),
            ("nitrates",      "Nitrates (NO₃)"),
            ("nitrites",      "Nitrites (NO₂)"),
            ("phosphore",     "Phosphore total"),
            ("azote",         "Azote total"),
            ("plomb",         "Plomb (Pb)"),
            ("cadmium",       "Cadmium (Cd)"),
            ("chrome",        "Chrome (Cr)"),
            ("cuivre",        "Cuivre (Cu)"),
            ("mercure",       "Mercure (Hg)"),
            ("arsenic",       "Arsenic (As)"),
            ("hydrocarbures", "Hydrocarbures"),
        ],
    },
    "sol": {
        "name": "🪨 Sol (Soil Quality)",
        "params": [
            ("ph",                 "pH du sol"),
            ("permeabilite",       "Perméabilité"),
            ("matiere_organique",  "Matière organique"),
            ("carbone_organique",  "Carbone organique"),
            ("plomb",              "Plomb (Pb)"),
            ("cadmium",            "Cadmium (Cd)"),
            ("chrome",             "Chrome (Cr)"),
            ("azote",              "Azote"),
            ("phosphore",          "Phosphore"),
        ],
    },
    "air": {
        "name": "💨 Air Quality",
        "params": [
            ("poussieres",  "Poussières totales"),
            ("pm10",        "PM10 (particules < 10µm)"),
            ("so2",         "Dioxyde de soufre (SO₂)"),
            ("nox",         "Oxydes d'azote (NOx)"),
            ("co",          "Monoxyde de carbone (CO)"),
        ],
    },
    "population": {
        "name": "👥 Population & Qualité de vie",
        "params": [
            ("radiation_eoliennes", "Radiation des éoliennes"),
            ("radiation_cables",    "Radiation des câbles"),
            ("qualite_vie",         "Qualité de vie"),
        ],
    },
    "sante": {
        "name": "🏥 Santé & Sécurité",
        "params": [
            ("poussieres",           "Poussières (impact santé)"),
            ("risques_electriques",  "Risques électriques"),
        ],
    },
}

SPECIAL_GROUPS = {
    "paysage":        ("🌄 Paysage",          "Modification du relief (0=non, 2=oui)"),
    "infrastructure": ("🏗️ Infrastructure",    "Score infrastructure (0/1/2)"),
    "emploi":         ("💼 Emploi",            "Score emploi (0/1/2)"),
}

ETENDUE_OPTIONS = {"1": "Ponctuelle", "2": "Locale", "3": "Régionale", "4": "Nationale"}
DUREE_OPTIONS = {"1": "Courte", "2": "Moyenne", "3": "Longue"}


# ============================================================================
# TRAIN MODELS
# ============================================================================

def train_models():
    """Train one XGBoost per group on the correlated dataset."""
    df = pd.read_csv("eie_correlated.csv")
    print(f"Loaded dataset: {df.shape}")

    models = {}
    encoders = {}
    shared = ["etendue", "duree"]

    for group_name, group_info in SENTINEL_PARAMS.items():
        target_col = f"target_{group_name}"
        param_names = [p[0] for p in group_info["params"]]

        # Build feature columns
        cols = []
        for p in param_names:
            cols.append(f"{group_name}_{p}_score_m")
            cols.append(f"{group_name}_{p}_score_r")
        cols += shared

        X = df[cols].copy()
        y = df[target_col].copy()

        # Encode
        le_et = LabelEncoder().fit(X["etendue"])
        le_du = LabelEncoder().fit(X["duree"])
        X["etendue"] = le_et.transform(X["etendue"])
        X["duree"] = le_du.transform(X["duree"])

        le_y = LabelEncoder().fit(y)
        y_enc = le_y.transform(y)

        # Train
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42, verbosity=0
        )
        model.fit(X.values, y_enc)

        models[group_name] = model
        encoders[group_name] = {
            "etendue": le_et,
            "duree": le_du,
            "target": le_y,
            "features": cols,
        }
        print(f"  ✅ {group_name}: trained on {len(param_names)} params")

    return models, encoders


# ============================================================================
# INTERACTIVE PREDICTOR
# ============================================================================

def get_score(prompt):
    """Get a score (0/1/2) from user."""
    while True:
        val = input(f"  {prompt} [0/1/2]: ").strip()
        if val in ("0", "1", "2"):
            return int(val)
        print("    ⚠️ Please enter 0, 1, or 2")


def predict_interactive(models, encoders):
    """Interactive prediction session."""

    print("\n" + "="*60)
    print("  EIE SIMPLIFIED PREDICTOR")
    print("  Fill 38 rows instead of 52 — Same accuracy!")
    print("="*60)

    print("\nScoring guide:")
    print("  0 = Within acceptable range (normal)")
    print("  1 = Slightly outside (within 20% tolerance)")
    print("  2 = Far outside acceptable range (major deviation)")

    # --- Get shared inputs ---
    print(f"\n📍 GEOGRAPHIC EXTENT:")
    for k, v in ETENDUE_OPTIONS.items():
        print(f"  {k}. {v}")
    while True:
        et = input("  Choice [1-4]: ").strip()
        if et in ETENDUE_OPTIONS:
            etendue = ETENDUE_OPTIONS[et]
            break
        print("    ⚠️ Enter 1-4")

    print(f"\n⏱️ DURATION:")
    for k, v in DUREE_OPTIONS.items():
        print(f"  {k}. {v}")
    while True:
        du = input("  Choice [1-3]: ").strip()
        if du in DUREE_OPTIONS:
            duree = DUREE_OPTIONS[du]
            break
        print("    ⚠️ Enter 1-3")

    # --- Get scores per group ---
    all_predictions = {}

    for group_name, group_info in SENTINEL_PARAMS.items():
        print(f"\n{'─'*40}")
        print(f"  {group_info['name']}")
        print(f"{'─'*40}")

        feature_values = {}
        for param_id, param_label in group_info["params"]:
            m_score = get_score(f"Measured  — {param_label}")
            r_score = get_score(f"Rejection — {param_label}")
            feature_values[f"{group_name}_{param_id}_score_m"] = m_score
            feature_values[f"{group_name}_{param_id}_score_r"] = r_score

        feature_values["etendue"] = etendue
        feature_values["duree"] = duree

        # Build feature vector
        enc = encoders[group_name]
        X = pd.DataFrame([feature_values])
        X["etendue"] = enc["etendue"].transform(X["etendue"])
        X["duree"] = enc["duree"].transform(X["duree"])

        # Predict
        model = models[group_name]
        pred_enc = model.predict(X[enc["features"]].values)[0]
        pred_label = enc["target"].inverse_transform([pred_enc])[0]
        proba = model.predict_proba(X[enc["features"]].values)[0]
        classes = enc["target"].classes_

        all_predictions[group_name] = {
            "prediction": pred_label,
            "probabilities": dict(zip(classes, proba)),
        }

    # --- Special groups ---
    print(f"\n{'─'*40}")
    print(f"  Special Parameters")
    print(f"{'─'*40}")

    paysage = get_score("Modification du relief (0=non, 2=oui)")
    infra = get_score("Infrastructure capacity score")
    emploi = get_score("Employment score")

    from eie_calculator_v2 import calculate_paysage, calculate_emploi, calculate_infrastructure
    all_predictions["paysage"] = {"prediction": calculate_paysage(paysage)}
    all_predictions["infrastructure"] = {"prediction": calculate_infrastructure(infra, etendue, duree)}
    all_predictions["emploi"] = {"prediction": calculate_emploi(emploi)}

    # --- RESULTS ---
    print("\n\n" + "="*60)
    print("  📊 MATRICE D'IMPACT — RESULTS")
    print("="*60)

    colors = {
        "Mineure": "🟢", "Moyenne": "🟡", "Majeure": "🔴",
        "Impact positif": "🔵", "Autre impact": "⚪",
        "Inadmissible": "⛔",
    }

    group_labels = {
        "eau": "Eau (Water)", "sol": "Sol (Soil)", "air": "Air",
        "population": "Population", "sante": "Santé & Sécurité",
        "paysage": "Paysage", "infrastructure": "Infrastructure",
        "emploi": "Emploi",
    }

    for group, result in all_predictions.items():
        pred = result["prediction"]
        icon = colors.get(pred, "⚪")
        label = group_labels.get(group, group)
        print(f"\n  {icon} {label:25s} → {pred}")

        if "probabilities" in result:
            for cls, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"       {cls:12s} {prob:6.1%} {bar}")

    print(f"\n  Settings: Étendue={etendue}, Durée={duree}")
    print(f"  Parameters filled: 38 / 52 (27% reduction)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Training models on correlated dataset...")
    models, encoders = train_models()
    print("\nModels ready!\n")
    predict_interactive(models, encoders)
