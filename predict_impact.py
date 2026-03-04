"""
EIE Impact Predictor — Interactive
===================================
Enter the 5 most important features and get the predicted impact classification.

Top 5 features (from XGBoost):
  1. etendue       — geographic extent
  2. duree         — duration
  3. avg_score_m   — average measured score (0-2)
  4. avg_score_r   — average rejection score (0-2)
  5. component     — environmental component
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =============================================
# STEP 1: TRAIN THE MODEL (quick, ~3 seconds)
# =============================================

print("Loading dataset and training model...")

# df = pd.read_csv("/kaggle/input/environmental-impact-assessment-eie-dataset/eie_unified.csv")
df = pd.read_csv("eie_unified.csv")

target = "importance_relative"
drop_cols = [target, "importance", "sensitivity", "impact_apprehende",
             "value_initial", "intensity_class"]

y_raw = df[target]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

# Encode categoricals
encoders = {}
for col in X.select_dtypes(include="object").columns:
    encoders[col] = LabelEncoder()
    X[col] = encoders[col].fit_transform(X[col])

feature_names = list(X.columns)

# Train
model = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                       eval_metric="mlogloss", random_state=42, verbosity=0)
model.fit(X.values, y)
print("✅ Model ready!\n")

# =============================================
# STEP 2: INTERACTIVE PREDICTION
# =============================================

# Valid options for categorical inputs
ETENDUE_OPTIONS = ["Ponctuelle", "Locale", "Régionale", "Nationale"]
DUREE_OPTIONS = ["Courte", "Moyenne", "Longue"]
COMPONENT_OPTIONS = ["Eau", "Sol", "Air", "Population", "Sante"]
PHASE_OPTIONS = ["PRE_construction", "Realisation", "Exploitation", "Demantelement"]


def ask_choice(prompt, options):
    """Ask user to pick from a list."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = int(input("Your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print(f"  ⚠️  Please enter a number between 1 and {len(options)}")


def ask_float(prompt, min_val=0, max_val=2):
    """Ask user for a number."""
    while True:
        try:
            val = float(input(f"\n{prompt} [{min_val} - {max_val}]: "))
            return val
        except ValueError:
            print("  ⚠️  Please enter a valid number")


def predict():
    """Get inputs and predict."""
    print("=" * 50)
    print("   EIE IMPACT PREDICTOR")
    print("=" * 50)

    # Get the 5 most important features
    etendue = ask_choice("1. Étendue géographique (Geographic Extent):", ETENDUE_OPTIONS)
    duree = ask_choice("2. Durée (Duration):", DUREE_OPTIONS)
    avg_m = ask_float("3. Average measured score", 0, 2)
    avg_r = ask_float("4. Average rejection score", 0, 2)
    component = ask_choice("5. Environmental Component:", COMPONENT_OPTIONS)

    # Fill remaining features with defaults (they have low importance)
    phase = "Realisation"
    max_m = min(2, int(np.ceil(avg_m)))
    max_r = min(2, int(np.ceil(avg_r)))
    pct2_m = max(0, (avg_m - 1) / 1) if avg_m > 1 else 0
    pct2_r = max(0, (avg_r - 1) / 1) if avg_r > 1 else 0
    pct0_m = max(0, 1 - avg_m) if avg_m < 1 else 0
    pct0_r = max(0, 1 - avg_r) if avg_r < 1 else 0
    n_params = {"Eau": 21, "Sol": 14, "Air": 7, "Population": 4, "Sante": 3}[component]

    # Build input row in the same column order as training
    row = {
        "phase": encoders["phase"].transform([phase])[0],
        "component": encoders["component"].transform([component])[0],
        "etendue": encoders["etendue"].transform([etendue])[0],
        "duree": encoders["duree"].transform([duree])[0],
        "avg_score_m": avg_m,
        "avg_score_r": avg_r,
        "max_score_m": max_m,
        "max_score_r": max_r,
        "pct_score2_m": round(pct2_m, 4),
        "pct_score2_r": round(pct2_r, 4),
        "pct_score0_m": round(pct0_m, 4),
        "pct_score0_r": round(pct0_r, 4),
        "n_params": n_params,
    }

    # Make sure columns are in the right order
    input_array = np.array([[row[f] for f in feature_names]], dtype=np.float32)

    # Predict
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    result = le_target.inverse_transform([prediction])[0]

    # Display result
    print(f"\n{'=' * 50}")
    print(f"   MATRICE D'IMPACT — RÉSULTAT")
    print(f"{'=' * 50}")
    print(f"\n   Inputs:")
    print(f"     Étendue:    {etendue}")
    print(f"     Durée:      {duree}")
    print(f"     Score mesuré:    {avg_m}")
    print(f"     Score rejet:     {avg_r}")
    print(f"     Composante: {component}")
    print(f"\n   ➡️  Importance relative: {result}")
    print(f"\n   Probabilities:")
    for cls, prob in zip(le_target.classes_, probabilities):
        bar = "█" * int(prob * 30)
        print(f"     {cls:10s}: {prob:.1%} {bar}")
    print(f"{'=' * 50}")


# =============================================
# MAIN LOOP
# =============================================

while True:
    predict()
    again = input("\nAnother prediction? (y/n): ").strip().lower()
    if again != "y":
        print("Bye! 👋")
        break
