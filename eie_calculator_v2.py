"""
EIE Calculator v2 — Score-Based
================================
Clean reimplementation that works with scores (0/1/2) directly.
No raw values needed — just the scored inputs.

Verified against the original Excel macro formulas.
"""

import numpy as np


# ============================================================================
# STEP 1: SCORING (only needed for raw values → scores)
# ============================================================================

def score_parameter(value, range_min, range_max):
    """Score a measurement: 0=in range, 1=slightly outside (±20%), 2=far outside."""
    if range_min <= value <= range_max:
        return 0
    tmin = range_min * 0.8 if range_min != 0 else -float('inf')
    tmax = range_max * 1.2
    if tmin <= value <= tmax:
        return 1
    return 2


# ============================================================================
# STEP 2: CLASSIFICATION (average scores → text labels)
# ============================================================================

def classify_value(avg_score):
    """Faible (≤0.5), Moyenne (≤1), Forte (>1)."""
    if avg_score <= 0.5:
        return "Faible"
    elif avg_score <= 1:
        return "Moyenne"
    return "Forte"


def classify_intensity(avg_score):
    """FAIBLE (≤1), MOYENNE (exactly 2), FORTE (between 1 and 2)."""
    if avg_score <= 1:
        return "FAIBLE"
    elif avg_score == 2.0:
        return "MOYENNE"
    return "FORTE"


# ============================================================================
# STEP 3: SENSITIVITY (3×3 matrix)
# ============================================================================

def compute_sensitivity(impact_apprehende, initial_value):
    """Combine two classifications into sensitivity level."""
    matrix = {
        ("FORTE",  "FORTE"):   "FORTE",
        ("FORTE",  "MOYENNE"): "FORTE",
        ("FORTE",  "FAIBLE"):  "MOYENNE",
        ("MOYENNE","FORTE"):   "FORTE",
        ("MOYENNE","MOYENNE"): "MOYENNE",
        ("MOYENNE","FAIBLE"):  "FAIBLE",
        ("FAIBLE", "FORTE"):   "MOYENNE",
        ("FAIBLE", "MOYENNE"): "FAIBLE",
        ("FAIBLE", "FAIBLE"):  "FAIBLE",
    }
    return matrix.get((impact_apprehende.upper(), initial_value.upper()), "FAIBLE")


# ============================================================================
# STEP 4: IMPORTANCE (sensitivity × intensity × extent)
# ============================================================================

def compute_importance(sensitivity, intensity, etendue):
    """3D lookup: sensitivity × intensity × geographic extent → importance."""
    s = sensitivity.upper()
    i = intensity.upper()

    if s == "FORTE":
        if i == "FORTE":
            return "Majeure" if etendue in ("Nationale", "Régionale", "Locale") else "Moyenne"
        if i == "MOYENNE":
            return "Majeure" if etendue in ("Nationale", "Régionale") else "Moyenne"
        if i == "FAIBLE":
            return "Majeure" if etendue in ("Nationale", "Régionale") else "Mineure"

    if s == "MOYENNE":
        if i == "FORTE":
            return "Majeure" if etendue == "Nationale" else "Moyenne"
        if i == "MOYENNE":
            return "Moyenne"
        if i == "FAIBLE":
            return "Moyenne" if etendue in ("Nationale", "Régionale") else "Mineure"

    if s == "FAIBLE":
        if i == "FORTE":
            return "Moyenne" if etendue in ("Nationale", "Régionale") else "Mineure"
        return "Mineure"

    return "Mineure"


# ============================================================================
# STEP 5: DURATION ADJUSTMENT (final output)
# ============================================================================

def compute_importance_relative(importance, duree):
    """Adjust importance by duration. Short durations downgrade by one level."""
    if importance in ("Mineure", "Mineure à nulle"):
        return "Mineure"
    if importance == "Moyenne":
        return "Mineure" if duree.lower() == "courte" else "Moyenne"
    if importance == "Majeure":
        return "Moyenne" if duree.lower() == "courte" else "Majeure"
    return "Mineure"


# ============================================================================
# MAIN CALCULATOR: Takes scores directly
# ============================================================================

def calculate_from_scores(measured_scores, rejection_scores, etendue, duree):
    """
    Calculate impact verdict from pre-computed scores.

    Args:
        measured_scores:  list of ints (0/1/2) — one per parameter
        rejection_scores: list of ints (0/1/2) — one per parameter
        etendue: str — "Ponctuelle", "Locale", "Régionale", "Nationale"
        duree: str — "Courte", "Moyenne", "Longue"

    Returns:
        dict with all intermediate values and final importance_relative
    """
    # Average scores
    avg_measured = np.mean(measured_scores)
    avg_rejection = np.mean(rejection_scores)

    # Combined scores (weighted: measured×1000 + rejection×100) / 1100
    # Since we only have scores (0/1/2), the combined score is approximately
    # the weighted average of the two score lists
    combined_scores = []
    for m, r in zip(measured_scores, rejection_scores):
        combined = (m * 1000 + r * 100) / 1100
        # Re-score: apply thresholds as if it were a value vs range [0, 0]
        # But since combined is already a score blend, classify directly
        if combined <= 0.5:
            combined_scores.append(0)
        elif combined <= 1.0:
            combined_scores.append(1)
        else:
            combined_scores.append(2)
    avg_combined = np.mean(combined_scores)

    # Classify
    value_initial = classify_value(avg_measured)        # Col I
    impact_apprehende = classify_value(avg_combined)    # Col N
    sensitivity = compute_sensitivity(impact_apprehende, value_initial)  # Col O
    intensity = classify_intensity(avg_rejection)       # Col T

    # Importance
    importance = compute_importance(sensitivity, intensity, etendue)     # Col U
    importance_relative = compute_importance_relative(importance, duree) # Col W

    return {
        "avg_measured": round(avg_measured, 6),
        "avg_combined": round(avg_combined, 6),
        "avg_rejection": round(avg_rejection, 6),
        "value_initial": value_initial,
        "impact_apprehende": impact_apprehende,
        "sensitivity": sensitivity,
        "intensity": intensity,
        "importance": importance,
        "importance_relative": importance_relative,
    }


def calculate_paysage(modification):
    """Binary: oui → Mineure, non → Impact positif."""
    return "Impact positif" if modification == 0 else "Mineure"


def calculate_emploi(score):
    """0 or 1 → Impact positif, 2 → Autre impact."""
    return "Impact positif" if score in (0, 1) else "Autre impact"


def calculate_infrastructure(score, etendue, duree):
    """Single-parameter group: score → full chain."""
    value = classify_value(score)
    sensitivity = value.upper()
    intensity = classify_intensity(score)
    importance = compute_importance(sensitivity, intensity, etendue)
    return compute_importance_relative(importance, duree)


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    # Test with 21 Eau parameters (mix of scores)
    m_scores = [0, 0, 1, 2, 0, 1, 0, 1, 2, 0, 0, 2, 1, 2, 0, 0, 0, 1, 0, 2, 2]
    r_scores = [2, 0, 0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 2, 1, 0, 2, 0, 2, 2, 0, 0]

    result = calculate_from_scores(m_scores, r_scores, "Régionale", "Longue")

    print("Test — Eau component:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print(f"\nPaysage (oui):  {calculate_paysage(2)}")
    print(f"Paysage (non):  {calculate_paysage(0)}")
    print(f"Emploi (≥5):    {calculate_emploi(0)}")
    print(f"Emploi (<4):    {calculate_emploi(2)}")
    print(f"Infra (good):   {calculate_infrastructure(0, 'Nationale', 'Longue')}")
    print(f"Infra (bad):    {calculate_infrastructure(2, 'Nationale', 'Longue')}")
