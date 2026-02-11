"""
EIE (Étude d'Impact Environnemental) Calculator
================================================
Python reimplementation of ALL formulas from the Excel macro:
  "Macro Standardiser-Levaluation EIE 2026 (003) (1).xlsm"

This module replicates the full calculation pipeline for:
  - PRE Construction (4 sub-phases)
  - Réalisation
  - Exploitation
  - Démantèlement
  - Biological milieu (Flore/Faune)
  - Socio-economic components
  - Final Impact Matrix aggregation

Author: Generated from Excel formula analysis
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ============================================================================
# SCORING FUNCTIONS (Core building blocks)
# ============================================================================

def score_parameter(value: float, range_min: float, range_max: float) -> int:
    """
    Score a single parameter measurement against its acceptable range.
    
    Excel formula: =IF(AND(F>=D,F<=E), 0, IF(AND(F>=D*0.8,F<=E*1.2), 1, 2))
    
    Returns:
        0 = within acceptable range
        1 = slightly outside (within 20% tolerance)
        2 = largely outside acceptable range
    """
    if range_min is None and range_max is None:
        return 0  # No range defined, assume conforming
    
    # Handle cases where only MAX is defined (MIN is None or 0)
    effective_min = range_min if range_min is not None else 0
    effective_max = range_max if range_max is not None else float('inf')
    
    # Check: within acceptable range
    if effective_min <= value <= effective_max:
        return 0
    
    # Check: within 20% tolerance (±20% of range boundaries)
    tolerance_min = effective_min * 0.8 if effective_min != 0 else -float('inf')
    tolerance_max = effective_max * 1.2 if effective_max != float('inf') else float('inf')
    
    if tolerance_min <= value <= tolerance_max:
        return 1
    
    return 2


def classify_value(avg_score: float) -> str:
    """
    Classify average score into impact level.
    
    Excel formula: =IF(avg<=0.5, "Faible", IF(avg<=1, "Moyenne", "Forte"))
    """
    if avg_score <= 0.5:
        return "Faible"
    elif avg_score <= 1:
        return "Moyenne"
    else:
        return "Forte"


def compute_sensitivity(impact_apprehende: str, initial_value: str) -> str:
    """
    Combine initial measurement state with rejection state to get sensitivity.
    
    This is a 3x3 matrix combining two classifications.
    
    Excel formula: nested IF with AND conditions for 9 combinations.
    """
    # Normalize to uppercase for comparison
    ia = impact_apprehende.upper()
    iv = initial_value.upper()
    
    # Sensitivity matrix
    if ia == "FORTE":
        if iv == "FORTE":
            return "FORTE"
        elif iv == "MOYENNE":
            return "FORTE"
        else:  # FAIBLE
            return "MOYENNE"
    elif ia == "MOYENNE":
        if iv == "FORTE":
            return "FORTE"
        elif iv == "MOYENNE":
            return "MOYENNE"
        else:  # FAIBLE
            return "FAIBLE"
    else:  # FAIBLE
        if iv == "FORTE":
            return "MOYENNE"
        elif iv == "MOYENNE":
            return "FAIBLE"
        else:  # FAIBLE
            return "FAIBLE"


def score_extent(etendue: str) -> float:
    """
    Convert extent category to numerical score.
    
    Excel formula: =IF(P="Ponctuelle",0, IF(P="Locale",0.5, IF(P="Régionale",1.25, IF(P="Nationale",2, ""))))
    """
    mapping = {
        "Ponctuelle": 0.0,
        "Locale": 0.5,
        "Régionale": 1.25,
        "Nationale": 2.0,
        "Internationale": 2.0,
    }
    return mapping.get(etendue, 0.0)


def classify_intensity(avg_score: float) -> str:
    """
    Classify intensity from average rejection score.
    
    Excel formula: =IF(S<=1,"FAIBLE",IF(S=2,"MOYENNE","FORTE"))
    
    NOTE: In Excel, IF(S=2,"MOYENNE","FORTE") means:
      - Exactly 2.0 → "MOYENNE"
      - Everything else (>1 and ≠2) → "FORTE"
    So values between 1 and 2 are "FORTE", not "MOYENNE".
    """
    if avg_score <= 1:
        return "FAIBLE"
    elif avg_score == 2.0:
        return "MOYENNE"
    else:
        return "FORTE"


def compute_importance(sensitivity: str, intensity: str, etendue: str) -> str:
    """
    Compute importance level from sensitivity × intensity × extent.
    
    Excel formula: Complex nested IF with 4 sensitivity levels.
    """
    sens = sensitivity.upper() if sensitivity else ""
    intens = intensity.upper() if intensity else ""
    ext = etendue if etendue else ""
    
    if sens == "ABSOLUE":
        return "Inadmissible"
    
    if sens == "FORTE":
        if intens == "FORTE":
            if ext in ("Nationale", "Régionale", "Locale"):
                return "Majeure"
            return "Moyenne"
        elif intens == "MOYENNE":
            if ext in ("Nationale", "Régionale"):
                return "Majeure"
            return "Moyenne"
        elif intens == "FAIBLE":
            if ext in ("Nationale", "Régionale"):
                return "Majeure"
            return "Mineure"
        return ""
    
    if sens == "MOYENNE":
        if intens == "FORTE":
            if ext == "Nationale":
                return "Majeure"
            if ext == "Régionale":
                return "Moyenne"
            return "Moyenne"
        elif intens == "MOYENNE":
            return "Moyenne"
        elif intens == "FAIBLE":
            if ext in ("Nationale", "Régionale"):
                return "Moyenne"
            return "Mineure"
        return ""
    
    if sens == "FAIBLE":
        if intens == "FORTE":
            if ext in ("Nationale", "Régionale"):
                return "Moyenne"
            return "Mineure"
        elif intens == "MOYENNE":
            return "Mineure"
        else:
            return "Mineure"
    
    if sens in ("TRÈS FAIBLE", "TRES FAIBLE"):
        if intens == "FORTE":
            return "Mineure"
        return "Mineure à nulle"
    
    return "Mineure"


def compute_importance_relative(importance: str, duree: str) -> str:
    """
    Adjust importance by duration to get final relative importance.
    
    Excel formula:
    =IF(OR(U="Mineure",U="Mineure à nulle"),"Mineure",
     IF(U="Moyenne", IF(V="Courte","Mineure","Moyenne"),
     IF(U="Majeure", IF(V="Courte","Moyenne","Majeure"), "")))
    """
    imp = importance if importance else ""
    dur = duree if duree else ""
    
    if imp in ("Mineure", "Mineure à nulle"):
        return "Mineure"
    elif imp == "Moyenne":
        if dur.lower() == "courte":
            return "Mineure"
        return "Moyenne"
    elif imp == "Majeure":
        if dur.lower() == "courte":
            return "Moyenne"
        return "Majeure"
    elif imp == "Inadmissible":
        return "Inadmissible"
    return ""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ParameterInput:
    """A single environmental parameter measurement."""
    name: str
    unit: str
    range_min: Optional[float]
    range_max: Optional[float]
    measured_value: float       # Column F (RED input)
    rejection_value: float      # Column J (RED input)


@dataclass
class ComponentGroup:
    """A group of parameters that share an impact assessment (e.g., Eau, Sol, Air)."""
    name: str                                # e.g., "Milieu PHYSIQUE/EAU"
    parameters: List[ParameterInput]         # All parameters in this group
    etendue: str                             # Column P (RED input): "Locale", "Régionale", etc.
    duree: str                               # Column V (RED input): "Courte", "Longue", etc.


@dataclass 
class BiologicalInput:
    """Input for biological milieu (different scoring system)."""
    species_type: str           # e.g., "mammifères", "amphibiens"
    presence: str               # "Aucune présence", "Présence potentielle", etc.
    protection_status: str      # "Aucune protection", "niveau local ou régional", etc.
    species_status: str         # "Aucune espèce protégée", "Espèce vulnérable", etc.


# ============================================================================
# STANDARD COMPONENT CALCULATOR (Eau, Sol, Air, Ambiance Sonore)
# Shared by Réalisation, Exploitation, Démantèlement
# ============================================================================

def calculate_standard_component(group: ComponentGroup) -> dict:
    """
    Calculate impact for a standard physical/chemical component group.
    
    This replicates the formula chain for groups like Eau, Sol, Air, etc.
    in the Réalisation/Exploitation/Démantèlement sheets.
    
    Returns all intermediate and final values.
    """
    # --- Step 1: Score each parameter (Column G) ---
    initial_scores = []
    combined_scores = []
    rejection_scores = []
    
    for p in group.parameters:
        # Score for initial measured value (Col G)
        g_score = score_parameter(p.measured_value, p.range_min, p.range_max)
        initial_scores.append(g_score)
        
        # Combined value = weighted average (Col K) via VBA macro
        # Cf = (Cm * 1000 + Cr * 100) / (1000 + 100)
        combined_value = (p.measured_value * 1000 + p.rejection_value * 100) / 1100
        
        # Score for combined value (Col L)
        l_score = score_parameter(combined_value, p.range_min, p.range_max)
        combined_scores.append(l_score)
        
        # Score for rejection value directly (Col R)
        r_score = score_parameter(p.rejection_value, p.range_min, p.range_max)
        rejection_scores.append(r_score)
    
    # --- Step 2: Average scores (Columns H, M, S) ---
    avg_initial = np.mean(initial_scores) if initial_scores else 0
    avg_combined = np.mean(combined_scores) if combined_scores else 0
    avg_rejection = np.mean(rejection_scores) if rejection_scores else 0
    
    # --- Step 3: Classify (Columns I, N, T) ---
    value_initial = classify_value(avg_initial)     # Col I
    value_combined = classify_value(avg_combined)    # Col N (Impact Appréhendé)
    intensity_class = classify_intensity(avg_rejection)  # Col T
    
    # --- Step 4: Sensitivity (Column O) ---
    sensitivity = compute_sensitivity(value_combined, value_initial)
    
    # --- Step 5: Extent score (Column Q) ---
    extent_score = score_extent(group.etendue)
    
    # --- Step 6: Importance (Column U) ---
    importance = compute_importance(sensitivity, intensity_class, group.etendue)
    
    # --- Step 7: Final importance relative (Column W) ---
    importance_relative = compute_importance_relative(importance, group.duree)
    
    return {
        "group_name": group.name,
        "initial_scores": initial_scores,
        "combined_scores": combined_scores,
        "rejection_scores": rejection_scores,
        "avg_initial": round(avg_initial, 6),
        "avg_combined": round(avg_combined, 6),
        "avg_rejection": round(avg_rejection, 6),
        "value_initial": value_initial,         # Col I
        "impact_apprehende": value_combined,     # Col N
        "sensitivity": sensitivity,              # Col O
        "extent_score": extent_score,            # Col Q
        "intensity_class": intensity_class,      # Col T
        "importance": importance,                # Col U
        "duree": group.duree,                    # Col V
        "importance_relative": importance_relative,  # Col W (FINAL OUTPUT)
    }


# ============================================================================
# PRE CONSTRUCTION CALCULATOR (4 sub-phases)
# ============================================================================

@dataclass
class PreConstructionGroup:
    """A component group for PRE construction with 4 sub-phase values."""
    name: str
    parameters: List[dict]  # Each dict: {name, unit, range_min, range_max, 
                            #              value_f, value_j, value_m, value_p, value_s}
    etendue: str            # Col Y (RED)
    duree: str              # Col AC (RED)


def calculate_pre_construction_component(group: PreConstructionGroup) -> dict:
    """
    Calculate impact for PRE construction (4 sub-phases).
    
    PRE construction has 4 measured value columns (sub-phases):
    F (Prospection), J (Signalisation), M (Installation), P/S (Transport)
    
    Each gets scored, averaged, then the 4 averages are averaged for final score.
    """
    scores_f = []  # Sub-phase 1
    scores_j = []  # Sub-phase 2
    scores_m = []  # Sub-phase 3
    scores_s = []  # Sub-phase 4
    
    for p in group.parameters:
        rmin = p.get('range_min')
        rmax = p.get('range_max')
        
        # Score each sub-phase value
        if p.get('value_f') is not None:
            scores_f.append(score_parameter(p['value_f'], rmin, rmax))
        if p.get('value_j') is not None:
            scores_j.append(score_parameter(p['value_j'], rmin, rmax))
        if p.get('value_m') is not None:
            scores_m.append(score_parameter(p['value_m'], rmin, rmax))
        if p.get('value_s') is not None:
            scores_s.append(score_parameter(p['value_s'], rmin, rmax))
    
    # Average per sub-phase (Cols L, O, R, U)
    avg_f = np.mean(scores_f) if scores_f else 0
    avg_j = np.mean(scores_j) if scores_j else 0
    avg_m = np.mean(scores_m) if scores_m else 0
    avg_s = np.mean(scores_s) if scores_s else 0
    
    # Initial value classification from first sub-phase (Col I)
    avg_initial = np.mean(scores_f) if scores_f else 0
    value_initial = classify_value(avg_initial)
    
    # Score FINAL = average of 4 sub-phase averages (Col V)
    sub_phase_avgs = [avg_f, avg_j, avg_m, avg_s]
    score_final = np.mean(sub_phase_avgs)
    
    # Impact Appréhendé (Col W)
    impact_apprehende = classify_value(score_final)
    
    # Sensitivity (Col X)
    sensitivity = compute_sensitivity(impact_apprehende, value_initial)
    
    # Extent score (Col Z)
    extent_score = score_extent(group.etendue)
    
    # Intensity from score final (Col AA)
    intensity = classify_intensity(score_final)
    
    # Importance (Col AB)
    importance = compute_importance(sensitivity, intensity, group.etendue)
    
    # Final importance relative (Col AD)
    importance_relative = compute_importance_relative(importance, group.duree)
    
    return {
        "group_name": group.name,
        "scores_subphase_1": scores_f,
        "scores_subphase_2": scores_j,
        "scores_subphase_3": scores_m,
        "scores_subphase_4": scores_s,
        "avg_subphase_1": round(avg_f, 6),
        "avg_subphase_2": round(avg_j, 6),
        "avg_subphase_3": round(avg_m, 6),
        "avg_subphase_4": round(avg_s, 6),
        "score_final": round(score_final, 6),
        "value_initial": value_initial,
        "impact_apprehende": impact_apprehende,
        "sensitivity": sensitivity,
        "extent_score": extent_score,
        "intensity": intensity,
        "importance": importance,
        "duree": group.duree,
        "importance_relative": importance_relative,
    }


# ============================================================================
# BIOLOGICAL MILIEU CALCULATOR (Flore/Faune)
# ============================================================================

def score_presence(presence: str) -> int:
    """
    Score biological presence level.
    
    Excel: =IF(G="Aucune présence",0, IF(G="Présence potentielle",1,
            IF(G="Présence occasionnelle",2, IF(G="Présence régulière",3,
            IF(G="Présence permanente",4, 0)))))
    """
    mapping = {
        "Aucune présence": 0,
        "Présence potentielle": 1,
        "Présence occasionnelle": 2,
        "Présence régulière": 3,
        "Présence permanente": 4,
    }
    return mapping.get(presence, 0)


def score_protection(protection: str) -> int:
    """
    Score protection status.
    
    Excel: =IF(I="Aucune protection",0, IF(I="niveau local ou régional",1,
            IF(I="niveau national",2, IF(I="Liste rouge UICN",3, 
            IF(I="Liste rouge critique",4, 0)))))
    """
    mapping = {
        "Aucune protection": 0,
        "niveau local ou régional": 1,
        "niveau national (loi marocaine par ex.)": 2,
        "Espèce inscrite sur la Liste rouge UICN": 3,
        "Espèce inscrite Liste rouge - en danger critique": 4,
    }
    return mapping.get(protection, 0)


def score_species(species_status: str) -> int:
    """
    Score species vulnerability.
    
    Excel: =IF(K="Aucune espèce protégée",0, IF(K="Espèce peu menacée",1,
            IF(K="Espèce vulnérable",2, IF(K="Espèce en danger / endémique",3,
            IF(K="Espèce en danger critique",4, IF(K="Espèce protégée + reproduction",5, 0))))))
    """
    mapping = {
        "Aucune espèce protégée": 0,
        "Espèce peu menacée ou rare": 1,
        "Espèce vulnérable": 2,
        "Espèce en danger / endémique": 3,
        "Espèce en danger critique": 4,
        "Espèce protégée + reproduction": 5,
    }
    return mapping.get(species_status, 0)


def calculate_biological_component(species_list: List[BiologicalInput], 
                                    component: str = "Flore") -> dict:
    """
    Calculate biological impact for a group of species.
    
    Biological milieu uses a completely different scoring system:
    - Score  = Presence + Protection + Species scores (sum)
    - Classification based on total (0-6: Faible/Mineure, 7-12: Moyenne, etc.)
    - Average across species for final importance
    """
    total_scores = []
    classifications = []
    
    for sp in species_list:
        h_score = score_presence(sp.presence)
        j_score = score_protection(sp.protection_status)
        l_score = score_species(sp.species_status)
        
        # M = Sum of all 3 scores
        total = h_score + j_score + l_score
        total_scores.append(total)
        
        # Classification (Row 61 uses Mineure/Moyenne/Majeure, Row 63 uses Faible/Moyenne/Grave)
        if component == "Flore":
            if total <= 6:
                classifications.append("Mineure")
            elif total <= 12:
                classifications.append("Moyenne")
            elif total <= 18:
                classifications.append("Majeure")
            else:
                classifications.append("Très grave")
        else:  # Faune
            if total <= 6:
                classifications.append("Faible")
            elif total <= 12:
                classifications.append("Moyenne")
            elif total <= 18:
                classifications.append("Grave")
            else:
                classifications.append("Très grave")
    
    # Average total score across species
    avg_score = np.mean(total_scores) if total_scores else 0
    
    # Final importance for fauna (from average)
    if avg_score <= 3:
        final_importance = "Mineure"
    elif avg_score <= 6:
        final_importance = "Moyenne"
    else:
        final_importance = "Majeure"
    
    return {
        "component": component,
        "species_scores": total_scores,
        "species_classifications": classifications,
        "avg_score": round(avg_score, 3),
        "final_importance": final_importance,
    }


# ============================================================================
# PAYSAGE (LANDSCAPE) CALCULATOR
# ============================================================================

def calculate_paysage(modification_relief: str) -> dict:
    """
    Calculate landscape impact (binary: oui/non).
    
    Excel: =IF(F39=D39, 0, IF(F39=E39, 2))
    Final: =IF(G39=0, "Impact positif", "Mineure")
    """
    if modification_relief.lower() == "non":
        score = 0
        importance_relative = "Impact positif"
    else:
        score = 2
        importance_relative = "Mineure"
    
    return {
        "component": "Paysage",
        "modification_relief": modification_relief,
        "score": score,
        "importance_relative": importance_relative,
    }


# ============================================================================
# SOCIO-ECONOMIC CALCULATORS (Infrastructure, Employment)
# ============================================================================

def calculate_infrastructure(capacity_pct: float, etendue: str, duree: str) -> dict:
    """
    Infrastructure & equipment impact.
    
    Excel Row 57 special formula: =IF(J57>=90, 0, IF(AND(J57>=85, J57<90), 1, 2))
    """
    # Score based on capacity maintenance percentage
    if capacity_pct >= 90:
        score = 0
    elif capacity_pct >= 85:
        score = 1
    else:
        score = 2
    
    value_class = classify_value(score)
    sensitivity = value_class  # O57 = N57
    intensity = classify_intensity(score)
    importance = compute_importance(sensitivity, intensity, etendue)
    importance_relative = compute_importance_relative(importance, duree)
    
    return {
        "component": "Infrastructure et équipement",
        "capacity_pct": capacity_pct,
        "score": score,
        "value_class": value_class,
        "sensitivity": sensitivity,
        "intensity": intensity,
        "importance": importance,
        "importance_relative": importance_relative,
    }


def calculate_employment(jobs_created: float, etendue: str, duree: str) -> dict:
    """
    Socio-economic activity / Employment impact.
    
    Excel Row 58: =IF(J58>=5, 0, IF(AND(J58>=4, J58<=4), 1, 2))
    Final: =IF(OR(score=0, score=1), "Impact positif", "Autre impact")
    """
    # Score based on jobs created
    if jobs_created >= 5:
        score = 0
    elif jobs_created >= 4:
        score = 1
    else:
        score = 2
    
    # Special final classification for employment
    if score in (0, 1):
        importance_relative = "Impact positif"
    else:
        importance_relative = "Autre impact"
    
    return {
        "component": "Activité socio-économique /Emploi",
        "jobs_created": jobs_created,
        "score": score,
        "importance_relative": importance_relative,
    }


# ============================================================================
# FULL PHASE CALCULATOR
# ============================================================================

@dataclass
class PhaseResult:
    """Complete results for one project phase."""
    phase_name: str
    eau: dict
    sol: dict
    ambiance_sonore: dict
    air: dict
    paysage: dict
    population: dict
    sante: dict
    infrastructure: dict
    emploi: dict
    flore: dict
    faune: dict


def calculate_phase(phase_name: str,
                    eau_group: ComponentGroup,
                    sol_group: ComponentGroup,
                    ambiance_sonore_group: ComponentGroup,
                    air_group: ComponentGroup,
                    paysage_modification: str,
                    population_group: ComponentGroup,
                    sante_group: ComponentGroup,
                    infrastructure_capacity: float,
                    infrastructure_etendue: str,
                    infrastructure_duree: str,
                    employment_jobs: float,
                    employment_etendue: str,
                    employment_duree: str,
                    flore_species: List[BiologicalInput],
                    faune_species: List[BiologicalInput]) -> PhaseResult:
    """
    Calculate the full impact assessment for one project phase.
    
    This brings together all component calculators for a single phase.
    """
    return PhaseResult(
        phase_name=phase_name,
        eau=calculate_standard_component(eau_group),
        sol=calculate_standard_component(sol_group),
        ambiance_sonore=calculate_standard_component(ambiance_sonore_group),
        air=calculate_standard_component(air_group),
        paysage=calculate_paysage(paysage_modification),
        population=calculate_standard_component(population_group),
        sante=calculate_standard_component(sante_group),
        infrastructure=calculate_infrastructure(
            infrastructure_capacity, infrastructure_etendue, infrastructure_duree),
        emploi=calculate_employment(
            employment_jobs, employment_etendue, employment_duree),
        flore=calculate_biological_component(flore_species, "Flore"),
        faune=calculate_biological_component(faune_species, "Faune"),
    )


# ============================================================================
# IMPACT MATRIX BUILDER
# ============================================================================

def importance_to_score(imp: str) -> float:
    """Convert importance relative to numerical score for matrix."""
    mapping = {
        "Mineure": 0.5,
        "Moyenne": 1.0,
        "Majeure": 1.5,
        "Inadmissible": 2.0,
        "Impact positif": 0.0,
        "Autre impact": 0.0,
        "Mineure à nulle": 0.25,
        "": 0.0,
    }
    return mapping.get(imp, 0.0)


def build_impact_matrix(pre_result, realisation_result: PhaseResult,
                        exploitation_result: PhaseResult,
                        demantelement_result: PhaseResult) -> dict:
    """
    Build the final impact matrix aggregating all phases.
    
    This replicates the "matrice d'impacts" sheet, combining
    importance_relative from each phase for each environmental component.
    """
    components = [
        "Sol", "Air", "Eau", "Paysage", "Ambiance sonore",
        "Flore", "Faune", "Milieu marin",
        "Population et qualité de vie", "Santé & Sécurité",
        "Activité socio-économique", "Infrastructures"
    ]
    
    matrix = {}
    
    def get_imp(result, component_key):
        """Get importance_relative from a phase result for a component."""
        if hasattr(result, component_key):
            r = getattr(result, component_key)
            return r.get("importance_relative", "") if isinstance(r, dict) else ""
        return ""
    
    for comp in components:
        key_map = {
            "Sol": "sol", "Air": "air", "Eau": "eau",
            "Paysage": "paysage", "Ambiance sonore": "ambiance_sonore",
            "Flore": "flore", "Faune": "faune",
            "Population et qualité de vie": "population",
            "Santé & Sécurité": "sante",
            "Activité socio-économique": "emploi",
            "Infrastructures": "infrastructure",
        }
        key = key_map.get(comp, "")
        
        if comp == "Milieu marin":
            # Marine milieu - typically empty or same as Exploitation refs
            pre_imp = ""
            real_imp = ""
            expl_imp = ""
            dema_imp = ""
        elif comp == "Flore" or comp == "Faune":
            pre_imp = ""  # Bio not in PRE
            real_imp = get_imp(realisation_result, key)
            if isinstance(getattr(realisation_result, key, {}), dict):
                real_imp = getattr(realisation_result, key).get("final_importance", "")
            expl_imp = get_imp(exploitation_result, key)
            if isinstance(getattr(exploitation_result, key, {}), dict):
                expl_imp = getattr(exploitation_result, key).get("final_importance", "")
            dema_imp = get_imp(demantelement_result, key)
            if isinstance(getattr(demantelement_result, key, {}), dict):
                dema_imp = getattr(demantelement_result, key).get("final_importance", "")
        else:
            pre_imp = pre_result.get(key, {}).get("importance_relative", "") if isinstance(pre_result.get(key, {}), dict) else ""
            real_imp = get_imp(realisation_result, key)
            expl_imp = get_imp(exploitation_result, key)
            dema_imp = get_imp(demantelement_result, key)
        
        # Calculate score
        all_imps = [pre_imp, real_imp, expl_imp, dema_imp]
        scores = [importance_to_score(i) for i in all_imps if i]
        total_score = sum(scores) / len(scores) if scores else 0
        
        matrix[comp] = {
            "pre_construction": pre_imp,
            "realisation": real_imp,
            "exploitation": expl_imp,
            "demantelement": dema_imp,
            "score": round(total_score, 3),
            "pct": round(total_score / 2, 3) if total_score else 0,
        }
    
    return matrix


# ============================================================================
# VERIFICATION: Test with known Excel values
# ============================================================================

def verify_against_excel():
    """
    Test the calculator against known values from the Excel file.
    Demonstrates that calculations match.
    """
    print("=" * 60)
    print("VERIFICATION: Testing against known Excel values")
    print("=" * 60)
    
    # Test 1: score_parameter
    print("\n--- Test 1: score_parameter ---")
    assert score_parameter(19, 10, 30) == 0, "19 in [10,30] should be 0"
    assert score_parameter(2001, 0, 2000) == 1, "2001 with max=2000 should be 1 (within 20%)"
    assert score_parameter(2900, 0, 2000) == 2, "2900 with max=2000 should be 2"
    assert score_parameter(7, 6.5, 8.5) == 0, "7 in [6.5,8.5] should be 0"
    print("  ✅ All score_parameter tests passed")
    
    # Test 2: classify_value
    print("\n--- Test 2: classify_value ---")
    assert classify_value(0.19) == "Faible"
    assert classify_value(0.5) == "Faible"
    assert classify_value(0.7) == "Moyenne"
    assert classify_value(1.5) == "Forte"
    print("  ✅ All classify_value tests passed")
    
    # Test 3: compute_sensitivity
    print("\n--- Test 3: compute_sensitivity ---")
    assert compute_sensitivity("Faible", "Faible") == "FAIBLE"
    assert compute_sensitivity("Forte", "Forte") == "FORTE"
    assert compute_sensitivity("Forte", "Faible") == "MOYENNE"
    print("  ✅ All compute_sensitivity tests passed")
    
    # Test 4: score_extent
    print("\n--- Test 4: score_extent ---")
    assert score_extent("Locale") == 0.5
    assert score_extent("Régionale") == 1.25
    assert score_extent("Nationale") == 2.0
    print("  ✅ All score_extent tests passed")
    
    # Test 5: compute_importance
    print("\n--- Test 5: compute_importance ---")
    assert compute_importance("FAIBLE", "FAIBLE", "Locale") == "Mineure"
    assert compute_importance("FORTE", "FORTE", "Régionale") == "Majeure"
    print("  ✅ All compute_importance tests passed")
    
    # Test 6: compute_importance_relative
    print("\n--- Test 6: compute_importance_relative ---")
    assert compute_importance_relative("Mineure", "Longue") == "Mineure"
    assert compute_importance_relative("Moyenne", "Courte") == "Mineure"
    assert compute_importance_relative("Moyenne", "Longue") == "Moyenne"
    assert compute_importance_relative("Majeure", "Longue") == "Majeure"
    assert compute_importance_relative("Majeure", "Courte") == "Moyenne"
    print("  ✅ All compute_importance_relative tests passed")
    
    # Test 7: Biological scoring
    print("\n--- Test 7: Biological scoring ---")
    assert score_presence("Aucune présence") == 0
    assert score_presence("Présence occasionnelle") == 2
    assert score_protection("niveau local ou régional") == 1
    assert score_species("Espèce vulnérable") == 2
    print("  ✅ All biological scoring tests passed")
    
    # Test 8: Full component calculation with known Excel data
    print("\n--- Test 8: Full Réalisation Eau Component ---")
    # From Excel: Démantalement Row 2, known result = "Moyenne"
    # With measured value=8, range [10,30], rejection=3
    demo_params = [
        ParameterInput("temperature", "°C", 10, 30, 8, 3),
        ParameterInput("Ph", "—", 6.5, 8.5, 7, 9),
    ]
    demo_group = ComponentGroup("Demo", demo_params, "Régionale", "Longue")
    result = calculate_standard_component(demo_group)
    print(f"  avg_initial={result['avg_initial']}, value_initial={result['value_initial']}")
    print(f"  importance={result['importance']}, importance_relative={result['importance_relative']}")
    
    # Test 9: Paysage
    print("\n--- Test 9: Paysage ---")
    p = calculate_paysage("oui")
    assert p["score"] == 2
    assert p["importance_relative"] == "Mineure"
    p2 = calculate_paysage("non")
    assert p2["score"] == 0
    assert p2["importance_relative"] == "Impact positif"
    print("  ✅ All paysage tests passed")
    
    # Test 10: Employment
    print("\n--- Test 10: Employment ---")
    e = calculate_employment(8, "Nationale", "Longue")
    assert e["importance_relative"] == "Impact positif"
    e2 = calculate_employment(2, "Nationale", "Longue")
    assert e2["importance_relative"] == "Autre impact"
    print("  ✅ All employment tests passed")
    
    print("\n" + "=" * 60)
    print("✅ ALL VERIFICATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    verify_against_excel()
