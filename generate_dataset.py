"""
EIE Dataset Generator
=====================
Generates synthetic training data for the Environmental Impact Assessment model.

Uses the verified eie_calculator.py to compute all outputs from randomized inputs.
Produces a single CSV file ready for Kaggle with:
  - All RED column inputs (randomized with realistic ranges)
  - All intermediate computed values
  - Final impact matrix outputs as targets

Usage:
    python generate_dataset.py --rows 100000 --output dataset.csv
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

# Import our verified calculator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eie_calculator import (
    ParameterInput, ComponentGroup, BiologicalInput,
    calculate_standard_component, calculate_paysage,
    calculate_biological_component, calculate_infrastructure,
    calculate_employment, score_parameter, score_presence,
    score_protection, score_species
)

np.random.seed(42)

# ============================================================================
# PARAMETER DEFINITIONS (extracted from Excel)
# ============================================================================

# Each parameter: (name, range_min, range_max)
# For generation, we create values that span all 3 scoring zones:
#   Zone 0: within range
#   Zone 1: ±20% outside range
#   Zone 2: far outside range

PARAMS_EAU = [
    ("temperature", 10, 30),
    ("Ph", 6.5, 8.5),
    ("Turbidite", 0, 5),
    ("Conductivite", 0, 2000),
    ("DBO5", 0, 5),
    ("DCO", 0, 25),
    ("Oxygene_dissous", 0, 5),
    ("Nitrates", 0, 50),
    ("Nitrites", 0, 0.1),
    ("Ammoniac", 0, 0.5),
    ("Phosphore_total", 0, 0.053),
    ("Azote_total", 0, 0.1),
    ("Plomb_Pb", 0, 0.01),
    ("Cadmium_Cd", 0, 0.003),
    ("Chrome_Cr", 0, 0.05),
    ("Cuivre_Cu", 0, 2),
    ("Zinc_Zn", 0, 3),
    ("Nickel_Ni", 0, 0.02),
    ("Mercure_Hg", 0, 0.001),
    ("Arsenic_As", 0, 0.01),
    ("Hydrocarbures", 0, 0.05),
]

PARAMS_SOL = [
    ("pH_sol", 5.5, 7.5),
    ("Permeabilite", 1e-6, 0.001),
    ("Matiere_organique", 2, 10),
    ("Carbone_organique", 1, 6),  # MAX inferred
    ("Plomb_Pb_sol", 50, 300),
    ("Cadmium_Cd_sol", 1, 3),
    ("Mercure_Hg_sol", 0.5, 2),
    ("Arsenic_As_sol", 10, 50),
    ("Chrome_Cr_sol", 50, 200),
    ("Cuivre_Cu_sol", 50, 140),
    ("Zinc_Zn_sol", 150, 300),
    ("Nickel_Ni_sol", 30, 75),
    ("Azote_total_sol", 1000, 5000),
    ("Phosphore_total_sol", 200, 3000),
]

PARAMS_AIR = [
    ("Poussieres_totales", 0, 0.23),
    ("PM10", 0, 50),
    ("PM2_5", 0, 25),
    ("SO2", 0, 120),
    ("NOx", 0, 200),
    ("CO", 0, 10000),
    ("O3_ozone", 0, 120),
]

PARAMS_POPULATION = [
    ("Radiation_eoliennes", 0, 5),
    ("Radiation_cables", 0, 5),
    ("Radiation_onduleurs", 0, 61),
    ("Qualite_vie", 0, 5),
]

PARAMS_SANTE = [
    ("Poussieres_sante", 0, 0.15),
    ("Risques_electriques", 0, 2),
    ("Sante_securite", 0, 2),
]

# Categorical options
ETENDUE_OPTIONS = ["Ponctuelle", "Locale", "Régionale", "Nationale"]
DUREE_OPTIONS = ["Courte", "Moyenne", "Longue"]

# Biological categorical options
PRESENCE_OPTIONS = [
    "Aucune présence", "Présence potentielle", "Présence occasionnelle",
    "Présence régulière", "Présence permanente"
]
PROTECTION_OPTIONS = [
    "Aucune protection", "niveau local ou régional",
    "niveau national (loi marocaine par ex.)",
    "Espèce inscrite sur la Liste rouge UICN",
    "Espèce inscrite Liste rouge - en danger critique"
]
SPECIES_OPTIONS = [
    "Aucune espèce protégée", "Espèce peu menacée ou rare",
    "Espèce vulnérable", "Espèce en danger / endémique",
    "Espèce en danger critique", "Espèce protégée + reproduction"
]


# ============================================================================
# VALUE GENERATION FUNCTIONS
# ============================================================================

def generate_value_for_param(range_min, range_max, n_samples, noise_std=0.05):
    """
    Generate realistic values spanning all 3 scoring zones.
    
    Strategy:
    - 40% in range (score 0)
    - 30% slightly outside (score 1)
    - 30% far outside (score 2)
    
    This ensures balanced representation of all score classes.
    """
    values = np.zeros(n_samples)
    range_width = range_max - range_min if (range_max - range_min) > 0 else 1
    
    n_zone0 = int(n_samples * 0.4)
    n_zone1 = int(n_samples * 0.3)
    n_zone2 = n_samples - n_zone0 - n_zone1
    
    # Zone 0: within acceptable range
    if range_width > 0:
        values[:n_zone0] = np.random.uniform(range_min, range_max, n_zone0)
    else:
        values[:n_zone0] = range_min
    
    # Zone 1: slightly outside (within 20% tolerance)
    # Either below range_min*0.8 to range_min, or range_max to range_max*1.2
    for i in range(n_zone0, n_zone0 + n_zone1):
        if np.random.random() < 0.5 and range_min > 0:
            # Below range
            values[i] = np.random.uniform(range_min * 0.8, range_min)
        else:
            # Above range
            values[i] = np.random.uniform(range_max, range_max * 1.2)
    
    # Zone 2: far outside range
    for i in range(n_zone0 + n_zone1, n_samples):
        if np.random.random() < 0.5 and range_min > 0:
            # Far below
            values[i] = np.random.uniform(0, range_min * 0.8)
        else:
            # Far above
            values[i] = np.random.uniform(range_max * 1.2, range_max * 2.5)
    
    # Add small noise to make data more realistic
    noise = np.random.normal(0, noise_std * range_width, n_samples)
    values = values + noise
    
    # Ensure non-negative where appropriate
    if range_min >= 0:
        values = np.maximum(values, 0)
    
    # Shuffle all values
    np.random.shuffle(values)
    
    return np.round(values, 6)


def generate_categorical(options, n_samples, weights=None):
    """Generate random categorical values from options."""
    if weights is None:
        weights = np.ones(len(options)) / len(options)
    return np.random.choice(options, n_samples, p=weights)


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(n_rows: int, phases: list = None) -> pd.DataFrame:
    """
    Generate a complete dataset with randomized inputs and computed outputs.
    
    Each row represents one complete assessment for one environmental component group
    in one project phase.
    
    Args:
        n_rows: Number of rows to generate
        phases: List of phases to include (default: all 4)
    
    Returns:
        DataFrame with all inputs, intermediates, and outputs
    """
    if phases is None:
        phases = ["PRE_construction", "Realisation", "Exploitation", "Demantelement"]
    
    all_component_groups = {
        "Eau": PARAMS_EAU,
        "Sol": PARAMS_SOL,
        "Air": PARAMS_AIR,
        "Population": PARAMS_POPULATION,
        "Sante": PARAMS_SANTE,
    }
    
    rows = []
    
    # Calculate how many rows per phase/component combination
    n_combinations = len(phases) * len(all_component_groups)
    rows_per_combo = max(1, n_rows // n_combinations)
    
    print(f"Generating {rows_per_combo} rows per combination ({n_combinations} combos)...")
    print(f"Total target: ~{rows_per_combo * n_combinations} rows")
    
    for phase in phases:
        for comp_name, param_defs in all_component_groups.items():
            print(f"  Generating: {phase} / {comp_name}...")
            
            n = rows_per_combo
            
            # Generate measured values and rejection values for each parameter
            measured_values = {}
            rejection_values = {}
            
            for pname, rmin, rmax in param_defs:
                measured_values[pname] = generate_value_for_param(rmin, rmax, n)
                rejection_values[pname] = generate_value_for_param(rmin, rmax, n)
            
            # Generate categorical inputs
            etendue_vals = generate_categorical(ETENDUE_OPTIONS, n)
            duree_vals = generate_categorical(DUREE_OPTIONS, n)
            
            # Compute outputs for each row
            for i in range(n):
                row = {
                    "phase": phase,
                    "component": comp_name,
                }
                
                # Build parameter inputs
                params = []
                for pname, rmin, rmax in param_defs:
                    mv = measured_values[pname][i]
                    rv = rejection_values[pname][i]
                    
                    # Store inputs
                    row[f"measured_{pname}"] = mv
                    row[f"rejection_{pname}"] = rv
                    row[f"score_measured_{pname}"] = score_parameter(mv, rmin, rmax)
                    row[f"score_rejection_{pname}"] = score_parameter(rv, rmin, rmax)
                    row[f"range_min_{pname}"] = rmin
                    row[f"range_max_{pname}"] = rmax
                    
                    params.append(ParameterInput(pname, "", rmin, rmax, mv, rv))
                
                row["etendue"] = etendue_vals[i]
                row["duree"] = duree_vals[i]
                
                # Compute using calculator
                group = ComponentGroup(comp_name, params, etendue_vals[i], duree_vals[i])
                result = calculate_standard_component(group)
                
                # Store all outputs
                row["avg_initial_score"] = result["avg_initial"]
                row["avg_combined_score"] = result["avg_combined"]
                row["avg_rejection_score"] = result["avg_rejection"]
                row["value_initial"] = result["value_initial"]
                row["impact_apprehende"] = result["impact_apprehende"]
                row["sensitivity"] = result["sensitivity"]
                row["extent_score"] = result["extent_score"]
                row["intensity_class"] = result["intensity_class"]
                row["importance"] = result["importance"]
                row["importance_relative"] = result["importance_relative"]
                
                rows.append(row)
    
    # Also generate biological component data
    print("\n  Generating biological data...")
    bio_rows = generate_biological_dataset(rows_per_combo * len(phases))
    rows.extend(bio_rows)
    
    # Generate socio-economic data
    print("  Generating socio-economic data...")
    socio_rows = generate_socioeconomic_dataset(rows_per_combo * len(phases))
    rows.extend(socio_rows)
    
    # Generate paysage data
    print("  Generating paysage data...")
    paysage_rows = generate_paysage_dataset(rows_per_combo * len(phases))
    rows.extend(paysage_rows)
    
    df = pd.DataFrame(rows)
    
    print(f"\n{'='*60}")
    print(f"Dataset generated: {len(df)} rows × {len(df.columns)} columns")
    print(f"{'='*60}")
    
    return df


def generate_biological_dataset(n_rows: int) -> list:
    """Generate biological milieu dataset (Flore + Faune)."""
    rows = []
    n_per_type = n_rows // 2
    
    for bio_type in ["Flore", "Faune"]:
        for i in range(n_per_type):
            # Each assessment has 2-5 species
            n_species = np.random.randint(2, 6)
            
            species_list = []
            total_scores = []
            
            for s in range(n_species):
                presence = np.random.choice(PRESENCE_OPTIONS)
                protection = np.random.choice(PROTECTION_OPTIONS)
                species_status = np.random.choice(SPECIES_OPTIONS)
                
                species_list.append(BiologicalInput(
                    f"species_{s}", presence, protection, species_status
                ))
                
                # Compute individual score
                h = score_presence(presence)
                j = score_protection(protection)
                l = score_species(species_status)
                total_scores.append(h + j + l)
            
            result = calculate_biological_component(species_list, bio_type)
            
            row = {
                "phase": np.random.choice(["Realisation", "Exploitation", "Demantelement"]),
                "component": bio_type,
                "n_species": n_species,
                "avg_presence_score": np.mean([score_presence(sp.presence) for sp in species_list]),
                "avg_protection_score": np.mean([score_protection(sp.protection_status) for sp in species_list]),
                "avg_species_score": np.mean([score_species(sp.species_status) for sp in species_list]),
                "total_bio_score": result["avg_score"],
                "importance_relative": result["final_importance"],
            }
            
            # Store individual species scores for first 5
            for s in range(min(5, n_species)):
                row[f"species_{s}_presence"] = score_presence(species_list[s].presence)
                row[f"species_{s}_protection"] = score_protection(species_list[s].protection_status)
                row[f"species_{s}_vulnerability"] = score_species(species_list[s].species_status)
                row[f"species_{s}_total"] = total_scores[s]
            
            rows.append(row)
    
    return rows


def generate_socioeconomic_dataset(n_rows: int) -> list:
    """Generate socio-economic dataset (Infrastructure + Employment)."""
    rows = []
    n_per = n_rows // 2
    
    # Infrastructure
    for i in range(n_per):
        capacity = np.random.uniform(50, 100)  # 50%-100%
        etendue = np.random.choice(ETENDUE_OPTIONS)
        duree = np.random.choice(DUREE_OPTIONS)
        
        result = calculate_infrastructure(capacity, etendue, duree)
        
        rows.append({
            "phase": np.random.choice(["Realisation", "Exploitation", "Demantelement"]),
            "component": "Infrastructure",
            "capacity_pct": round(capacity, 2),
            "etendue": etendue,
            "duree": duree,
            "importance_relative": result["importance_relative"],
        })
    
    # Employment
    for i in range(n_per):
        jobs = np.random.uniform(0, 15)
        etendue = np.random.choice(ETENDUE_OPTIONS)
        duree = np.random.choice(DUREE_OPTIONS)
        
        result = calculate_employment(jobs, etendue, duree)
        
        rows.append({
            "phase": np.random.choice(["Realisation", "Exploitation", "Demantelement"]),
            "component": "Emploi",
            "jobs_created": round(jobs, 2),
            "etendue": etendue,
            "duree": duree,
            "importance_relative": result["importance_relative"],
        })
    
    return rows


def generate_paysage_dataset(n_rows: int) -> list:
    """Generate paysage (landscape) dataset."""
    rows = []
    for i in range(n_rows):
        modification = np.random.choice(["oui", "non"])
        result = calculate_paysage(modification)
        
        rows.append({
            "phase": "PRE_construction",
            "component": "Paysage",
            "modification_relief": modification,
            "importance_relative": result["importance_relative"],
        })
    
    return rows


# ============================================================================
# SIMPLIFIED FLAT DATASET (better for ML)
# ============================================================================

def generate_flat_dataset(n_rows: int) -> pd.DataFrame:
    """
    Generate a FLAT dataset optimized for ML training.
    
    Instead of variable-length parameter lists, this creates fixed columns:
    - One row = one complete assessment for one component in one phase
    - All parameter scores are flattened into columns
    - Target variable: importance_relative
    
    This is the recommended format for Kaggle.
    """
    all_component_groups = {
        "Eau": PARAMS_EAU,
        "Sol": PARAMS_SOL,
        "Air": PARAMS_AIR,
        "Population": PARAMS_POPULATION,
        "Sante": PARAMS_SANTE,
    }
    
    phases = ["PRE_construction", "Realisation", "Exploitation", "Demantelement"]
    
    rows = []
    rows_per_combo = max(1, n_rows // (len(phases) * len(all_component_groups)))
    
    print(f"Generating flat dataset: {rows_per_combo} rows × "
          f"{len(phases)} phases × {len(all_component_groups)} components")
    
    for phase in phases:
        for comp_name, param_defs in all_component_groups.items():
            n = rows_per_combo
            
            for i in range(n):
                row = {"phase": phase, "component": comp_name}
                
                params = []
                all_measured_scores = []
                all_rejection_scores = []
                all_combined_scores = []
                
                for pname, rmin, rmax in param_defs:
                    mv = generate_value_for_param(rmin, rmax, 1)[0]
                    rv = generate_value_for_param(rmin, rmax, 1)[0]
                    
                    ms = score_parameter(mv, rmin, rmax)
                    rs = score_parameter(rv, rmin, rmax)
                    
                    # Combined value (weighted average)
                    cv = (mv * 1000 + rv * 100) / 1100
                    cs = score_parameter(cv, rmin, rmax)
                    
                    all_measured_scores.append(ms)
                    all_rejection_scores.append(rs)
                    all_combined_scores.append(cs)
                    
                    row[f"score_m_{pname}"] = ms
                    row[f"score_r_{pname}"] = rs
                    
                    params.append(ParameterInput(pname, "", rmin, rmax, mv, rv))
                
                etendue = np.random.choice(ETENDUE_OPTIONS)
                duree = np.random.choice(DUREE_OPTIONS)
                
                row["etendue"] = etendue
                row["duree"] = duree
                
                # Aggregate scores (these are the most important features)
                row["avg_score_measured"] = round(np.mean(all_measured_scores), 6)
                row["avg_score_rejection"] = round(np.mean(all_rejection_scores), 6)
                row["avg_score_combined"] = round(np.mean(all_combined_scores), 6)
                row["max_score_measured"] = max(all_measured_scores)
                row["max_score_rejection"] = max(all_rejection_scores)
                row["count_score2_measured"] = all_measured_scores.count(2)
                row["count_score2_rejection"] = all_rejection_scores.count(2)
                row["count_score0_measured"] = all_measured_scores.count(0)
                row["count_score0_rejection"] = all_rejection_scores.count(0)
                row["n_params"] = len(param_defs)
                
                # Compute outputs
                group = ComponentGroup(comp_name, params, etendue, duree)
                result = calculate_standard_component(group)
                
                # Intermediate outputs (useful for multi-output prediction)
                row["value_initial"] = result["value_initial"]
                row["impact_apprehende"] = result["impact_apprehende"]
                row["sensitivity"] = result["sensitivity"]
                row["intensity_class"] = result["intensity_class"]
                row["importance"] = result["importance"]
                
                # TARGET
                row["importance_relative"] = result["importance_relative"]
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    
    print(f"\nFlat dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"Target distribution:")
    print(df["importance_relative"].value_counts().to_string())
    
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate EIE training dataset")
    parser.add_argument("--rows", type=int, default=100000,
                        help="Number of rows to generate (default: 100000)")
    parser.add_argument("--output", type=str, default="eie_dataset.csv",
                        help="Output CSV filename (default: eie_dataset.csv)")
    parser.add_argument("--format", choices=["full", "flat"], default="flat",
                        help="Dataset format: 'full' (all raw values) or 'flat' (ML-ready scores)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print(f"EIE Dataset Generator")
    print(f"{'='*60}")
    print(f"Rows: {args.rows}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")
    
    if args.format == "full":
        df = generate_dataset(args.rows)
    else:
        df = generate_flat_dataset(args.rows)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nTarget column: importance_relative")
    print(f"\nClass distribution:")
    print(df["importance_relative"].value_counts().to_string())
    print(f"\nComponent distribution:")
    print(df["component"].value_counts().to_string())
    print(f"\nPhase distribution:")
    print(df["phase"].value_counts().to_string())
    
    # Print first few rows
    print(f"\nSample rows:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
