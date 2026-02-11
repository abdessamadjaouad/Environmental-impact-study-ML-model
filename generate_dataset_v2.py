"""
EIE Dataset Generator v2 — Clean, ML-Ready Format
===================================================
Generates one CSV per component group (Eau, Sol, Air, etc.) with NO NaN columns.
Also generates a unified dataset with normalized aggregate features.

Each row = one impact assessment for one component in one phase.

Usage:
    python generate_dataset_v2.py --rows 100000
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eie_calculator import (
    ParameterInput, ComponentGroup,
    calculate_standard_component, score_parameter,
    score_extent, classify_value, classify_intensity,
    compute_sensitivity, compute_importance, compute_importance_relative,
)

np.random.seed(42)


# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

PARAMS = {
    "Eau": [
        ("temperature", 10, 30), ("Ph", 6.5, 8.5), ("Turbidite", 0, 5),
        ("Conductivite", 0, 2000), ("DBO5", 0, 5), ("DCO", 0, 25),
        ("Oxygene_dissous", 0, 5), ("Nitrates", 0, 50), ("Nitrites", 0, 0.1),
        ("Ammoniac", 0, 0.5), ("Phosphore_total", 0, 0.053),
        ("Azote_total", 0, 0.1), ("Plomb", 0, 0.01), ("Cadmium", 0, 0.003),
        ("Chrome", 0, 0.05), ("Cuivre", 0, 2), ("Zinc", 0, 3),
        ("Nickel", 0, 0.02), ("Mercure", 0, 0.001), ("Arsenic", 0, 0.01),
        ("Hydrocarbures", 0, 0.05),
    ],
    "Sol": [
        ("pH_sol", 5.5, 7.5), ("Permeabilite", 1e-6, 0.001),
        ("Matiere_organique", 2, 10), ("Carbone_organique", 1, 6),
        ("Plomb_sol", 50, 300), ("Cadmium_sol", 1, 3),
        ("Mercure_sol", 0.5, 2), ("Arsenic_sol", 10, 50),
        ("Chrome_sol", 50, 200), ("Cuivre_sol", 50, 140),
        ("Zinc_sol", 150, 300), ("Nickel_sol", 30, 75),
        ("Azote_total_sol", 1000, 5000), ("Phosphore_total_sol", 200, 3000),
    ],
    "Air": [
        ("Poussieres_totales", 0, 0.23), ("PM10", 0, 50),
        ("PM2_5", 0, 25), ("SO2", 0, 120), ("NOx", 0, 200),
        ("CO", 0, 10000), ("O3_ozone", 0, 120),
    ],
    "Population": [
        ("Radiation_eoliennes", 0, 5), ("Radiation_cables", 0, 5),
        ("Radiation_onduleurs", 0, 61), ("Qualite_vie", 0, 5),
    ],
    "Sante": [
        ("Poussieres_sante", 0, 0.15), ("Risques_electriques", 0, 2),
        ("Sante_securite", 0, 2),
    ],
}

ETENDUE_OPTIONS = ["Ponctuelle", "Locale", "Régionale", "Nationale"]
DUREE_OPTIONS = ["Courte", "Moyenne", "Longue"]
PHASES = ["PRE_construction", "Realisation", "Exploitation", "Demantelement"]


# ============================================================================
# VALUE GENERATION
# ============================================================================

def gen_values(rmin, rmax, n):
    """Generate values spanning all 3 scoring zones with balanced distribution."""
    rng = rmax - rmin if rmax > rmin else max(rmax, 1e-6)
    
    n0 = int(n * 0.4)  # Zone 0: in range
    n1 = int(n * 0.3)  # Zone 1: slight deviation
    n2 = n - n0 - n1   # Zone 2: large deviation
    
    vals = np.empty(n)
    
    # Zone 0: within [rmin, rmax]
    vals[:n0] = np.random.uniform(rmin, rmax, n0)
    
    # Zone 1: within 20% tolerance
    for i in range(n0, n0 + n1):
        if np.random.random() < 0.5 and rmin > 0:
            vals[i] = np.random.uniform(rmin * 0.8, rmin)
        else:
            vals[i] = np.random.uniform(rmax, rmax * 1.2)
    
    # Zone 2: far outside
    for i in range(n0 + n1, n):
        if np.random.random() < 0.5 and rmin > 0:
            vals[i] = np.random.uniform(0, rmin * 0.8)
        else:
            vals[i] = np.random.uniform(rmax * 1.2, rmax * 3)
    
    if rmin >= 0:
        vals = np.maximum(vals, 0)
    
    np.random.shuffle(vals)
    return np.round(vals, 6)


# ============================================================================
# DATASET 1: Per-Component CSVs (No NaN)
# ============================================================================

def generate_per_component(comp_name, param_defs, n_rows):
    """Generate dataset for one component. Clean columns, no NaN."""
    rows = []
    
    # Pre-generate all values (vectorized for speed)
    measured = {p[0]: gen_values(p[1], p[2], n_rows) for p in param_defs}
    rejection = {p[0]: gen_values(p[1], p[2], n_rows) for p in param_defs}
    etendues = np.random.choice(ETENDUE_OPTIONS, n_rows)
    durees = np.random.choice(DUREE_OPTIONS, n_rows)
    phases = np.random.choice(PHASES, n_rows)
    
    for i in range(n_rows):
        row = {"phase": phases[i]}
        
        params = []
        m_scores = []
        r_scores = []
        
        for pname, rmin, rmax in param_defs:
            mv = measured[pname][i]
            rv = rejection[pname][i]
            ms = score_parameter(mv, rmin, rmax)
            rs = score_parameter(rv, rmin, rmax)
            
            row[f"score_m_{pname}"] = ms
            row[f"score_r_{pname}"] = rs
            m_scores.append(ms)
            r_scores.append(rs)
            
            params.append(ParameterInput(pname, "", rmin, rmax, mv, rv))
        
        row["etendue"] = etendues[i]
        row["duree"] = durees[i]
        
        # Aggregate features
        row["avg_score_m"] = round(np.mean(m_scores), 6)
        row["avg_score_r"] = round(np.mean(r_scores), 6)
        row["max_score_m"] = max(m_scores)
        row["max_score_r"] = max(r_scores)
        row["pct_score2_m"] = round(m_scores.count(2) / len(m_scores), 4)
        row["pct_score2_r"] = round(r_scores.count(2) / len(r_scores), 4)
        row["pct_score0_m"] = round(m_scores.count(0) / len(m_scores), 4)
        row["pct_score0_r"] = round(r_scores.count(0) / len(r_scores), 4)
        row["n_params"] = len(param_defs)
        
        # Compute output via calculator
        group = ComponentGroup(comp_name, params, etendues[i], durees[i])
        result = calculate_standard_component(group)
        
        # Intermediate outputs (can serve as multi-task targets)
        row["value_initial"] = result["value_initial"]
        row["impact_apprehende"] = result["impact_apprehende"]
        row["sensitivity"] = result["sensitivity"]
        row["intensity_class"] = result["intensity_class"]
        row["importance"] = result["importance"]
        
        # MAIN TARGET
        row["importance_relative"] = result["importance_relative"]
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# DATASET 2: Unified Normalized (All components, no NaN)
# ============================================================================

def generate_unified(n_rows):
    """
    Generate a UNIFIED dataset where every row has the same columns.
    
    Instead of per-parameter scores (which cause NaN across components),
    we use aggregate statistics as features. This is cleaner for ML.
    
    Features:
    - phase (categorical)
    - component (categorical)
    - avg/max/pct scores for measured and rejection values
    - etendue, duree (categorical)
    - n_params (numerical)
    
    Targets:
    - value_initial, impact_apprehende, sensitivity, importance, importance_relative
    """
    rows_per_comp = n_rows // len(PARAMS)
    all_dfs = []
    
    for comp_name, param_defs in PARAMS.items():
        print(f"  Generating {comp_name}: {rows_per_comp} rows...")
        df = generate_per_component(comp_name, param_defs, rows_per_comp)
        
        # For unified: keep only aggregate features + targets
        keep_cols = [
            "phase", "etendue", "duree",
            "avg_score_m", "avg_score_r",
            "max_score_m", "max_score_r",
            "pct_score2_m", "pct_score2_r",
            "pct_score0_m", "pct_score0_r",
            "n_params",
            "value_initial", "impact_apprehende", "sensitivity",
            "intensity_class", "importance", "importance_relative",
        ]
        unified_df = df[keep_cols].copy()
        unified_df.insert(1, "component", comp_name)
        all_dfs.append(unified_df)
    
    unified = pd.concat(all_dfs, ignore_index=True)
    # Shuffle
    unified = unified.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return unified


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate EIE dataset v2")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 60)
    print("EIE DATASET GENERATOR v2")
    print("=" * 60)
    
    # === 1. Generate per-component datasets ===
    print("\n--- Per-Component Datasets ---")
    rows_per_comp = args.rows // len(PARAMS)
    
    for comp_name, param_defs in PARAMS.items():
        print(f"\n  {comp_name} ({rows_per_comp} rows)...")
        df = generate_per_component(comp_name, param_defs, rows_per_comp)
        
        fname = f"eie_{comp_name.lower()}.csv"
        fpath = os.path.join(output_dir, fname)
        df.to_csv(fpath, index=False)
        
        print(f"    ✅ {fname}: {df.shape} | {os.path.getsize(fpath)/1024:.0f} KB")
        print(f"    Target distribution: {dict(df['importance_relative'].value_counts())}")
    
    # === 2. Generate unified dataset ===
    print("\n--- Unified Dataset ---")
    unified = generate_unified(args.rows)
    
    uf_path = os.path.join(output_dir, "eie_unified.csv")
    unified.to_csv(uf_path, index=False)
    
    print(f"\n  ✅ eie_unified.csv: {unified.shape}")
    print(f"     Size: {os.path.getsize(uf_path)/1024:.0f} KB")
    print(f"     Columns: {list(unified.columns)}")
    print(f"\n  Target distribution:")
    print(f"  {dict(unified['importance_relative'].value_counts())}")
    print(f"\n  No NaN columns: {unified.isna().sum().sum() == 0}")
    
    print(f"\n{'=' * 60}")
    print("✅ ALL DATASETS GENERATED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
