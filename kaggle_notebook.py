#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🌍 Environmental Impact Assessment (EIE) — ML Prediction Model
# 
# **Objective**: Predict the **relative importance** of environmental impacts 
# (`Mineure`, `Moyenne`, `Majeure`) from environmental measurement scores, 
# geographic extent, and duration — replicating an Excel-based EIA macro used in Morocco.
#
# **Project Structure:**
# 1. **Calculator**: Python reimplementation of all Excel formulas
# 2. **Dataset Generation**: 100K rows of synthetic training data  
# 3. **Model Training**: 4 classifiers compared (DT, RF, XGBoost, MLP)
# 4. **Results**: Feature importance & model comparison
#
# ---

# %% [markdown]
# ## 📦 Setup & Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("✅ All imports successful")

# %% [markdown]
# ---
# ## 1️⃣ EIE Calculator — Formula Reimplementation
#
# These functions replicate the exact formulas from the Excel macro
# `Macro Standardiser-Levaluation EIE 2026`. Each function has been 
# verified against the Excel computed values.

# %%
# === SCORING FUNCTIONS ===

def score_parameter(value, range_min, range_max):
    """Score a measurement: 0=in range, 1=slightly outside (±20%), 2=far outside."""
    if range_min is None and range_max is None:
        return 0
    effective_min = range_min if range_min is not None else 0
    effective_max = range_max if range_max is not None else float('inf')
    
    if effective_min <= value <= effective_max:
        return 0
    
    tolerance_min = effective_min * 0.8 if effective_min != 0 else -float('inf')
    tolerance_max = effective_max * 1.2 if effective_max != float('inf') else float('inf')
    
    if tolerance_min <= value <= tolerance_max:
        return 1
    return 2


def classify_value(avg_score):
    """Classify average score: Faible (≤0.5), Moyenne (≤1), Forte (>1)."""
    if avg_score <= 0.5:
        return "Faible"
    elif avg_score <= 1:
        return "Moyenne"
    return "Forte"


def compute_sensitivity(impact_apprehende, initial_value):
    """3×3 matrix combining two classifications into sensitivity."""
    ia = impact_apprehende.upper()
    iv = initial_value.upper()
    matrix = {
        ("FORTE", "FORTE"): "FORTE", ("FORTE", "MOYENNE"): "FORTE", ("FORTE", "FAIBLE"): "MOYENNE",
        ("MOYENNE", "FORTE"): "FORTE", ("MOYENNE", "MOYENNE"): "MOYENNE", ("MOYENNE", "FAIBLE"): "FAIBLE",
        ("FAIBLE", "FORTE"): "MOYENNE", ("FAIBLE", "MOYENNE"): "FAIBLE", ("FAIBLE", "FAIBLE"): "FAIBLE",
    }
    return matrix.get((ia, iv), "FAIBLE")


def score_extent(etendue):
    """Convert extent to score: Ponctuelle=0, Locale=0.5, Régionale=1.25, Nationale=2."""
    return {"Ponctuelle": 0.0, "Locale": 0.5, "Régionale": 1.25, "Nationale": 2.0}.get(etendue, 0.0)


def classify_intensity(avg_score):
    """Classify intensity: ≤1=FAIBLE, exactly 2=MOYENNE, else=FORTE."""
    if avg_score <= 1:
        return "FAIBLE"
    elif avg_score == 2.0:
        return "MOYENNE"
    return "FORTE"


def compute_importance(sensitivity, intensity, etendue):
    """Compute importance from sensitivity × intensity × extent."""
    sens = sensitivity.upper() if sensitivity else ""
    intens = intensity.upper() if intensity else ""
    
    if sens == "ABSOLUE": return "Inadmissible"
    if sens == "FORTE":
        if intens == "FORTE": return "Majeure" if etendue in ("Nationale","Régionale","Locale") else "Moyenne"
        if intens == "MOYENNE": return "Majeure" if etendue in ("Nationale","Régionale") else "Moyenne"
        if intens == "FAIBLE": return "Majeure" if etendue in ("Nationale","Régionale") else "Mineure"
    if sens == "MOYENNE":
        if intens == "FORTE": return "Majeure" if etendue == "Nationale" else "Moyenne"
        if intens == "MOYENNE": return "Moyenne"
        if intens == "FAIBLE": return "Moyenne" if etendue in ("Nationale","Régionale") else "Mineure"
    if sens == "FAIBLE":
        if intens == "FORTE": return "Moyenne" if etendue in ("Nationale","Régionale") else "Mineure"
        return "Mineure"
    return "Mineure à nulle" if sens in ("TRÈS FAIBLE","TRES FAIBLE") and intens != "FORTE" else "Mineure"


def compute_importance_relative(importance, duree):
    """Adjust importance by duration."""
    if importance in ("Mineure", "Mineure à nulle"): return "Mineure"
    if importance == "Moyenne": return "Mineure" if duree.lower() == "courte" else "Moyenne"
    if importance == "Majeure": return "Moyenne" if duree.lower() == "courte" else "Majeure"
    if importance == "Inadmissible": return "Inadmissible"
    return ""


def calculate_standard_component(params, etendue, duree):
    """
    Full calculation chain for one environmental component.
    params: list of (measured_value, rejection_value, range_min, range_max)
    """
    initial_scores = [score_parameter(mv, rmin, rmax) for mv, rv, rmin, rmax in params]
    combined_scores = [score_parameter((mv*1000 + rv*100)/1100, rmin, rmax) for mv, rv, rmin, rmax in params]
    rejection_scores = [score_parameter(rv, rmin, rmax) for mv, rv, rmin, rmax in params]
    
    avg_initial = np.mean(initial_scores) if initial_scores else 0
    avg_combined = np.mean(combined_scores) if combined_scores else 0
    avg_rejection = np.mean(rejection_scores) if rejection_scores else 0
    
    value_initial = classify_value(avg_initial)
    impact_apprehende = classify_value(avg_combined)
    sensitivity = compute_sensitivity(impact_apprehende, value_initial)
    intensity = classify_intensity(avg_rejection)
    importance = compute_importance(sensitivity, intensity, etendue)
    importance_relative = compute_importance_relative(importance, duree)
    
    return {
        "avg_initial": round(avg_initial, 6),
        "avg_combined": round(avg_combined, 6),
        "avg_rejection": round(avg_rejection, 6),
        "value_initial": value_initial,
        "impact_apprehende": impact_apprehende,
        "sensitivity": sensitivity,
        "intensity_class": intensity,
        "importance": importance,
        "importance_relative": importance_relative,
    }

print("✅ Calculator functions defined")

# %% [markdown]
# ---
# ## 2️⃣ Dataset Generation
# 
# Generate 100K rows of synthetic EIA data by randomizing input measurements
# across the 3 scoring zones (within range, slightly outside, far outside).

# %%
# === LOAD DATASET ===
# Try loading from Kaggle first, otherwise generate it below
KAGGLE_PATH = "/kaggle/input/environmental-impact-assessment-eie-dataset/eie_unified.csv"
LOCAL_PATH = "eie_unified.csv"

DATASET_LOADED = False
if os.path.exists(KAGGLE_PATH):
    df = pd.read_csv(KAGGLE_PATH)
    DATASET_LOADED = True
    print(f"✅ Loaded dataset from Kaggle: {df.shape}")
elif os.path.exists(LOCAL_PATH):
    df = pd.read_csv(LOCAL_PATH)
    DATASET_LOADED = True
    print(f"✅ Loaded dataset from local file: {df.shape}")
else:
    print("📦 Dataset not found — will generate it below...")

if DATASET_LOADED:
    print(f"\nTarget distribution:")
    print(df["importance_relative"].value_counts())

# %%
# Parameter definitions: (name, range_min, range_max)
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


def gen_values(rmin, rmax, n):
    """Generate values spanning all 3 scoring zones."""
    n0, n1 = int(n * 0.4), int(n * 0.3)
    n2 = n - n0 - n1
    vals = np.empty(n)
    vals[:n0] = np.random.uniform(rmin, rmax, n0)
    for i in range(n0, n0 + n1):
        if np.random.random() < 0.5 and rmin > 0:
            vals[i] = np.random.uniform(rmin * 0.8, rmin)
        else:
            vals[i] = np.random.uniform(rmax, rmax * 1.2)
    for i in range(n0 + n1, n):
        if np.random.random() < 0.5 and rmin > 0:
            vals[i] = np.random.uniform(0, rmin * 0.8)
        else:
            vals[i] = np.random.uniform(rmax * 1.2, rmax * 3)
    if rmin >= 0:
        vals = np.maximum(vals, 0)
    np.random.shuffle(vals)
    return np.round(vals, 6)

print("✅ Parameter definitions and generators ready")

# %%
# Generate the dataset ONLY if not loaded from file
if not DATASET_LOADED:
    np.random.seed(42)
    N_ROWS = 100000
    rows_per_comp = N_ROWS // len(PARAMS)
    
    all_rows = []
    
    for comp_name, param_defs in PARAMS.items():
        print(f"  Generating {comp_name}: {rows_per_comp} rows...")
        
        measured = {p[0]: gen_values(p[1], p[2], rows_per_comp) for p in param_defs}
        rejection = {p[0]: gen_values(p[1], p[2], rows_per_comp) for p in param_defs}
        etendues = np.random.choice(ETENDUE_OPTIONS, rows_per_comp)
        durees = np.random.choice(DUREE_OPTIONS, rows_per_comp)
        phases = np.random.choice(PHASES, rows_per_comp)
        
        for i in range(rows_per_comp):
            row = {"phase": phases[i], "component": comp_name}
            
            params_tuples = []
            m_scores, r_scores = [], []
            
            for pname, rmin, rmax in param_defs:
                mv, rv = measured[pname][i], rejection[pname][i]
                ms = score_parameter(mv, rmin, rmax)
                rs = score_parameter(rv, rmin, rmax)
                m_scores.append(ms)
                r_scores.append(rs)
                row[f"score_m_{pname}"] = ms
                row[f"score_r_{pname}"] = rs
                params_tuples.append((mv, rv, rmin, rmax))
            
            row["etendue"] = etendues[i]
            row["duree"] = durees[i]
            row["avg_score_m"] = round(np.mean(m_scores), 6)
            row["avg_score_r"] = round(np.mean(r_scores), 6)
            row["max_score_m"] = max(m_scores)
            row["max_score_r"] = max(r_scores)
            row["pct_score2_m"] = round(m_scores.count(2) / len(m_scores), 4)
            row["pct_score2_r"] = round(r_scores.count(2) / len(r_scores), 4)
            row["pct_score0_m"] = round(m_scores.count(0) / len(m_scores), 4)
            row["pct_score0_r"] = round(r_scores.count(0) / len(r_scores), 4)
            row["n_params"] = len(param_defs)
            
            result = calculate_standard_component(params_tuples, etendues[i], durees[i])
            row["value_initial"] = result["value_initial"]
            row["impact_apprehende"] = result["impact_apprehende"]
            row["sensitivity"] = result["sensitivity"]
            row["intensity_class"] = result["intensity_class"]
            row["importance"] = result["importance"]
            row["importance_relative"] = result["importance_relative"]
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    print(f"\n✅ Dataset generated: {df.shape}")
    print(f"\nTarget distribution:")
    print(df["importance_relative"].value_counts())
else:
    print(f"✅ Using pre-loaded dataset: {df.shape}")

# %% [markdown]
# ### 📊 Data Exploration

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Target distribution
colors = {'Mineure': '#4CAF50', 'Moyenne': '#FF9800', 'Majeure': '#F44336'}
target_counts = df['importance_relative'].value_counts()
axes[0].bar(target_counts.index, target_counts.values, 
            color=[colors.get(x, '#999') for x in target_counts.index], edgecolor='white', linewidth=1.5)
axes[0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, (idx, val) in enumerate(target_counts.items()):
    axes[0].text(i, val + 500, f'{val:,}', ha='center', fontweight='bold')

# Component distribution
comp_counts = df['component'].value_counts()
axes[1].bar(comp_counts.index, comp_counts.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(comp_counts))))
axes[1].set_title('Samples per Component', fontsize=14, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

# Average scores by target
for target in ['Mineure', 'Moyenne', 'Majeure']:
    subset = df[df['importance_relative'] == target]
    axes[2].scatter(subset['avg_score_m'].values[:500], subset['avg_score_r'].values[:500], 
                    alpha=0.3, s=10, label=target, color=colors[target])
axes[2].set_xlabel('Average Measured Score')
axes[2].set_ylabel('Average Rejection Score')
axes[2].set_title('Score Distribution by Target', fontsize=14, fontweight='bold')
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 3️⃣ Feature Engineering & Model Training

# %%
# Prepare features and target
# Drop intermediate targets (they would leak the answer)
drop_cols = ['importance_relative', 'importance', 'sensitivity',
             'impact_apprehende', 'value_initial', 'intensity_class']

feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categoricals
le_encoders = {}
for col in feature_df.select_dtypes(include='object').columns:
    le_encoders[col] = LabelEncoder()
    feature_df[col] = le_encoders[col].fit_transform(feature_df[col].astype(str))

le_target = LabelEncoder()
y = le_target.fit_transform(df['importance_relative'])
X = feature_df.values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0)

feature_names = list(feature_df.columns)

print(f"Features: {len(feature_names)} columns")
print(f"Classes: {list(le_target.classes_)}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %%
# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=15, min_samples_split=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, 
                                           early_stopping=True, random_state=42),
}
if HAS_XGBOOST:
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                       eval_metric='mlogloss', n_jobs=-1, random_state=42, verbosity=0)

# Train and evaluate all models
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"🔄 Training: {name}")
    
    if "MLP" in name or "Neural" in name:
        scaler = StandardScaler()
        X_tr, X_te = scaler.fit_transform(X_train), scaler.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test
        scaler = None
    
    t0 = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - t0
    
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1, 'Train Time': f"{train_time:.2f}s"})
    trained_models[name] = {'model': model, 'scaler': scaler, 'accuracy': acc, 'f1': f1}
    
    print(f"  ✅ Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.2f}s")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_, digits=4))

# %% [markdown]
# ---
# ## 4️⃣ Results & Comparison

# %%
# Results table
results_df = pd.DataFrame(results)
print("\n📊 MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

# %%
# Model comparison chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

bars1 = ax.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', 
               color=colors_bar[:len(results_df)], alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, results_df['F1-Score'], width, label='F1-Score',
               color=colors_bar[:len(results_df)], alpha=0.5, edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title('🏆 Model Comparison — EIE Impact Classification', fontsize=14, fontweight='bold')
ax.set_ylim(0.9, 1.0)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Confusion matrices
fig, axes = plt.subplots(1, len(trained_models), figsize=(5*len(trained_models), 4))
if len(trained_models) == 1:
    axes = [axes]

for ax, (name, data) in zip(axes, trained_models.items()):
    model = data['model']
    if data['scaler']:
        y_pred = model.predict(data['scaler'].transform(X_test))
    else:
        y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le_target.classes_, yticklabels=le_target.classes_)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5️⃣ Feature Importance

# %%
fig, axes = plt.subplots(1, sum(1 for m in trained_models.values() if hasattr(m['model'], 'feature_importances_')),
                          figsize=(8, 5))
if not hasattr(axes, '__len__'):
    axes = [axes]

ax_idx = 0
for name, data in trained_models.items():
    model = data['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]
        
        colors_fi = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_names)))
        axes[ax_idx].barh(range(len(top_names)), top_vals[::-1], color=colors_fi)
        axes[ax_idx].set_yticks(range(len(top_names)))
        axes[ax_idx].set_yticklabels(top_names[::-1], fontsize=9)
        axes[ax_idx].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[ax_idx].set_xlabel('Importance')
        ax_idx += 1

plt.suptitle('🔍 Feature Importance — Top 15', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Print top features from best tree-based model
best_tree = None
for name in ["Random Forest", "XGBoost", "Decision Tree"]:
    if name in trained_models and hasattr(trained_models[name]['model'], 'feature_importances_'):
        best_tree = name
        break

if best_tree:
    importances = trained_models[best_tree]['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\n🔍 Top Features ({best_tree}):")
    for i, idx in enumerate(indices[:10]):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f} ({importances[idx]*100:.1f}%)")

# %% [markdown]
# ---
# ## 6️⃣ Cross-Validation

# %%
print("Running 5-fold cross-validation...")
cv_results = []

for name, data in trained_models.items():
    print(f"  {name}...", end=" ")
    model_fresh = type(data['model'])(**data['model'].get_params())
    
    if data['scaler']:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scores = cross_val_score(model_fresh, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1)
    else:
        scores = cross_val_score(model_fresh, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    
    cv_results.append({'Model': name, 'CV Mean': f"{scores.mean():.4f}", 
                        'CV Std': f"±{scores.std():.4f}"})
    print(f"Mean: {scores.mean():.4f} ±{scores.std():.4f}")

cv_df = pd.DataFrame(cv_results)
print(f"\n{cv_df.to_string(index=False)}")

# %% [markdown]
# ---
# ## 🏆 Conclusion
# 
# | Finding | Detail |
# |---------|--------|
# | **Best Model** | Neural Network (MLP) — 96.25% accuracy |
# | **All models** | >95.9% accuracy — the problem is well-structured |
# | **#1 Feature** | `etendue` (geographic extent) — ~47% importance |
# | **#2 Feature** | `avg_score_m` (measured value scores) — ~22% |
# | **#3 Feature** | `duree` (duration) — ~18% |
# 
# The environmental impact classification is highly predictable from the 
# measurement scores and categorical inputs. Geographic extent is the single
# most important factor in determining impact severity.

# %%
# Save the dataset for others to use
df.to_csv("eie_dataset_output.csv", index=False)
print(f"💾 Dataset saved: eie_dataset_output.csv ({len(df)} rows)")

best_name = max(trained_models, key=lambda k: trained_models[k]['f1'])
print(f"\n🏆 Best model: {best_name} (F1={trained_models[best_name]['f1']:.4f})")
